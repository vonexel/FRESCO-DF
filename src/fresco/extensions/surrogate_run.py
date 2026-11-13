from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from fresco.recompute_curves import saved_logits
from fresco.research_audit import paired_gap_uncertainty
from fresco.training import predict
from fresco.utils import atomic_json
from .analysis_store import create_analysis_run
from .inventory import sha256
from .paired import response_metrics
from .preflight import HEADS, SEEDS
from .runtime import FrozenInputs, digest_id
from .source_pairs import load_source_pairs, transform_source_pairs


METHODS = ("zero", "ridge", "gam", "ebm", "ebm_interactions")


def _teacher_distribution(frame, head, seed):
    result = frame.groupby(["dataset", "condition", "label"], observed=True, as_index=False).agg(
        pairs=("total", "size"), clean_logit_mean=("clean_teacher_logit", "mean"),
        clean_logit_median=("clean_teacher_logit", "median"),
        changed_logit_mean=("changed_teacher_logit", "mean"),
        changed_logit_median=("changed_teacher_logit", "median"),
        changed_logit_q05=("changed_teacher_logit", lambda x: x.quantile(.05)),
        changed_logit_q95=("changed_teacher_logit", lambda x: x.quantile(.95)))
    result["head"], result["seed"] = head, seed
    return result


def ebm_environment(executable):
    if not executable or not Path(executable).is_file():
        raise FileNotFoundError("BLOCKED_DEPENDENCY: supply the separate InterpretML Python via --ebm-python")
    code = ("import json,sys; from importlib.metadata import version; "
            "from interpret.glassbox import ExplainableBoostingRegressor; "
            "print(json.dumps({k:version(k) for k in ['interpret-core','numpy','pandas','scikit-learn']}))")
    result = subprocess.run([str(executable), "-c", code], capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError("BLOCKED_DEPENDENCY: separate EBM import failed: " + result.stderr[-2000:])
    versions = json.loads(result.stdout.strip())
    expected = {"interpret-core": "0.7.8", "numpy": "2.4.6", "pandas": "2.3.3", "scikit-learn": "1.9.1"}
    if versions != expected:
        raise ValueError(f"BLOCKED_DEPENDENCY: EBM environment differs from constraints: {versions}")
    return versions


def _surrogate_video(frame):
    temp = frame.copy()
    temp["total_sse"] = temp.total**2
    for name in METHODS:
        temp[name+"_error_sse"] = (temp[name]-temp.total)**2
    grouped = temp.groupby(["dataset", "condition", "label", "video_hash", "group_hash"],
                           sort=True, observed=True, as_index=False)
    aggregation = {"pairs": ("total", "size"), "total_sse": ("total_sse", "sum"),
                   "total_mean": ("total", "mean")}
    aggregation.update({name+"_error_sse": (name+"_error_sse", "sum") for name in METHODS})
    aggregation.update({name+"_mean": (name, "mean") for name in METHODS})
    return grouped.agg(**aggregation)


def _surrogate_metrics(frame, head, seed, epsilon):
    records = []
    for dataset, data in frame.groupby("dataset", observed=True, sort=True):
        populations = [("all", "all", data)]
        populations += [(condition, "all", part) for condition, part in data.groupby("condition", observed=True)]
        populations += [("all", int(label), part) for label, part in data.groupby("label", observed=True)]
        for condition, label, part in populations:
            y = part.total.to_numpy()
            for method in METHODS:
                p = part[method].to_numpy()
                metric = response_metrics(y, p, part.video_hash, sign_tolerance=epsilon)
                if len(y) > 1 and np.std(y) > 0 and np.std(p) > 0:
                    metric["pearson_frame_weighted"] = float(pearsonr(y, p).statistic)
                    metric["spearman_frame_weighted"] = float(spearmanr(y, p).statistic)
                else:
                    metric["pearson_frame_weighted"] = metric["spearman_frame_weighted"] = None
                records.append({"head": head, "seed": seed, "dataset": dataset, "condition": condition,
                                "label": label, "method": method, **metric})
    return pd.DataFrame(records)


def _cluster_matrix(tables, head, dataset):
    reference, numerator, denominator = None, {m: [] for m in METHODS}, []
    for seed in SEEDS:
        data = tables[(head, seed)]
        columns = ["pairs", "total_sse"]+[m+"_error_sse" for m in METHODS]
        video = data.loc[data.dataset == dataset].groupby(["video_hash", "group_hash"], observed=True,
                                                         sort=True)[columns].sum()
        if reference is None:
            reference = video.index
        if not video.index.equals(reference):
            raise ValueError("Surrogate evaluation video groups differ between seeds")
        n = video.pairs.to_numpy(float)
        denominator.append(video.total_sse.to_numpy(float)/n)
        for method in METHODS:
            numerator[method].append(video[method+"_error_sse"].to_numpy(float)/n)
    return {"groups": reference.get_level_values("group_hash").to_numpy(), "videos": len(reference),
            "total": np.stack(denominator), "error": {m: np.stack(v) for m, v in numerator.items()}}


def _transfer(source, target, method, resamples, seed):
    rng = np.random.default_rng(seed)
    seed_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    estimates, distributions = [], []
    for pop in (source, target):
        _, inverse = np.unique(pop["groups"], return_inverse=True)
        n_group = int(inverse.max()+1)
        if n_group < 5:
            return {"status": "INCONCLUSIVE", "reason": "fewer than five clusters"}
        e, t = pop["error"][method], pop["total"]
        estimates.append(np.sqrt(e.mean(1)/t.mean(1)))
        egroup = np.stack([np.bincount(inverse, weights=row, minlength=n_group) for row in e])
        tgroup = np.stack([np.bincount(inverse, weights=row, minlength=n_group) for row in t])
        counts = np.stack([np.bincount(rng.integers(n_group, size=n_group), minlength=n_group)
                           for _ in range(resamples)])
        distributions.append(np.sqrt((egroup @ counts.T)/(tgroup @ counts.T)))
    observed = estimates[1]-estimates[0]
    delta = distributions[1]-distributions[0]
    draws = seed_rng.integers(10, size=(resamples, 10))
    crossed = delta[draws.T, np.arange(resamples)].mean(0)
    fixed = delta.mean(0)
    if not np.isfinite(crossed).all() or not np.isfinite(fixed).all():
        return {"status": "INCONCLUSIVE", "reason": "near-zero denominator in bootstrap draw"}
    return {"status": "COMPUTED", "difference": float(observed.mean()), "seed_differences": observed.tolist(),
            "fixed_cluster_ci95": np.quantile(fixed, [.025, .975]).tolist(),
            "crossed_seed_cluster_ci95": np.quantile(crossed, [.025, .975]).tolist(),
            "scope": "independent dataset-cluster resampling; paired teacher seeds"}


def _summary(tables, resamples, seed):
    result, fidelity, transfer = {}, [], []
    for head in HEADS:
        pops = {d: _cluster_matrix(tables, head, d) for d in ("ffpp", "celebdfpp")}
        for dataset, pop in pops.items():
            opts = {"resamples": resamples, "seed": seed, "min_total_rms": 1e-8}
            reports = {}
            for method in METHODS:
                reports[method+"_minus_zero"] = paired_gap_uncertainty(
                    pop["error"][method], pop["total"], pop["total"], pop["total"], pop["groups"], **opts)
                if method in ("gam", "ebm"):
                    reports[method+"_minus_ridge"] = paired_gap_uncertainty(
                        pop["error"][method], pop["total"], pop["error"]["ridge"], pop["total"],
                        pop["groups"], **opts)
                item = reports[method+"_minus_zero"]
                ci = item.get("crossed_seeds_clusters", {}).get("ci95")
                fidelity.append({"head": head, "dataset": dataset, "method": method,
                                 "G": item.get("difference", np.nan)+1,
                                 "ci_low": ci[0]+1 if ci else np.nan, "ci_high": ci[1]+1 if ci else np.nan,
                                 "videos": pop["videos"], "seeds": 10})
            result[f"{head}/{dataset}"] = reports
        for method in METHODS:
            t = _transfer(pops["ffpp"], pops["celebdfpp"], method, resamples, seed)
            transfer.append({"head": head, "method": method, "difference": t.get("difference"),
                             "ci_low": (t.get("crossed_seed_cluster_ci95") or [None, None])[0],
                             "ci_high": (t.get("crossed_seed_cluster_ci95") or [None, None])[1],
                             "source_videos": pops["ffpp"]["videos"], "target_videos": pops["celebdfpp"]["videos"],
                             "status": t["status"]})
            result[f"{head}/transfer/{method}"] = t
    return result, pd.DataFrame(fidelity), pd.DataFrame(transfer)


def run_surrogates(cfg, cache, main, replay, source_cache, ebm_python, run_id, device="cpu"):
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; select cpu")
    inputs = FrozenInputs(cache, main, replay)
    versions = ebm_environment(ebm_python)
    if source_cache is None:
        raise FileNotFoundError("BLOCKED_INPUT: separately cached FF++ source transformed pairs are required")
    source = load_source_pairs(inputs, source_cache)
    out, signature = create_analysis_run(cfg, inputs, run_id, "surrogates", extra={
        "source_cache_manifest_sha256": source["manifest_sha256"], "ebm_versions": versions, "device": device})
    statuses_path = out / "seed_status.json"
    statuses = (json.loads(statuses_path.read_text(encoding="utf-8")) if statuses_path.is_file() else
                {f"{h}/seed{s}": {"status": "PENDING"} for h in HEADS for s in SEEDS})
    atomic_json(statuses_path, statuses)
    srcmeta = source["metadata"]
    clean_x, changed_x, source_quality = transform_source_pairs(inputs, source)
    train, val = (srcmeta.split == "train").to_numpy(), (srcmeta.split == "val").to_numpy()
    pairs = inputs.pairs()
    test_quality = np.column_stack((inputs.quality[pairs.clean_row.to_numpy()],
                                    inputs.quality[pairs.changed_row.to_numpy()])).astype(np.float32)
    pairs["video_hash"] = pairs.video_id.astype(str).map(lambda x: digest_id("video", x))
    pairs["group_hash"] = pairs.group_id.astype(str).map(lambda x: digest_id("group", x))
    pairs["pair_hash"] = [digest_id(d, v, int(f), c) for d, v, f, c in zip(
        pairs.dataset.astype(str), pairs.video_id.astype(str), pairs.frame_idx, pairs.condition.astype(str))]
    pairs[["dataset", "condition", "label", "video_hash", "group_hash", "pair_hash"]].to_csv(
        out / "evaluated_pair_ids.csv.gz", index=False, compression="gzip")
    source_ids = [digest_id(d, v, int(f), c) for d, v, f, c in zip(
        srcmeta.dataset.astype(str), srcmeta.video_id.astype(str), srcmeta.frame_idx, srcmeta.condition.astype(str))]
    pd.DataFrame({"split": srcmeta.split, "sample_sha256": source_ids}).to_csv(
        out / "source_pair_ids.csv.gz", index=False, compression="gzip")
    source_video = np.array([digest_id("video", v) for v in srcmeta.video_id.astype(str)])
    source_group = np.array([digest_id("group", v) for v in srcmeta.group_id.astype(str)])
    shared = out / "worker_inputs.npz"
    shared_marker = out / "worker_inputs.done.json"
    if shared.exists() or shared_marker.exists():
        marker = json.loads(shared_marker.read_text(encoding="utf-8"))
        if marker.get("signature") != signature or marker.get("sha256") != sha256(shared):
            raise ValueError("Saved surrogate worker inputs do not match their signature")
    else:
        np.savez(shared, train_x=source_quality[train], val_x=source_quality[val], test_x=test_quality,
                 train_video=source_video[train], train_group=source_group[train],
                 val_video=source_video[val], val_group=source_group[val])
        atomic_json(shared_marker, {"signature": signature, "sha256": sha256(shared)})
    tables, metrics_all, functions_all, agreement_all, teacher_distributions = {}, [], [], [], []
    environment = os.environ.copy()
    package_root = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = package_root + os.pathsep + environment.get("PYTHONPATH", "")
    for head in HEADS:
        for seed in SEEDS:
            status_key = f"{head}/seed{seed}"
            statuses[status_key] = {"status": "RUNNING"}
            atomic_json(statuses_path, statuses)
            dest = out / head / f"seed{seed}"
            dest.mkdir(parents=True, exist_ok=True)
            done = dest / "DONE.json"
            names = ["video.csv", "metrics.csv", "agreement.csv", "functions.csv", "worker.json", "predictions.npy",
                     "source_targets.npz", "worker.log", "teacher_distribution.csv"]
            if done.is_file():
                marker = json.loads(done.read_text(encoding="utf-8"))
                if marker.get("signature") != signature or any(sha256(dest/n) != marker["files"].get(n) for n in names):
                    raise ValueError(f"Corrupt or mismatched saved surrogate seed: {dest}")
                video, metrics, agreement = (pd.read_csv(dest/n) for n in names[:3])
                functions = pd.read_csv(dest/"functions.csv")
                teacher_distribution = pd.read_csv(dest/"teacher_distribution.csv")
            else:
                model = inputs.teacher(head, seed, device)
                y_source = predict(model, changed_x, device)-predict(model, clean_x, device)
                np.savez(dest / "source_targets.npz", train_y=y_source[train], val_y=y_source[val])
                command = [str(ebm_python), "-m", "fresco.extensions.ebm_worker", "--shared", str(shared),
                           "--targets", str(dest/"source_targets.npz"), "--destination", str(dest), "--seed", str(seed)]
                result = subprocess.run(command, env=environment, capture_output=True, text=True, check=False)
                (dest / "worker.log").write_text(result.stdout+result.stderr, encoding="utf-8")
                if result.returncode:
                    raise RuntimeError(f"Separate EBM worker failed for {status_key}: {result.stderr[-2000:]}")
                preds = np.load(dest/"predictions.npy", mmap_mode="r")
                if preds.shape != (len(METHODS), len(pairs)) or not np.isfinite(preds).all():
                    raise ValueError("Surrogate worker returned incomplete or nonfinite predictions")
                teacher_path = inputs.common / head / "params" / f"seed{seed}"
                marker = json.loads((teacher_path / "DONE.json").read_text(encoding="utf-8"))
                logits = saved_logits(teacher_path, inputs.meta, expected_hash=marker["files"]["frame_scores.csv"])
                y = logits[pairs.changed_row.to_numpy()]-logits[pairs.clean_row.to_numpy()]
                frame = pairs[["dataset", "condition", "label", "video_hash", "group_hash"]].copy()
                frame["total"] = y
                frame["clean_teacher_logit"] = logits[pairs.clean_row.to_numpy()]
                frame["changed_teacher_logit"] = logits[pairs.changed_row.to_numpy()]
                for j, method in enumerate(METHODS):
                    frame[method] = preds[j]
                video = _surrogate_video(frame)
                metrics = _surrogate_metrics(frame, head, seed, float(cfg.sign_tolerance))
                frame["observed_bin"] = np.floor(np.clip(frame.total, -10, 10)*10)/10
                agg = {"n": ("total", "size"), "observed": ("total", "mean")}
                agg.update({method: (method, "mean") for method in METHODS})
                agreement = frame.groupby(["dataset", "observed_bin"], observed=True, as_index=False).agg(**agg)
                agreement["head"], agreement["seed"] = head, seed
                teacher_distribution = _teacher_distribution(frame, head, seed)
                functions = pd.read_csv(dest / "functions.csv")
                functions["head"], functions["seed"] = head, seed
                functions.to_csv(dest / "functions.csv", index=False)
                for table, name in ((video, "video.csv"), (metrics, "metrics.csv"), (agreement, "agreement.csv"),
                                    (teacher_distribution, "teacher_distribution.csv")):
                    table.to_csv(dest/name, index=False)
                atomic_json(done, {"signature": signature, "files": {n: sha256(dest/n) for n in names}})
            video["head"], video["seed"] = head, seed
            tables[(head, seed)] = video
            metrics_all.append(metrics)
            functions_all.append(functions)
            agreement_all.append(agreement)
            teacher_distributions.append(teacher_distribution)
            statuses[status_key] = {"status": "COMPLETED"}
            atomic_json(statuses_path, statuses)
    (out / "figures").mkdir(exist_ok=True)
    pd.concat(tables.values(), ignore_index=True).to_csv(out / "per_video.csv.gz", index=False, compression="gzip")
    pd.concat(metrics_all, ignore_index=True).to_csv(out / "metrics.csv", index=False)
    pd.concat(teacher_distributions, ignore_index=True).to_csv(out / "teacher_score_distribution.csv", index=False)
    pd.concat(functions_all, ignore_index=True).to_csv(out / "figures" / "surrogate_functions.csv", index=False)
    pd.concat(agreement_all, ignore_index=True).to_csv(out / "figures" / "surrogate_agreement.csv", index=False)
    reports, fidelity, transfer = _summary(tables, int(cfg.bootstrap_resamples), int(cfg.bootstrap_seed))
    fidelity.to_csv(out / "figures" / "surrogate_fidelity.csv", index=False)
    transfer.to_csv(out / "figures" / "surrogate_transfer.csv", index=False)
    atomic_json(out / "metrics.json", {"status": "RUN", "seeds": SEEDS, "contrasts": reports,
                                       "pairs": len(pairs), "ebm_versions": versions,
                                       "input_features": "four clean and four changed quality values only",
                                       "original_verdict": "NOT_SUPPORTED; unchanged"})
    from .surrogate_visuals import surrogate_figures
    surrogate_figures(out / "figures")
    from .figure_manifest import write_figure_manifest
    write_figure_manifest(out / "figures", "surrogates")
    from .surrogate_report import render_report
    (out / "REPORT.md").write_text(render_report(json.loads((out/"metrics.json").read_text(encoding="utf-8"))),
                                   encoding="utf-8")
    return out