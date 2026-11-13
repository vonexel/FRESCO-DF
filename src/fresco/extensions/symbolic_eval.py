from __future__ import annotations

import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
from fresco.heads import benchmark
from fresco.recompute_curves import saved_logits
from fresco.research_audit import _crossed_bootstrap, paired_auc_uncertainty, paired_gap_uncertainty
from fresco.training import predict
from fresco.utils import atomic_json
from .analysis_store import create_analysis_run
from .inventory import sha256
from .preflight import SEEDS
from .runtime import FrozenInputs, digest_id
from .symbolic import _design, extract_edges
from .symbolic_model import build_clones
from .symbolic_run import _sample_ids, _sample_source


def load_attempt(path, inputs, cfg):
    path = Path(path).resolve()
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    identity = manifest["identity"]
    if (identity.get("main_manifest_sha256") != inputs.preflight["main_manifest_sha256"]
            or identity.get("coordinates_sha256") != inputs.preflight["coordinates_sha256"]
            or identity.get("array_hashes") != {k: v["expected_sha256"]
                                                 for k, v in inputs.preflight["array_checks"].items()}):
        raise ValueError("Symbolic edge attempt does not match original teachers/cache/coordinates")
    previous = identity["config"]["symbolic"]
    for key in ("vocabulary", "max_rows_per_split", "edge_validation_rmse_tolerance", "max_candidates_per_edge"):
        if previous.get(key) != cfg.symbolic.get(key):
            raise ValueError(f"Symbolic source fitting policy changed: {key}")
    proof = {str(t["seed"]): t["checkpoint_sha256"] for t in inputs.preflight["teachers"] if t["head"] == "kan_spline"}
    if identity.get("teacher_hashes") != proof:
        raise ValueError("Symbolic attempt teachers differ from verified original KANs")
    records = {}
    for seed in SEEDS:
        data = path/f"seed{seed}.json"
        marker = json.loads((path/f"seed{seed}.done.json").read_text(encoding="utf-8"))
        if marker.get("signature") != manifest["signature"] or marker.get("sha256") != sha256(data):
            raise ValueError(f"Symbolic attempt seed {seed} failed completion hash")
        value = json.loads(data.read_text(encoding="utf-8"))
        if value.get("seed") != seed or value.get("status") != "FULL_SYMBOLIFICATION_NOT_EVALUATED":
            raise ValueError(f"Symbolic attempt seed {seed} is missing or unsuccessful")
        records[seed] = value
    return manifest, records


def _test_coordinates(inputs, path, signature, reference_run=None):
    meta = inputs.meta.loc[inputs.meta.split == "test"].reset_index(drop=True)
    if reference_run is not None:
        reference = Path(reference_run).resolve()
        manifest = json.loads((reference/"manifest.json").read_text(encoding="utf-8"))
        old_identity = manifest["identity"]
        expected_hashes = {k: v["expected_sha256"] for k, v in inputs.preflight["array_checks"].items()}
        if (old_identity.get("original_manifest_sha256") != inputs.preflight["main_manifest_sha256"]
                or old_identity.get("coordinates_sha256") != inputs.preflight["coordinates_sha256"]
                or old_identity.get("cache_hashes") != expected_hashes):
            raise ValueError("Referenced test projection uses different original inputs")
        source, done = reference/"test_coordinates.npy", reference/"test_coordinates.done.json"
        marker = json.loads(done.read_text(encoding="utf-8"))
        if marker.get("signature") != manifest.get("signature") or marker.get("sha256") != sha256(source):
            raise ValueError("Referenced test projection failed its run signature or checksum")
        projection = np.load(source, mmap_mode="r")
        if projection.shape != (len(meta), 20) or projection.dtype != np.float32:
            raise ValueError("Referenced test projection shape or dtype differs")
        atomic_json(path/"test_coordinates_reference.json", {"source_run": str(reference),
                    "source_manifest_sha256": sha256(reference/"manifest.json"), "sha256": marker["sha256"]})
        return meta, projection
    target, marker_path = path/"test_coordinates.npy", path/"test_coordinates.done.json"
    if target.is_file() != marker_path.is_file():
        raise ValueError("Incomplete test projection; retain the partial evidence and choose a new run-id")
    if target.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("signature") != signature or marker.get("sha256") != sha256(target):
            raise ValueError("Saved test coordinates failed signature/hash")
    else:
        temporary = path/"test_coordinates.npy.tmp"
        mmap = np.lib.format.open_memmap(str(temporary), mode="w+", dtype=np.float32, shape=(len(meta), 20))
        rows = meta.row.to_numpy(dtype=np.int64)
        for start in tqdm(range(0, len(rows), 4096), desc="Project held-out CLIP", unit="batch"):
            mmap[start:start+4096] = inputs.coordinates(rows[start:start+4096])
        mmap.flush()
        del mmap
        temporary.replace(target)
        atomic_json(marker_path, {"signature": signature, "sha256": sha256(target)})
    return meta, np.load(target, mmap_mode="r")


def _model_bytes(model):
    data = io.BytesIO()
    torch.save(model.state_dict(), data)
    return len(data.getvalue())


def _support(model, x, records, device, batch=4096):
    bounds = []
    for layer_index in range(len(model.network)):
        rows = [r for r in records if r["layer"] == layer_index]
        width = model.network[layer_index].base_weight.shape[1]
        lo = np.array([min(r["source_min"] for r in rows if r["input"] == j) for j in range(width)])
        hi = np.array([max(r["source_max"] for r in rows if r["input"] == j) for j in range(width)])
        bounds.append((lo, hi))
    result = np.empty(len(x), dtype=bool)
    with torch.inference_mode():
        for start in range(0, len(x), batch):
            a = torch.as_tensor(np.array(x[start:start+batch], dtype=np.float32), device=device)
            layer0 = a.cpu().numpy()
            hidden = model.network[0](a).detach().cpu().numpy()
            result[start:start+len(a)] = (((layer0 >= bounds[0][0]) & (layer0 <= bounds[0][1])).all(1)
                                         & ((hidden >= bounds[1][0]) & (hidden <= bounds[1][1])).all(1))
    return result


def _video_table(meta, original, pruned, mixed, supported):
    table = meta[["dataset", "condition", "video_id", "group_id", "label"]].copy()
    table["teacher_probability"] = expit(original)
    table["teacher_logit"] = original
    for name, values in (("pruned", pruned), ("mixed", mixed)):
        table[name+"_probability"] = expit(values)
        table[name+"_error_sse"] = (values-original)**2
        table[name+"_absolute_error"] = abs(values-original)
        table[name+"_logit"] = values
        table[name+"_logit_sse"] = values**2
        table[name+"_cross_logit"] = values*original
    table["teacher_logit_sse"] = original**2
    table["source_supported"] = supported
    grouped = table.groupby(["dataset", "condition", "video_id", "group_id", "label"], observed=True, as_index=False)
    return grouped.agg(frames=("teacher_probability", "size"),
        teacher_probability=("teacher_probability", "mean"), pruned_probability=("pruned_probability", "mean"),
        mixed_probability=("mixed_probability", "mean"), pruned_error_sse=("pruned_error_sse", "sum"),
        mixed_error_sse=("mixed_error_sse", "sum"), pruned_absolute_error=("pruned_absolute_error", "sum"),
        mixed_absolute_error=("mixed_absolute_error", "sum"), teacher_logit_sse=("teacher_logit_sse", "sum"),
        source_supported=("source_supported", "sum"), teacher_logit_sum=("teacher_logit", "sum"),
        pruned_logit_sum=("pruned_logit", "sum"), mixed_logit_sum=("mixed_logit", "sum"),
        pruned_logit_sse=("pruned_logit_sse", "sum"), mixed_logit_sse=("mixed_logit_sse", "sum"),
        pruned_cross_logit=("pruned_cross_logit", "sum"), mixed_cross_logit=("mixed_cross_logit", "sum"))


def _per_condition(video, seed, threshold, complexity, timing, val_metrics):
    rows = []
    for (dataset, condition), part in video.groupby(["dataset", "condition"], observed=True):
        teacher_auc = roc_auc_score(part.label, part.teacher_probability) if part.label.nunique() == 2 else None
        for name in ("pruned", "mixed"):
            score = part[name+"_probability"]
            auc = roc_auc_score(part.label, score) if part.label.nunique() == 2 else None
            frame_n = part.frames.to_numpy(float)
            error = (part[name+"_error_sse"]/frame_n).mean()
            denom = (part.teacher_logit_sse/frame_n).mean()
            teacher_mean = (part.teacher_logit_sum/frame_n).mean()
            clone_mean = (part[name+"_logit_sum"]/frame_n).mean()
            teacher_variance = denom-teacher_mean**2
            clone_variance = (part[name+"_logit_sse"]/frame_n).mean()-clone_mean**2
            covariance = (part[name+"_cross_logit"]/frame_n).mean()-teacher_mean*clone_mean
            logit_correlation = (float(np.clip(covariance/np.sqrt(teacher_variance*clone_variance), -1, 1))
                                 if teacher_variance > 1e-16 and clone_variance > 1e-16 else None)
            rows.append({"seed": seed, "dataset": dataset, "condition": condition, "clone": name,
                         "videos": len(part), "frames": int(frame_n.sum()), "teacher_auc": teacher_auc,
                         "clone_auc": auc, "auc_change": auc-teacher_auc if auc is not None else None,
                         "logit_rmse_equal_video": float(np.sqrt(error)),
                         "logit_mae_equal_video": float((part[name+"_absolute_error"]/frame_n).mean()),
                         "normalized_logit_error": float(np.sqrt(error/denom)) if denom > 1e-16 else None,
                         "hard_label_disagreement": float(((part.teacher_probability >= threshold)
                                                          != (score >= threshold)).mean()),
                         "video_probability_correlation": float(part.teacher_probability.corr(score))
                         if part.teacher_probability.std() > 0 and score.std() > 0 else None,
                         "logit_pearson_equal_video": logit_correlation,
                         "source_validation_selected_threshold": threshold,
                         "source_supported_fraction": float(part.source_supported.sum()/frame_n.sum()),
                         "head_latency_ms": timing[name]["p50_ms"], "model_bytes": timing[name]["model_bytes"],
                         "source_validation_rmse": val_metrics[name], **complexity})
    return pd.DataFrame(rows)


def _symbolic_intervals(tables, resamples, seed):
    result = {}
    for dataset in ("ffpp", "celebdfpp"):
        for clone in ("pruned", "mixed"):
            labels, groups, reference, teachers, simplified, errors, totals = None, None, None, [], [], [], []
            for index in SEEDS:
                table = tables[index]
                p = table[(table.dataset == dataset) & (table.condition == "clean")].sort_values("video_id")
                identity = p[["video_id", "group_id", "label"]].reset_index(drop=True)
                if reference is None:
                    reference, labels, groups = identity, p.label.to_numpy(), p.group_id.to_numpy()
                elif not reference.equals(identity):
                    raise ValueError("Symbolic evaluation video groups differ between seeds")
                teachers.append(p.teacher_probability.to_numpy())
                simplified.append(p[clone+"_probability"].to_numpy())
                errors.append((p[clone+"_error_sse"]/p.frames).to_numpy())
                totals.append((p.teacher_logit_sse/p.frames).to_numpy())
            result[f"{dataset}/{clone}_auc_change"] = paired_auc_uncertainty(
                labels, np.stack(simplified), np.stack(teachers), groups, resamples=resamples, seed=seed)
            error, total = np.stack(errors), np.stack(totals)
            unique, inverse = np.unique(groups.astype(str), return_inverse=True)

            def rmse(weights, values=error):
                return np.sqrt(values @ weights / weights.sum())

            result[f"{dataset}/{clone}_logit_rmse"] = _crossed_bootstrap(
                rmse, len(SEEDS), inverse, len(unique), resamples, seed)
            result[f"{dataset}/{clone}_normalized_logit_error"] = paired_gap_uncertainty(
                error, total, np.zeros_like(error), total, groups, resamples=resamples, seed=seed)
    return result


def _symbolic_transfer(tables, resamples, seed):
    """Difference in equal-video RMSE, independent dataset clusters, paired seeds"""
    rng = np.random.default_rng(seed)
    seed_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    observed, bootstraps, sizes = [], [], []
    for dataset in ("ffpp", "celebdfpp"):
        reference, errors = None, []
        for index in SEEDS:
            table = tables[index]
            part = table[(table.dataset == dataset) & (table.condition == "clean")].sort_values("video_id")
            identity = part[["video_id", "group_id", "label"]].reset_index(drop=True)
            if reference is not None and not reference.equals(identity):
                raise ValueError("Symbolic transfer populations differ between seeds")
            reference = identity
            errors.append((part.mixed_error_sse/part.frames).to_numpy())
        if reference.empty:
            return {"status": "INCONCLUSIVE", "reason": f"no clean videos in {dataset}"}
        _, inverse = np.unique(reference.group_id.astype(str), return_inverse=True)
        clusters = int(inverse.max()+1)
        if clusters < 5:
            return {"status": "INCONCLUSIVE", "reason": "fewer than five test clusters in a dataset"}
        error = np.stack(errors)
        grouped = np.stack([np.bincount(inverse, weights=row, minlength=clusters) for row in error])
        counts = np.stack([np.bincount(rng.integers(clusters, size=clusters), minlength=clusters)
                           for _ in range(resamples)])
        sizes_by_group = np.bincount(inverse, minlength=clusters)
        bootstraps.append(np.sqrt((grouped @ counts.T)/(sizes_by_group @ counts.T)))
        observed.append(np.sqrt(error.mean(axis=1)))
        sizes.append({"videos": len(reference), "clusters": clusters})
    delta = bootstraps[1]-bootstraps[0]
    draws = seed_rng.integers(len(SEEDS), size=(resamples, len(SEEDS)))
    crossed = delta[draws.T, np.arange(resamples)].mean(axis=0)
    fixed = delta.mean(axis=0)
    if not np.isfinite(crossed).all() or not np.isfinite(fixed).all():
        return {"status": "INCONCLUSIVE", "reason": "nonfinite resampled error"}
    return {"status": "COMPUTED", "difference": float((observed[1]-observed[0]).mean()),
            "seed_differences": (observed[1]-observed[0]).tolist(),
            "fixed_cluster_ci95": np.quantile(fixed, [.025, .975]).tolist(),
            "crossed_seed_cluster_ci95": np.quantile(crossed, [.025, .975]).tolist(),
            "source": sizes[0], "target": sizes[1], "seeds": len(SEEDS),
            "scope": "independent dataset clusters; paired teacher seeds; equal-video logit RMSE"}


def _edge_figure_data(teacher, records, source_x, representative):
    layers = extract_edges(teacher.cpu(), source_x)
    result = []
    for item in representative:
        layer, out, inp = int(item.layer), int(item.output), int(item.input)
        record = next(r for r in records if r["layer"] == layer and r["output"] == out and r["input"] == inp)
        values = layers[layer]["inputs"][:, inp]
        grid = np.linspace(values.min(), values.max(), 101)
        shape = (len(grid), teacher.network[layer].base_weight.shape[1])
        source = torch.zeros(shape, dtype=torch.float32)
        source[:, inp] = torch.as_tensor(grid, dtype=torch.float32)
        with torch.no_grad():
            actual = teacher.network[layer].edges(source)[:, out, inp].numpy()
        approximate = _design(grid, record["family"]) @ np.asarray(record["coefficients"], dtype=float)
        occupancy, bins = np.histogram(values, bins=20, range=(values.min(), values.max()))
        index = np.clip(np.searchsorted(bins, grid, side="right")-1, 0, len(occupancy)-1)
        for x, a, b, n in zip(grid, actual, approximate, occupancy[index]):
            result.append({"layer": layer, "output": out, "input": inp, "x": x, "actual": a,
                           "approximation": b, "source_bin_count": int(n), "family": record["family"],
                           "accepted": record["accepted"], "edge_validation_rmse": record["validation_rmse"]})
    return pd.DataFrame(result)


def evaluate_symbolic(cfg, cache, main, replay, attempt, run_id, device="cpu", coordinates_run=None):
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; select cpu")
    inputs = FrozenInputs(cache, main, replay)
    attempt_manifest, attempts = load_attempt(attempt, inputs, cfg)
    out, signature = create_analysis_run(cfg, inputs, run_id, "symbolic", extra={
        "attempt_manifest_sha256": sha256(Path(attempt)/"manifest.json"), "device": device,
        "coordinate_source_manifest_sha256": sha256(Path(coordinates_run)/"manifest.json")
        if coordinates_run is not None else None})
    samples, _, _ = _sample_source(inputs.cache, int(cfg.symbolic.max_rows_per_split))
    ids = {name: _sample_ids(meta) for name, meta in samples.items()}
    for split, values in ids.items():
        import hashlib
        if hashlib.sha256("\n".join(values).encode()).hexdigest() != attempt_manifest["identity"]["sample_ids_sha256"][split]:
            raise ValueError("Source samples differ from bounded symbolic attempt")
    source = {split: inputs.coordinates(meta.row.to_numpy()) for split, meta in samples.items()}
    meta, test_x = _test_coordinates(inputs, out, signature, coordinates_run)
    status_path = out/"seed_status.json"
    statuses = (json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else
                {str(s): {"status": "PENDING"} for s in SEEDS})
    atomic_json(status_path, statuses)
    tables, metrics = {}, []
    for seed in SEEDS:
        statuses[str(seed)] = {"status": "RUNNING"}
        atomic_json(status_path, statuses)
        directory = out/f"seed{seed}"
        directory.mkdir(exist_ok=True)
        names = ["video.csv.gz", "metrics.csv", "complexity.json"]
        marker = directory/"DONE.json"
        if marker.exists():
            done = json.loads(marker.read_text(encoding="utf-8"))
            if done.get("signature") != signature or any(sha256(directory/n) != done["files"].get(n) for n in names):
                raise ValueError("Saved symbolic evaluation seed is corrupted")
            video = pd.read_csv(directory/"video.csv.gz")
            result = pd.read_csv(directory/"metrics.csv")
        else:
            teacher = inputs.teacher("kan_spline", seed, device)
            records = attempts[seed]["records"]
            pruned, mixed, complexity = build_clones(teacher, records)
            pruned, mixed = pruned.to(device), mixed.to(device)
            validation = source["val"]
            original_val = predict(teacher, validation, device)
            val_metrics = {name: float(np.sqrt(np.mean((predict(model, validation, device)-original_val)**2)))
                           for name, model in (("pruned", pruned), ("mixed", mixed))}
            timing = {}
            sample = torch.as_tensor(validation[:1], dtype=torch.float32, device=device)
            for name, model in (("teacher", teacher), ("pruned", pruned), ("mixed", mixed)):
                timing[name] = {**benchmark(model, sample, warmup=10, repeats=50), "model_bytes": _model_bytes(model)}
            if seed == 0:
                (out/"figures").mkdir(exist_ok=True)
                _edge_figure_data(teacher, records, source["train"], cfg.symbolic.representative_edges).to_csv(
                    out/"figures"/"symbolic_edges.csv", index=False)
                teacher = teacher.to(device)
            original_path = inputs.common/"kan_spline"/"params"/f"seed{seed}"
            original_done = json.loads((original_path/"DONE.json").read_text(encoding="utf-8"))
            all_logits = saved_logits(original_path, inputs.meta, expected_hash=original_done["files"]["frame_scores.csv"])
            original = all_logits[meta.row.to_numpy(dtype=np.int64)]
            replay_sample = predict(teacher, test_x[:2048], device)
            replay_max_abs = float(np.max(abs(replay_sample-original[:len(replay_sample)])))
            if replay_max_abs > 1e-3:
                raise ValueError(f"Teacher replay differs from verified saved test logits: {replay_max_abs:g}")
            p = original if complexity["pruned_edges"] == 0 else predict(pruned, test_x, device)
            m = predict(mixed, test_x, device)
            supported = _support(teacher, test_x, records, device)
            video = _video_table(meta, original, p, m, supported)
            temp = json.loads((original_path/"temperature.json").read_text(encoding="utf-8"))
            threshold = float(temp["thresholds"]["video"]["raw"])
            result = _per_condition(video, seed, threshold, complexity, timing, val_metrics)
            result["teacher_head_latency_ms"] = timing["teacher"]["p50_ms"]
            result["teacher_replay_sample_max_abs"] = replay_max_abs
            result["fit_elapsed_seconds"] = attempts[seed]["elapsed_seconds"]
            result["pruned_edge_fraction"] = complexity["pruned_edges"]/complexity["total_edges"]
            result["edge_search_success_fraction"] = complexity["accepted_symbolic_edges"]/complexity["total_edges"]
            result["nontrivial_symbolic_functions"] = sum(
                r["accepted"] and r["family"] not in ("zero", "constant") for r in records)
            result["symbolic_expression_nodes_proxy"] = sum(
                {"zero": 0, "constant": 1, "linear": 3, "quadratic": 5, "cubic": 7, "silu": 3}[r["family"]]
                for r in records if r["accepted"])
            result["full_symbolic_status"] = ("RUN" if complexity["full_symbolification"]
                                               else "FULL_SYMBOLIFICATION_NOT_EVALUATED")
            # Hash IDs before public artifacts; original IDs never leave memory.
            video["video_id"] = video.video_id.astype(str).map(lambda v: digest_id("video", v))
            video["group_id"] = video.group_id.astype(str).map(lambda v: digest_id("group", v))
            video.to_csv(directory/"video.csv.gz", index=False, compression="gzip")
            result.to_csv(directory/"metrics.csv", index=False)
            atomic_json(directory/"complexity.json", {**complexity, "timing": timing, "validation": val_metrics})
            atomic_json(marker, {"signature": signature, "files": {n: sha256(directory/n) for n in names}})
        tables[seed] = video
        metrics.append(result)
        statuses[str(seed)] = {"status": "COMPLETED"}
        atomic_json(status_path, statuses)
    all_metrics = pd.concat(metrics, ignore_index=True)
    all_metrics.to_csv(out/"metrics.csv", index=False)
    columns = ["logit_rmse_equal_video", "logit_mae_equal_video", "normalized_logit_error",
               "logit_pearson_equal_video", "hard_label_disagreement", "auc_change", "model_bytes",
               "head_latency_ms", "pruned_edge_fraction", "edge_search_success_fraction"]
    summary = all_metrics.groupby(["dataset", "condition", "clone"], observed=True)[columns].agg(
        ["median", lambda x: x.quantile(.25), lambda x: x.quantile(.75)])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.rename(columns=lambda name: name.replace("<lambda_0>", "q25").replace("<lambda_1>", "q75"),
                   inplace=True)
    summary.reset_index().to_csv(out/"seed_summary.csv", index=False)
    pd.concat([v.assign(seed=s) for s, v in tables.items()], ignore_index=True).to_csv(
        out/"per_video.csv.gz", index=False, compression="gzip")
    intervals = _symbolic_intervals(tables, int(cfg.bootstrap_resamples), int(cfg.bootstrap_seed))
    (out/"figures").mkdir(exist_ok=True)
    all_metrics.loc[all_metrics.condition == "clean"].to_csv(out/"figures"/"symbolic_tradeoff.csv", index=False)
    complete = int(all_metrics.groupby("seed").full_symbolification.first().sum())
    if complete == len(SEEDS):
        transfer = _symbolic_transfer(tables, int(cfg.bootstrap_resamples), int(cfg.bootstrap_seed))
        intervals["complete_symbolic_transfer"] = transfer
        if transfer["status"] == "COMPUTED":
            pd.DataFrame([{"difference": transfer["difference"],
                           "ci_low": transfer["crossed_seed_cluster_ci95"][0],
                           "ci_high": transfer["crossed_seed_cluster_ci95"][1],
                           "source_videos": transfer["source"]["videos"],
                           "target_videos": transfer["target"]["videos"], "seeds": len(SEEDS)}]).to_csv(
                               out/"figures"/"symbolic_transfer.csv", index=False)
    from .symbolic_visuals_v2 import symbolic_figures
    symbolic_figures(out/"figures")
    from .figure_manifest import write_figure_manifest
    write_figure_manifest(out/"figures", "symbolic")
    status = "RUN" if complete == len(SEEDS) else "FULL_SYMBOLIFICATION_NOT_EVALUATED"
    atomic_json(out/"metrics.json", {"status": status, "seed_attempts": 10, "full_symbolic_seeds": complete,
                                     "partial_or_failed_denominator": 10, "intervals": intervals,
                                     "original_verdict": "NOT_SUPPORTED; unchanged"})
    (out/"REPORT.md").write_text(
        "# KAN simplification: whole-model held-out evaluation\n\n"
        f"Ten original seeds evaluated; fully symbolified models: {complete}/10. "
        "Pruned copies and mixed symbolic/spline copies were compared with the original frozen KAN on identical "
        "FF++ and Celeb-DF++ inputs. Mixed copies retain rejected spline edges and are not fully symbolic.\n\n"
        "The source-selected threshold is fixed before test; metrics, seed medians/interquartile ranges and "
        "exploratory paired AUROC, logit RMSE and normalized logit error intervals are in the result tables. "
        "The representative edge plot was fixed before this held-out evaluation. "
        "Retained-edge counts describe structure; stored mixed copies still include original spline weights. "
        "Head latency is measured in the current environment and excludes the encoder; concurrent work can affect it.\n\n"
        "## Limitations\n\nThis is not a KAN-vs-MLP structural benchmark or a causal explanation. "
        "An edge fit alone is not whole-model fidelity. Source bounding boxes do not guarantee manifold support. "
        "Intervals are exploratory, unadjusted, and conditional on source data and test generators.\n", encoding="utf-8")
    return out