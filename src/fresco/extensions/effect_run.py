from __future__ import annotations

import json
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from fresco.recompute_curves import saved_logits
from fresco.research_audit import paired_gap_uncertainty
from fresco.training import COVARIATE_NAMES, predict
from fresco.utils import atomic_json
from .analysis_store import create_analysis_run
from .effects import additive_delta, fit_ale, fit_pd
from .inventory import sha256
from .paired import response_metrics
from .preflight import HEADS, SEEDS
from .response_run import _quality_support
from .runtime import FrozenInputs, digest_id


def _effects(model, background, source_meta, cfg, device):
    def teacher(x):
        return predict(model, np.asarray(x, dtype=np.float32), device, batch=4096)
    pd_curves, ale_curves = [], []
    for j in range(4):
        pd_curves.append(fit_pd(teacher, background, source_meta, 16+j, bins=int(cfg.bins)))
        ale_curves.append(fit_ale(teacher, background, source_meta, 16+j, bins=int(cfg.bins),
                                  min_bin_count=int(cfg.min_bin_count)))
    return pd_curves, ale_curves


def _geometry(effects):
    return [(e.grid.tolist(), e.occupied.tolist(), e.counts.tolist()) for e in effects]


def _effect_video(frame):
    values = frame.copy()
    values["total_sse"] = values.total**2
    values["pd_error_sse"] = (values["pd"]-values.total)**2
    values["ale_error_sse"] = (values["ale"]-values.total)**2
    return values.groupby(["dataset", "condition", "label", "video_hash", "group_hash"],
                          sort=True, observed=True, as_index=False).agg(
        pairs=("total", "size"), total_sse=("total_sse", "sum"),
        pd_error_sse=("pd_error_sse", "sum"), ale_error_sse=("ale_error_sse", "sum"),
        total_mean=("total", "mean"), pd_mean=("pd", "mean"), ale_mean=("ale", "mean"))


def _effect_metrics(frame, head, seed, epsilon):
    rows = []
    for dataset, d in frame.groupby("dataset", observed=True, sort=True):
        groups = [("all", "all", d)]
        groups += [(condition, "all", part) for condition, part in d.groupby("condition", observed=True, sort=True)]
        groups += [("all", int(label), part) for label, part in d.groupby("label", observed=True, sort=True)]
        for condition, label, part in groups:
            for name in ("pd", "ale", "zero"):
                p = np.zeros(len(part)) if name == "zero" else part[name].to_numpy()
                y = part.total.to_numpy()
                metrics = response_metrics(y, p, part.video_hash, sign_tolerance=epsilon)
                if len(y) > 1 and y.std() > 0 and p.std() > 0:
                    metrics["pearson_frame_weighted"] = float(pearsonr(y, p).statistic)
                    metrics["spearman_frame_weighted"] = float(spearmanr(y, p).statistic)
                else:
                    metrics["pearson_frame_weighted"] = metrics["spearman_frame_weighted"] = None
                rows.append({"head": head, "seed": seed, "dataset": dataset,
                             "condition": condition, "label": label, "method": name, **metrics})
    return pd.DataFrame(rows)


def _effect_bootstrap(tables, resamples, random_seed):
    reports, figure = {}, []
    for head in HEADS:
        for dataset in ("ffpp", "celebdfpp"):
            reference, ae, pe, total = None, [], [], []
            for seed in SEEDS:
                frame = tables[(head, seed)]
                frame = frame.loc[frame.dataset == dataset].groupby(
                    ["video_hash", "group_hash"], observed=True, sort=True)[
                        ["pairs", "ale_error_sse", "pd_error_sse", "total_sse"]].sum()
                if reference is None:
                    reference = frame.index
                if not frame.index.equals(reference):
                    raise ValueError("PD/ALE video populations differ across seeds")
                n = frame.pairs.to_numpy(float)
                ae.append(frame.ale_error_sse.to_numpy(float)/n)
                pe.append(frame.pd_error_sse.to_numpy(float)/n)
                total.append(frame.total_sse.to_numpy(float)/n)
            ae, pe, total = np.stack(ae), np.stack(pe), np.stack(total)
            groups = reference.get_level_values("group_hash").to_numpy()
            opts = {"resamples": resamples, "seed": random_seed, "min_total_rms": 1e-8}
            records = {"ale_minus_pd": paired_gap_uncertainty(ae, total, pe, total, groups, **opts),
                       "ale_minus_zero": paired_gap_uncertainty(ae, total, total, total, groups, **opts),
                       "pd_minus_zero": paired_gap_uncertainty(pe, total, total, total, groups, **opts)}
            reports[f"{head}/{dataset}"] = records
            for method in ("pd", "ale"):
                item = records[method+"_minus_zero"]
                ci = item.get("crossed_seeds_clusters", {}).get("ci95")
                figure.append({"head": head, "dataset": dataset, "method": method,
                               "G": item.get("difference", np.nan)+1,
                               "ci_low": ci[0]+1 if ci else np.nan, "ci_high": ci[1]+1 if ci else np.nan,
                               "videos": len(reference), "clusters": len(np.unique(groups)), "seeds": 10})
    return reports, pd.DataFrame(figure)


def run_pd_ale(cfg, cache, main, replay, run_id, device="cpu"):
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; select cpu")
    inputs = FrozenInputs(cache, main, replay)
    out, signature = create_analysis_run(cfg, inputs, run_id, "pd_ale", extra={"device": device})
    status_file = out / "seed_status.json"
    statuses = (json.loads(status_file.read_text(encoding="utf-8")) if status_file.is_file() else
                {f"{h}/seed{s}": {"status": "PENDING"} for h in HEADS for s in SEEDS})
    atomic_json(status_file, statuses)
    bg_meta, background = inputs.source_background(int(cfg.pd_ale.background_rows))
    bg_ids = [digest_id(r.dataset, r.video_id, int(r.frame_idx), r.condition) for r in bg_meta.itertuples()]
    pd.DataFrame({"source_train_pair_id_sha256": bg_ids}).to_csv(out / "source_background_ids.csv", index=False)
    pd.DataFrame(background[:, 16:], columns=COVARIATE_NAMES).corr().to_csv(out / "source_correlations.csv")
    pairs = inputs.pairs()
    c0 = inputs.standardized_quality(pairs.clean_row.to_numpy())
    ca = inputs.standardized_quality(pairs.changed_row.to_numpy())
    quantiles = []
    for j, feature in enumerate(COVARIATE_NAMES):
        for dataset, values in [("ffpp/train/background", background[:, 16+j]),
                                ("ffpp/test/clean", c0[pairs.dataset == "ffpp", j]),
                                ("ffpp/test/degraded", ca[pairs.dataset == "ffpp", j]),
                                ("celebdfpp/test/clean", c0[pairs.dataset == "celebdfpp", j]),
                                ("celebdfpp/test/degraded", ca[pairs.dataset == "celebdfpp", j])]:
            q = np.quantile(values, [.01, .5, .99])
            quantiles.append({"feature": feature, "population": dataset, "n": len(values),
                              "q01": q[0], "q50": q[1], "q99": q[2]})
    pd.DataFrame(quantiles).to_csv(out / "source_target_quality_quantiles.csv", index=False)
    distance, support_bin, support_meta = _quality_support(inputs, c0, ca)
    atomic_json(out / "support_model.json", support_meta)
    # Native support is a function only of the fixed source background and bins, so the first teacher defines geometry that is checked for every seed.
    initial = inputs.teacher(HEADS[0], 0, device)
    initial_pd, initial_ale = _effects(initial, background, bg_meta, cfg.pd_ale, device)
    _, mask_pd = additive_delta(initial_pd, c0, ca)
    _, mask_ale = additive_delta(initial_ale, c0, ca)
    common = mask_pd & mask_ale
    if not common.any():
        raise ValueError("No common PD/ALE supported pairs")
    for method, mask in (("pd", mask_pd), ("ale", mask_ale), ("common", common)):
        pairs[method+"_support"] = mask
    coverage = pairs.groupby(["dataset", "condition", "label"], observed=True, as_index=False).agg(
        paired=("pd_support", "size"), pd_supported=("pd_support", "sum"),
        ale_supported=("ale_support", "sum"), common_supported=("common_support", "sum"))
    (out / "figures").mkdir(exist_ok=True)
    coverage.to_csv(out / "figures" / "pd_ale_coverage.csv", index=False)
    pairs = pairs.loc[common].reset_index(drop=True)
    c0, ca = c0[common], ca[common]
    pairs["support_distance"], pairs["support_bin"] = distance[common], support_bin[common]
    pairs["video_hash"] = pairs.video_id.astype(str).map(lambda x: digest_id("video", x))
    pairs["group_hash"] = pairs.group_id.astype(str).map(lambda x: digest_id("group", x))
    pairs["pair_hash"] = [digest_id(d, v, int(f), c) for d, v, f, c in zip(
        pairs.dataset.astype(str), pairs.video_id.astype(str), pairs.frame_idx, pairs.condition.astype(str))]
    pairs[["dataset", "condition", "label", "video_hash", "group_hash", "pair_hash", "support_distance",
           "support_bin"]].to_csv(out / "common_support_ids.csv.gz", index=False, compression="gzip")
    tables, metrics_all, curves_all, agreement_all, distance_all = {}, [], [], [], []
    for head in HEADS:
        for seed in SEEDS:
            key = f"{head}/seed{seed}"
            statuses[key] = {"status": "RUNNING"}
            atomic_json(status_file, statuses)
            destination = out / head / f"seed{seed}"
            destination.mkdir(parents=True, exist_ok=True)
            names = ["video.csv", "metrics.csv", "curves.csv", "agreement.csv", "distance.csv"]
            done = destination / "DONE.json"
            if done.is_file():
                marker = json.loads(done.read_text(encoding="utf-8"))
                if marker.get("signature") != signature or any(sha256(destination/n) != marker["files"].get(n)
                                                                for n in names):
                    raise ValueError(f"Corrupt or mismatched saved effect seed: {destination}")
                video, metrics, curves, agreement, distances = [pd.read_csv(destination/n) for n in names]
            else:
                model = inputs.teacher(head, seed, device)
                pd_curves, ale_curves = _effects(model, background, bg_meta, cfg.pd_ale, device)
                if (_geometry(pd_curves) != _geometry(initial_pd)
                        or _geometry(ale_curves) != _geometry(initial_ale)):
                    raise ValueError("PD/ALE support geometry changed across frozen teachers")
                pp, mp = additive_delta(pd_curves, c0, ca)
                ap, ma = additive_delta(ale_curves, c0, ca)
                if not (mp.all() and ma.all()):
                    raise ValueError("Common PD/ALE support mask changed")
                teacher_path = inputs.common / head / "params" / f"seed{seed}"
                marker = json.loads((teacher_path / "DONE.json").read_text(encoding="utf-8"))
                logits = saved_logits(teacher_path, inputs.meta, expected_hash=marker["files"]["frame_scores.csv"])
                actual = logits[pairs.changed_row.to_numpy()]-logits[pairs.clean_row.to_numpy()]
                frame = pairs[["dataset", "condition", "label", "video_hash", "group_hash", "support_bin"]].copy()
                frame["total"], frame["pd"], frame["ale"] = actual, pp, ap
                video = _effect_video(frame)
                metrics = _effect_metrics(frame, head, seed, float(cfg.sign_tolerance))
                curve_rows = []
                for j, name in enumerate(COVARIATE_NAMES):
                    for method, effect in (("pd", pd_curves[j]), ("ale", ale_curves[j])):
                        for i, (point, value) in enumerate(zip(effect.grid, effect.values)):
                            curve_rows.append({"head": head, "seed": seed, "feature": name, "method": method,
                                               "grid": point, "response": value,
                                               "bin_count": int(effect.counts[i]) if i < len(effect.counts) else None,
                                               "bin_occupied": bool(effect.occupied[i]) if i < len(effect.occupied) else None})
                curves = pd.DataFrame(curve_rows)
                frame["observed_bin"] = np.floor(np.clip(frame.total, -10, 10)*10)/10
                agreement = frame.groupby(["dataset", "observed_bin"], observed=True, as_index=False).agg(
                    n=("total", "size"), observed=("total", "mean"), pd=("pd", "mean"), ale=("ale", "mean"))
                agreement["head"], agreement["seed"] = head, seed
                dist_rows = []
                for (dataset, bin_id), part in frame.groupby(["dataset", "support_bin"], observed=True):
                    for method in ("pd", "ale"):
                        val = response_metrics(part.total, part[method], part.video_hash,
                                               sign_tolerance=float(cfg.sign_tolerance))
                        dist_rows.append({"head": head, "seed": seed, "dataset": dataset,
                                          "support_bin": bin_id, "method": method, **val})
                distances = pd.DataFrame(dist_rows)
                for table, name in zip((video, metrics, curves, agreement, distances), names):
                    table.to_csv(destination / name, index=False)
                atomic_json(done, {"signature": signature, "files": {n: sha256(destination/n) for n in names}})
            video["head"], video["seed"] = head, seed
            tables[(head, seed)] = video
            metrics_all.append(metrics)
            curves_all.append(curves)
            agreement_all.append(agreement)
            distance_all.append(distances)
            statuses[key] = {"status": "COMPLETED"}
            atomic_json(status_file, statuses)
    pd.concat(tables.values(), ignore_index=True).to_csv(out / "per_video.csv.gz", index=False, compression="gzip")
    pd.concat(metrics_all, ignore_index=True).to_csv(out / "metrics.csv", index=False)
    pd.concat(curves_all, ignore_index=True).to_csv(out / "figures" / "pd_ale_curves.csv", index=False)
    pd.concat(agreement_all, ignore_index=True).to_csv(out / "pd_ale_agreement.csv", index=False)
    pd.concat(distance_all, ignore_index=True).to_csv(out / "pd_ale_distance.csv", index=False)
    confidence, fidelity = _effect_bootstrap(tables, int(cfg.bootstrap_resamples), int(cfg.bootstrap_seed))
    fidelity.to_csv(out / "figures" / "pd_ale_fidelity.csv", index=False)
    atomic_json(out / "metrics.json", {"status": "RUN", "seeds": SEEDS, "common_pairs": len(pairs),
                                       "contrasts": confidence, "weighting": "equal video",
                                       "original_verdict": "NOT_SUPPORTED; unchanged"})
    from .effect_visuals import effect_figures
    effect_figures(out / "figures")
    from .figure_manifest import write_figure_manifest
    write_figure_manifest(out / "figures", "pd_ale")
    (out / "REPORT.md").write_text(
        "# Matched PD and ALE response comparison\n\n"
        f"All ten frozen teachers per head; {len(pairs)} common-support pairs. Negative ALE−PD G favors ALE. "
        "Both methods use the same source-train background and identical held-out pairs and equal-video weights.\n\n"
        "See metrics.json for paired fixed-seed cluster and crossed seed × cluster percentile intervals. "
        "Zero-change prediction is G=1 where the denominator is defined.\n\n"
        "## Limitations\n\nExploratory after target inspection; intervals are unadjusted and conditional on the "
        "observed source data and generators. PD/ALE describe model responses, not causal effects; "
        "marginal support does not ensure joint-manifold support. The original PD and G_curve remain unchanged.\n",
        encoding="utf-8")
    return out