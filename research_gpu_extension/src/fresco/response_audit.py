from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .heads import BSplineActivation, SplineEdgeLayer
from .interpret import response_curves
from .training import train_mask, validation_mask, selected_covariate_names


COVARIATE_NAMES = ("jpeg_blockiness_proxy", "log_laplacian_variance", "face_size",
                   "high_frequency_ratio")
CURVE_COVERAGE_VERSION = 2
CURVE_COVERAGE_RULE = (
    "finite exactly unchanged coordinates contribute zero; changed coordinates require both endpoints "
    "within the frozen source curve grid"
)
CURVE_POPULATION_WARNING = (
    "G_curve uses covered paired frames; G_block from the original full-test audit is not directly "
    "comparable unless evaluated on identical paired frames with identical weighting"
)
PAIR_KEYS = ["dataset", "condition", "video_id", "label", "method", "group_id"]
PAIR_COLUMNS = PAIR_KEYS + ["paired_frames", "covered_frames", "frame_count", "error_sse",
    "total_sse", "total_sse_all", "predicted_sse", "absolute_error_sum", "sign_agree_count",
    "sign_trials", "near_zero_total_frames", "curve_coverage", "curve_rmse", "curve_mae",
    "total_rms_covered", "total_rms_all", "sign_agreement", "curve_response_gap"]


def source_curve_grids(X, meta, coords, points=31, held_out=None):
    if points < 2:
        raise ValueError("At least two curve grid points are required")
    k = len(coords.zmean)
    clean = train_mask(meta, held_out)
    source_degraded = ((meta.dataset == "ffpp") & meta.split.isin(["train", "val"])
                       & (meta.condition != "clean"))
    source_degraded &= meta.method.isin(set(meta.loc[clean, "method"]))
    if held_out is not None:
        source_degraded &= meta.method != held_out
    grids = []
    for index in range(k, X.shape[1]):
        low, high = np.quantile(X[clean, index], [.01, .99])
        if source_degraded.any():
            values = X[source_degraded.to_numpy(), index]
            low, high = min(low, float(values.min())), max(high, float(values.max()))
        if not np.isfinite([low, high]).all():
            raise ValueError("Nonfinite source grid")
        if high-low < 1e-6:
            low, high = float(low)-1e-6, float(high)+1e-6
        grids.append(np.linspace(low, high, points, dtype=np.float64))
    return grids


def curve_delta(clean_covariates, changed_covariates, grids, responses):
    clean, changed = np.asarray(clean_covariates), np.asarray(changed_covariates)
    if clean.ndim != 2 or clean.shape != changed.shape or clean.shape[1] != len(grids):
        raise ValueError("Aligned [frames,covariates] endpoints and curves required")
    if len(grids) != len(responses):
        raise ValueError("Every grid needs one response curve")
    delta, covered = np.zeros(len(clean), dtype=np.float64), np.ones(len(clean), dtype=bool)
    for j, (grid, response) in enumerate(zip(grids, responses)):
        grid, response = np.asarray(grid, float), np.asarray(response, float)
        if (grid.ndim != 1 or len(grid) < 2 or grid.shape != response.shape
                or not np.isfinite(grid).all() or not np.isfinite(response).all()
                or not (np.diff(grid) > 0).all()):
            raise ValueError("Curves need finite, strictly increasing aligned grids")
        finite = np.isfinite(clean[:, j]) & np.isfinite(changed[:, j])
        unchanged = finite & (clean[:, j] == changed[:, j])
        in_range = (finite & (clean[:, j] >= grid[0]) & (clean[:, j] <= grid[-1])
                    & (changed[:, j] >= grid[0]) & (changed[:, j] <= grid[-1]))
        covered &= unchanged | in_range
        evaluate = in_range & ~unchanged
        delta[evaluate] += (np.interp(changed[evaluate, j], grid, response)
                            - np.interp(clean[evaluate, j], grid, response))
    delta[~covered] = np.nan
    return delta, covered


def summarize_curve_response(table, epsilon=1e-8):
    metadata = {"coverage_rule_version": CURVE_COVERAGE_VERSION,
                "coverage_rule": CURVE_COVERAGE_RULE,
                "population_warning": CURVE_POPULATION_WARNING,
                "zero_change_reference_gap": None}
    if table.empty or table.paired_frames.sum() == 0:
        return {**metadata, "status": "unavailable", "reason": "no matched clean/degraded frames"}
    paired, covered = int(table.paired_frames.sum()), int(table.covered_frames.sum())
    result = {**metadata, "paired_frames": paired, "covered_frames": covered, "coverage": covered/paired,
              "coverage_complete": covered == paired, "videos": int(table.video_id.nunique()),
              "covered_videos": int(table.loc[table.covered_frames > 0, "video_id"].nunique()),
              "scope": "equal-video mean-square ratio on covered endpoints only",
              "confirmatory_primary": False}
    if covered == 0:
        return {**result, "status": "unavailable", "gap": None,
                "reason": "all paired endpoints outside source curve grids"}
    video = table.groupby(["dataset", "video_id"])[[
        "error_sse", "total_sse", "covered_frames", "total_sse_all", "paired_frames"]].sum()
    observed = video.loc[video.covered_frames > 0]
    error = float((observed.error_sse/observed.covered_frames).mean())
    total = float((observed.total_sse/observed.covered_frames).mean())
    sign_trials = int(table.sign_trials.sum())
    result.update({"rmse": float(np.sqrt(error)), "total_rms_covered": float(np.sqrt(total)),
        "total_rms_all": float(np.sqrt((video.total_sse_all/video.paired_frames).mean())),
        "mae_frame_weighted": float(table.absolute_error_sum.sum()/covered),
        "sign_trials": sign_trials,
        "sign_agreement": float(table.sign_agree_count.sum()/sign_trials) if sign_trials else None,
        "near_zero_total_frames": int(table.near_zero_total_frames.sum())})
    if total <= epsilon**2:
        return {**result, "status": "unavailable", "gap": None,
                "reason": "near-zero total response on covered endpoints"}
    return {**result, "status": "computed" if covered == paired else "partial_coverage",
            "gap": float(np.sqrt(error/total)), "zero_change_reference_gap": 1.0}


def _predict(model, values, device):
    with torch.inference_mode():
        result = model(torch.as_tensor(values, dtype=torch.float32, device=device)).reshape(-1)
    values = result.detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite model predictions during response audit")
    return values


def _curve_pairs(model, X, meta, k, grids, responses, device, batch, epsilon, logits=None):
    clean = meta.loc[(meta.split == "test") & (meta.condition == "clean"),
                     ["dataset", "video_id", "frame_idx"]].copy()
    clean["clean_row"] = clean.index
    keys = ["dataset", "video_id", "frame_idx"]
    changed = meta.loc[(meta.split == "test") & (meta.condition != "clean")].copy()
    if clean.duplicated(keys).any() or changed.duplicated(keys+["condition"]).any():
        raise ValueError("Curve audit requires unique paired frame-condition keys")
    pieces, matched = [], 0
    names = ["paired_frames", "covered_frames", "error_sse", "total_sse", "total_sse_all",
             "predicted_sse", "absolute_error_sum", "sign_trials", "sign_agree_count", "near_zero_total_frames"]
    for _, condition in changed.groupby(["dataset", "condition"], sort=True):
        condition["changed_row"] = condition.index
        pairs = condition.merge(clean, on=keys, how="inner", validate="many_to_one")
        matched += len(pairs)
        for start in range(0, len(pairs), batch):
            part = pairs.iloc[start:start+batch]
            a, b = X[part.clean_row.to_numpy()], X[part.changed_row.to_numpy()]
            prediction, covered = curve_delta(a[:, k:], b[:, k:], grids, responses)
            if logits is None:
                total = _predict(model, b, device)-_predict(model, a, device)
            else:
                total = logits[part.changed_row.to_numpy()]-logits[part.clean_row.to_numpy()]
            error = np.where(covered, prediction-total, 0.)
            nonzero = covered & (np.abs(total) > epsilon)
            rows = part[PAIR_KEYS].copy()
            rows["paired_frames"], rows["covered_frames"] = 1, covered.astype(int)
            rows["error_sse"], rows["total_sse"] = error**2, np.where(covered, total**2, 0.)
            rows["total_sse_all"] = total**2
            rows["predicted_sse"] = np.where(covered, prediction**2, 0.)
            rows["absolute_error_sum"] = np.abs(error)
            rows["sign_trials"] = nonzero.astype(int)
            rows["sign_agree_count"] = (nonzero & (np.sign(prediction) == np.sign(total))).astype(int)
            rows["near_zero_total_frames"] = (covered & ~nonzero).astype(int)
            pieces.append(rows.groupby(PAIR_KEYS, dropna=False, sort=False, as_index=False)[names].sum())
    if not pieces:
        return pd.DataFrame(columns=PAIR_COLUMNS), len(changed)-matched
    table = pd.concat(pieces, ignore_index=True).groupby(
        PAIR_KEYS, dropna=False, sort=True, as_index=False)[names].sum()
    for name in ["paired_frames", "covered_frames", "sign_trials", "sign_agree_count", "near_zero_total_frames"]:
        table[name] = table[name].astype(int)
    table["frame_count"] = table.covered_frames
    table["curve_coverage"] = table.covered_frames/table.paired_frames
    denominator = table.covered_frames.replace(0, np.nan)
    table["curve_rmse"] = np.sqrt(table.error_sse/denominator)
    table["curve_mae"] = table.absolute_error_sum/denominator
    table["total_rms_covered"] = np.sqrt(table.total_sse/denominator)
    table["total_rms_all"] = np.sqrt(table.total_sse_all/table.paired_frames)
    table["sign_agreement"] = table.sign_agree_count/table.sign_trials.replace(0, np.nan)
    table["curve_response_gap"] = np.divide(table.curve_rmse, table.total_rms_covered,
        out=np.full(len(table), np.nan), where=table.total_rms_covered.to_numpy() > epsilon)
    return table[PAIR_COLUMNS], len(changed)-matched


def _support(model, X, meta, coords, device, batch, sample_limit, out):
    k, records, spline_records = len(coords.zmean), [], []
    layers = [(name, layer) for name, layer in model.named_modules()
              if isinstance(layer, (SplineEdgeLayer, BSplineActivation))]
    if sample_limit < 1:
        raise ValueError("spline_support_samples must be positive")
    for key, population in meta.groupby(["dataset", "split", "condition"], sort=True):
        info = dict(zip(["dataset", "split", "condition"], key))
        indices = population.index.to_numpy()
        if coords.use_covariates:
            low = (np.asarray(coords.clow)-coords.cmean)/coords.cstd
            high = (np.asarray(coords.chigh)-coords.cmean)/coords.cstd
            n = X.shape[1]-k
            outside, excess, largest, joint = np.zeros(n), np.zeros(n), np.zeros(n), 0
            for begin in range(0, len(indices), batch):
                values = X[indices[begin:begin+batch], k:].astype(np.float64)
                over = np.maximum(low-values, 0)+np.maximum(values-high, 0)
                mask = over > 0
                outside += mask.sum(0)
                excess += over.sum(0)
                largest = np.maximum(largest, over.max(0))
                joint += int(mask.any(1).sum())
            for j in range(n):
                records.append({**info, "feature": selected_covariate_names(coords)[j], "feature_index": k+j,
                    "rows": len(indices), "outside_rows": int(outside[j]),
                    "outside_fraction": float(outside[j]/len(indices)), "low": float(low[j]),
                    "high": float(high[j]), "mean_overshoot_standardized": float(excess[j]/len(indices)),
                    "max_overshoot_standardized": float(largest[j]),
                    "definition": "clean-train marginal 1%-99% interval; not spline grid or manifold"})
            records.append({**info, "feature": "all_covariates", "feature_index": -1,
                "rows": len(indices), "outside_rows": joint, "outside_fraction": joint/len(indices),
                "definition": "outside any clean-train marginal 1%-99% interval; not manifold support"})
        if not layers:
            continue
        chosen = indices[np.linspace(0, len(indices)-1, min(sample_limit, len(indices)), dtype=int)]
        stats = {}
        handles = []
        for name, layer in layers:
            def hook(module, inputs, name=name):
                values = inputs[0].detach().cpu().numpy().astype(np.float64)
                excess = np.maximum(module.low-values, 0)+np.maximum(values-module.high, 0)
                if name not in stats:
                    stats[name] = [0, np.zeros(values.shape[1]), np.zeros(values.shape[1]),
                                   np.zeros(values.shape[1]), np.zeros(values.shape[1])]
                item = stats[name]
                item[0] += len(values)
                item[1] += (values < module.low).sum(0)
                item[2] += (values > module.high).sum(0)
                item[3] += excess.sum(0)
                item[4] = np.maximum(item[4], excess.max(0))
            handles.append(layer.register_forward_pre_hook(hook))
        try:
            for begin in range(0, len(chosen), batch):
                _predict(model, X[chosen[begin:begin+batch]], device)
        finally:
            for handle in handles:
                handle.remove()
        for name, layer in layers:
            count, below, above, excess, largest = stats[name]
            for j in range(len(below)):
                spline_records.append({**info, "layer": name, "module": type(layer).__name__,
                    "input_index": j, "population_rows": len(indices), "sampled_rows": count,
                    "low": layer.low, "high": layer.high, "below_fraction": below[j]/count,
                    "above_fraction": above[j]/count, "outside_fraction": (below[j]+above[j])/count,
                    "mean_overshoot": excess[j]/count, "max_overshoot": largest[j],
                    "sampling": "deterministic evenly spaced rows per dataset/split/condition",
                    "definition": "actual module input outside spline argument grid; residual may extrapolate"})
    pd.DataFrame(records, columns=["dataset", "split", "condition", "feature", "feature_index", "rows",
        "outside_rows", "outside_fraction", "low", "high", "mean_overshoot_standardized",
        "max_overshoot_standardized", "definition"]).to_csv(Path(out)/"response_support.csv", index=False)
    pd.DataFrame(spline_records, columns=["dataset", "split", "condition", "layer", "module", "input_index",
        "population_rows", "sampled_rows", "low", "high", "below_fraction", "above_fraction",
        "outside_fraction", "mean_overshoot", "max_overshoot", "sampling", "definition"]).to_csv(
            Path(out)/"spline_support.csv", index=False)
    return {"status": "computed" if coords.use_covariates else "not_applicable",
            "rows": len(X), "definition": "marginal quantile intervals, not manifold support"}, {
            "status": "computed" if layers else "not_applicable", "layers": len(layers),
            "samples_per_stratum_maximum": sample_limit,
            "sampling": "deterministic evenly spaced rows; quantile diagnostics use all rows"}


def audit_responses(model, X, meta, coords, cfg, out, held_out=None, logits=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta = meta.reset_index(drop=True)
    X = np.asarray(X)
    if X.ndim != 2 or len(X) != len(meta):
        raise ValueError("Finite, aligned [rows,features] values required")
    if any(not np.isfinite(X[start:start+16384]).all() for start in range(0, len(X), 16384)):
        raise ValueError("Finite, aligned [rows,features] values required")
    if logits is not None:
        logits = np.asarray(logits, dtype=np.float64)
        if logits.shape != (len(X),) or not np.isfinite(logits[meta.split == "test"]).all():
            raise ValueError("Supplied logits must align with all rows and be finite on test rows")
    device, options = str(cfg.train.device), cfg.get("eval", {})
    batch = int(options.get("audit_batch_size", 4096))
    maximum = int(options.get("interpretation_samples", 512))
    epsilon = float(options.get("response_zero_epsilon", 1e-8))
    if batch < 1 or maximum < 1 or epsilon <= 0:
        raise ValueError("Positive batch size, background size, and response epsilon required")
    was_training = model.training
    model.eval()
    try:
        support, splines = _support(model, X, meta, coords, device, batch,
                                    int(options.get("spline_support_samples", 512)), out)
        if not coords.use_covariates:
            pd.DataFrame(columns=PAIR_COLUMNS).to_csv(out/"curve_intervention.csv", index=False)
            pd.DataFrame(columns=["feature", "grid", "response"]).to_csv(out/"curve_predictor.csv", index=False)
            unavailable = {"status": "not_applicable", "reason": "covariates disabled"}
            return {"curve_predictor": unavailable, "curve_response": unavailable,
                    "covariate_support": support, "spline_support": splines}
        k = len(coords.zmean)
        if X.shape[1]-k != len(selected_covariate_names(coords)):
            raise ValueError("Observed covariate count disagrees with saved physical column selection")
        seen = set(meta.loc[train_mask(meta, held_out), "method"])
        background_mask = validation_mask(meta, held_out) & meta.method.isin(seen).to_numpy()
        indices = np.flatnonzero(background_mask)
        if not len(indices):
            raise ValueError("No source validation background from seen training methods")
        indices = indices[np.linspace(0, len(indices)-1, min(maximum, len(indices)), dtype=int)]
        background = torch.as_tensor(X[indices], device=device, dtype=torch.float32)
        grids = source_curve_grids(X, meta, coords, int(options.get("grid_points", 31)), held_out)
        curves, rows = [], []
        for j, grid in enumerate(grids):
            values = response_curves(model, background, [k+j],
                torch.tensor(grid, dtype=torch.float32, device=device), batch_size=min(batch, maximum))
            response = values["curves"][str(k+j)]["mean"].astype(np.float64)
            curves.append(response)
            rows.extend({"feature": selected_covariate_names(coords)[j], "feature_index": k+j, "grid": float(x),
                         "response": float(y), "background_rows": len(indices),
                         "selection": "FF++ clean seen validation; grid from source train/validation only"}
                        for x, y in zip(grid, response))
        pd.DataFrame(rows).to_csv(out/"curve_predictor.csv", index=False)
        table, unmatched = _curve_pairs(model, X, meta, k, grids, curves, device, batch, epsilon, logits)
        table.to_csv(out/"curve_intervention.csv", index=False)
        populations = []
        for (dataset, condition), group in table.groupby(["dataset", "condition"], sort=True):
            populations.append({"dataset": dataset, "condition": condition,
                                **summarize_curve_response(group, epsilon)})
        overall = summarize_curve_response(table, epsilon)
        datasets = [{"dataset": dataset, **summarize_curve_response(group, epsilon)}
                    for dataset, group in table.groupby("dataset", sort=True)]
        methods = [{"dataset": dataset, "method": method, **summarize_curve_response(group, epsilon)}
                   for (dataset, method), group in table.groupby(["dataset", "method"], sort=True)]
        classes = [{"dataset": dataset, "label": int(label), **summarize_curve_response(group, epsilon)}
                   for (dataset, label), group in table.groupby(["dataset", "label"], sort=True)]
        return {"curve_predictor": {"status": "computed", "background_rows": len(indices),
                    "background": "FF++ clean seen validation only",
                    "grid_selection": "clean train 1%-99%, extended by seen source train/val degradation rows",
                    "outside_grid_policy": "changed outside-grid endpoints unavailable; exactly unchanged finite coordinates cancel",
                    "coverage_rule_version": CURVE_COVERAGE_VERSION, "coverage_rule": CURVE_COVERAGE_RULE,
                    "formula": "sum_j(r_j(changed_c_j)-r_j(clean_c_j))", "causal_interpretation": False},
                "curve_response": {**overall, "unmatched_changed_frames": unmatched,
                                   "by_dataset_condition": populations, "by_dataset": datasets,
                                   "by_dataset_method": methods, "by_dataset_class": classes},
                "covariate_support": support, "spline_support": splines}
    finally:
        model.train(was_training)