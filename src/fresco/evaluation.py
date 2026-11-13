"""Held-out evaluation: frozen validation choices, paired responses and novelty.

All score directions and populations are explicit. Real/fake classification,
unseen-manipulation ranking, and cross-dataset shift are different endpoints.
No test outcome selects a temperature, threshold, curve, or symbolic surrogate.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

from .heads import benchmark, count_params
from .interpret import intervention_decomposition, plasticity_score, response_curves
from .statistics import metrics, reliability, validation_threshold
from .training import predict, train_mask, validation_mask, video_predictions, selected_covariate_names
from .utils import atomic_json


COVARIATE_NAMES = ("jpeg_blockiness_proxy", "log_laplacian_variance", "face_size",
                   "high_frequency_ratio")
VIDEO_KEYS = ["dataset", "condition", "video_id", "label", "split", "method", "scenario", "group_id"]


def fit_video_temperature(meta: pd.DataFrame, logits: np.ndarray) -> tuple[float, dict]:
    """Minimize equal-video BCE of mean(sigmoid(frame_logit / temperature)).

    Optimizing sigmoid(mean(logit)/T) is a different objective and is not used.
    Caller supplies only disjoint source validation rows, with seen methods.
    """
    meta = meta.reset_index(drop=True)
    logits = np.asarray(logits, dtype=float)
    if len(meta) != len(logits) or not len(meta) or meta.label.nunique() != 2:
        raise ValueError("Need aligned, two-class validation metadata and logits")
    ids = meta.groupby(["dataset", "video_id"], sort=False).ngroup().to_numpy()
    count = np.bincount(ids)
    labels = np.bincount(ids, weights=meta.label.to_numpy()) / count
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("A video cannot have mixed labels")

    def objective(log_temperature):
        probability = np.bincount(ids, weights=expit(logits / np.exp(log_temperature))) / count
        probability = np.clip(probability, 1e-12, 1 - 1e-12)
        return float(-np.mean(labels * np.log(probability) + (1-labels) * np.log1p(-probability)))

    optimum = minimize_scalar(objective, bounds=(-3, 3), method="bounded")
    if not optimum.success:
        raise RuntimeError("Validation-only temperature optimization failed")
    temperature = float(np.exp(optimum.x))
    return temperature, {"temperature": temperature, "validation_video_nll_before": objective(0),
                         "validation_video_nll_after": objective(optimum.x), "videos": len(count),
                         "objective": "BCE(label_video, mean_frames sigmoid(logit_frame/T))",
                         "log_temperature_bounds": [-3, 3], "selection": "FF++ validation clean seen methods only"}


def _metric_records(table, level, thresholds):
    records, bins = [], []
    test = table.loc[table.split == "test"]
    for (dataset, condition), group in test.groupby(["dataset", "condition"], sort=True):
        methods = sorted(group.loc[group.label == 1, "method"].unique())
        populations = [("all", group)] + [(m, group.loc[(group.label == 0) | (group.method == m)]) for m in methods]
        for method, population in populations:
            for calibration, column in [("raw", "probability"), ("temperature", "calibrated")]:
                result = metrics(population.label, population[column], thresholds[level][calibration])
                if level == "frame":
                    # Frames from a video are dependent; do not report an iid frame CI.
                    result.pop("auc_ci_delong_video", None)
                records.append({"dataset": str(dataset), "condition": str(condition), "level": level,
                                "method": str(method), "calibration": calibration, **result})
                if method == "all":
                    bins.extend({"dataset": dataset, "condition": condition, "level": level,
                                 "calibration": calibration, **row}
                                for row in reliability(population.label.to_numpy(), population[column].to_numpy()))
    return records, bins


def _pd(model, X, meta, coords, cfg, device, out, held_out):
    columns = ["feature", "feature_index", "grid_type", "grid", "response", "raw_response", "q05", "q95",
               "background_rows", "within_train_support"]
    if not coords.use_covariates:
        pd.DataFrame(columns=columns).to_csv(out / "pd_curves.csv", index=False)
        return {"status": "not_applicable", "reason": "covariates disabled"}
    tm, vm = train_mask(meta, held_out), validation_mask(meta, held_out)
    indices = np.flatnonzero(vm)
    maximum = int(cfg.eval.get("interpretation_samples", 512))
    if maximum < 1:
        raise ValueError("interpretation_samples must be positive")
    indices = indices[np.linspace(0, len(indices)-1, min(maximum, len(indices)), dtype=int)]
    background = torch.as_tensor(X[indices], device=device, dtype=torch.float32)
    k, points = len(coords.zmean), int(cfg.eval.get("grid_points", 31))
    if points < 2:
        raise ValueError("grid_points must be >= 2")
    records = []
    for j in range(X.shape[1]-k):
        index = k + j
        low, high = np.quantile(X[tm, index], [.01, .99])
        for kind, values in [("train_quantile", np.linspace(low, high, points)),
                             ("shared_standardized", np.linspace(-2, 2, points))]:
            curve = response_curves(model, background, [index],
                                    torch.tensor(values, dtype=torch.float32, device=device))["curves"][str(index)]
            for g, value in enumerate(values):
                records.append({"feature": selected_covariate_names(coords)[j], "feature_index": index, "grid_type": kind,
                                "grid": float(value), "response": float(curve["mean"][g]),
                                "raw_response": float(curve["raw_mean"][g]), "q05": float(curve["q05"][g]),
                                "q95": float(curve["q95"][g]), "background_rows": len(indices),
                                "within_train_support": bool(low <= value <= high)})
    pd.DataFrame(records, columns=columns).to_csv(out / "pd_curves.csv", index=False)
    return {"status": "computed", "background": "FF++ validation clean seen only; deterministic evenly spaced rows",
            "background_rows": len(indices), "features": X.shape[1]-k, "grid_points": points,
            "response_scale": "logit change", "interval": "individual response quantiles, not confidence interval",
            "scope": "prediction partial dependence; hybrid rows may be off manifold"}


def _interventions(model, X, meta, coords, device, out):
    columns = ["dataset", "condition", "video_id", "label", "method", "group_id", "frames", "frame_count", "total",
               "covariate", "embedding", "interaction", "covariate_residual_rmse", "covariate_relative_l2",
               "embedding_residual_rmse", "embedding_relative_l2", "decomposition_residual_max",
               "total_rms", "covariate_train_support_fraction", "covariate_outside_support_fraction",
               "error_sse", "total_sse", "embedding_error_sse", "paired_response_agreement_gap"]
    if not coords.use_covariates:
        pd.DataFrame(columns=columns).to_csv(out / "intervention.csv", index=False)
        return {"status": "not_applicable", "reason": "covariates disabled"}
    test = meta.loc[meta.split == "test"].copy()
    test["_row"] = np.flatnonzero((meta.split == "test").to_numpy())
    keys = ["dataset", "video_id", "frame_idx"]
    clean = test.loc[test.condition == "clean", keys + ["_row"]].rename(columns={"_row": "_clean_row"})
    if clean.duplicated(keys).any() or test.duplicated(keys + ["condition"]).any():
        raise ValueError("Paired evaluation requires unique video/frame/condition rows")
    changed = test.loc[test.condition != "clean"]
    pairs = changed.merge(clean, on=keys, how="inner", validate="many_to_one")
    k = len(coords.zmean)
    support_low = (coords.clow - coords.cmean) / coords.cstd
    support_high = (coords.chigh - coords.cmean) / coords.cstd
    frames = []
    for start in range(0, len(pairs), 4096):
        part = pairs.iloc[start:start+4096]
        clean_x = torch.as_tensor(X[part._clean_row.to_numpy()], dtype=torch.float32, device=device)
        changed_array = X[part._row.to_numpy()]
        changed_x = torch.as_tensor(changed_array, dtype=torch.float32, device=device)
        values = intervention_decomposition(model, clean_x, changed_x, k)
        frame = part[["dataset", "condition", "video_id", "label", "method", "group_id"]].copy()
        for key in ["total", "covariate", "embedding", "interaction"]:
            frame[key] = values[key]
        frame["_cov_error_sq"] = (values["total"]-values["covariate"]) ** 2
        frame["_emb_error_sq"] = (values["total"]-values["embedding"]) ** 2
        frame["_total_sq"] = values["total"] ** 2
        frame["_closure"] = abs(values["total"]-values["covariate"]-values["embedding"]-values["interaction"])
        frame["_support"] = ((changed_array[:, k:] >= support_low) & (changed_array[:, k:] <= support_high)).all(1)
        frames.append(frame)
    if frames:
        raw = pd.concat(frames, ignore_index=True)
        grouped = raw.groupby(["dataset", "condition", "video_id", "label", "method", "group_id"], as_index=False).agg(
            frames=("total", "size"), total=("total", "mean"), covariate=("covariate", "mean"),
            embedding=("embedding", "mean"), interaction=("interaction", "mean"),
            _cov_error_sq=("_cov_error_sq", "mean"), _emb_error_sq=("_emb_error_sq", "mean"),
            _total_sq=("_total_sq", "mean"), decomposition_residual_max=("_closure", "max"),
            covariate_train_support_fraction=("_support", "mean"))
        grouped["total_rms"] = np.sqrt(grouped._total_sq)
        grouped["frame_count"] = grouped.frames
        grouped["error_sse"] = grouped._cov_error_sq * grouped.frames
        grouped["total_sse"] = grouped._total_sq * grouped.frames
        grouped["embedding_error_sse"] = grouped._emb_error_sq * grouped.frames
        grouped["covariate_outside_support_fraction"] = 1-grouped.covariate_train_support_fraction
        for prefix, source in [("covariate", "_cov_error_sq"), ("embedding", "_emb_error_sq")]:
            grouped[f"{prefix}_residual_rmse"] = np.sqrt(grouped[source])
            # Undefined relative errors for zero response are missing, not forced to zero.
            grouped[f"{prefix}_relative_l2"] = np.divide(np.sqrt(grouped[source]), grouped.total_rms,
                out=np.full(len(grouped), np.nan), where=grouped.total_rms.to_numpy() > 1e-8)
        grouped["paired_response_agreement_gap"] = grouped.covariate_relative_l2
        grouped[columns].to_csv(out / "intervention.csv", index=False)
        result = {"status": "computed", "paired_frames": len(raw), "paired_video_conditions": len(grouped),
                  "unmatched_changed_frames": len(changed)-len(pairs),
                  "max_decomposition_residual": float(grouped.decomposition_residual_max.max()),
                  "mean_covariate_residual_rmse": float(grouped.covariate_residual_rmse.mean()),
                  "mean_embedding_residual_rmse": float(grouped.embedding_residual_rmse.mean()),
                  "pooled_paired_response_agreement_gap": float(np.sqrt(grouped.error_sse.sum()/grouped.total_sse.sum()))
                    if grouped.total_sse.sum() > 1e-12 else None,
                  "covariate_train_support_fraction": float(raw._support.mean())}
    else:
        pd.DataFrame(columns=columns).to_csv(out / "intervention.csv", index=False)
        result = {"status": "unavailable", "reason": "no matched clean/changed test frames",
                  "unmatched_changed_frames": len(changed)}
    return {**result, "relative_error_definition": "sqrt(mean_frame((total-component)^2))/sqrt(mean_frame(total^2)); per video",
            "scope": "agreement of covariate-only model response with total observed paired response; not automatically a false explanation",
            "causal_interpretation": False, "support": "all observed covariates in train 1%-99% box; box is not manifold support"}


def _mahalanobis(X, train_indices, labels):
    models = []
    for label in (0, 1):
        idx = train_indices[labels[train_indices] == label]
        if len(idx) < 2:
            raise ValueError("Mahalanobis needs >=2 clean training frames per class")
        models.append(LedoitWolf().fit(np.asarray(X[idx], dtype=np.float64)))
    result = np.empty(len(X), dtype=np.float64)
    for start in range(0, len(X), 4096):
        chunk = np.asarray(X[start:start+4096], dtype=np.float64)
        result[start:start+len(chunk)] = np.sqrt(np.maximum(0, np.minimum.reduce([m.mahalanobis(chunk) for m in models])))
    return result


def _openset(model, initial_state, X, meta, logits, device, out, held_out):
    tm = train_mask(meta, held_out)
    test = meta.loc[meta.split == "test"].copy()
    test_indices = np.flatnonzero((meta.split == "test").to_numpy())
    lp = logits[test_indices]
    probabilities = expit(lp)
    test["msp"] = np.minimum(probabilities, 1-probabilities)
    test["energy_raw"] = -np.logaddexp(0, lp)
    test["energy_centered"] = -np.logaddexp(-lp/2, lp/2)
    mahalanobis = _mahalanobis(X, np.flatnonzero(tm), meta.label.to_numpy())
    test["mahalanobis"] = mahalanobis[test_indices]
    names = ["msp", "energy_raw", "energy_centered", "mahalanobis"]
    plasticity_status = {"status": "unsupported", "reason": "head has no explicit edge functions"}
    if any(callable(getattr(m, "edges", None)) for m in model.modules()):
        if initial_state is None:
            raise ValueError("Explicit edge head requires its saved initialization for plasticity")
        initial = copy.deepcopy(model)
        initial.load_state_dict(initial_state, strict=True)
        initial.to(device).eval()
        values = []
        for start in range(0, len(test_indices), 4096):
            chunk = torch.as_tensor(X[test_indices[start:start+4096]], dtype=torch.float32, device=device)
            values.append(-plasticity_score(model, initial, chunk))
        test["plasticity_negative"] = np.concatenate(values) if values else np.array([])
        names.append("plasticity_negative")
        del initial
        plasticity_status = {"status": "computed", "definition": "negative median absolute first-layer edge change from saved initialization",
                             "full_canevaro_pipeline_reproduced": False, "hypothesis_pass_inferred": False}
    grouped = test.groupby(VIDEO_KEYS, as_index=False, dropna=False)[names].mean()
    grouped.to_csv(out / "openset_scores.csv", index=False)
    records = []
    for condition, group in grouped.groupby("condition", sort=True):
        ff = (group.dataset == "ffpp") & (group.label == 1)
        seen_methods = set(meta.loc[train_mask(meta, held_out) & (meta.label == 1).to_numpy(), "method"])
        known = ff & group.method.isin(seen_methods)
        populations = []
        if held_out is not None:
            populations.append(("FF++ seen fake vs held-out fake", known, ff & (group.method == held_out),
                                "unseen manipulation within FF++; no real controls in novelty endpoint"))
        if "FaceShifter" not in seen_methods and (ff & (group.method == "FaceShifter")).any():
            populations.append(("FF++ seen fake vs unseen FaceShifter fake", known,
                                ff & (group.method == "FaceShifter"),
                                "same-dataset unseen method; FaceShifter excluded from all training/validation"))
        for target in sorted(set(group.dataset) - {"ffpp"}):
            populations.append((f"FF++ fake vs {target} fake", known, (group.dataset == target) & (group.label == 1),
                                "dataset and manipulation-method shift confounded; not pure open-set evidence"))
        for comparison, id_mask, unknown_mask, confounding in populations:
            selected = id_mask | unknown_mask
            y = unknown_mask[selected].to_numpy(dtype=int)
            for name in names:
                score = group.loc[selected, name].to_numpy()
                auc = float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None
                records.append({"comparison": comparison, "condition": str(condition), "score": name, "auc": auc,
                                "n_id": int(id_mask.sum()), "n_unknown": int(unknown_mask.sum()),
                                "higher_is_novel": True, "confounding": confounding})
    return {"metrics": records, "plasticity": plasticity_status,
            "mahalanobis_fit": "class-conditional Ledoit-Wolf; FF++ train clean seen only; minimum class distance",
            "energy_warning": "raw binary energy uses logits [0,l] and is class-asymmetric; centered energy uses [-l/2,l/2]",
            "unit": "mean frame novelty score per video", "selection_on_unknown_data": False}


def _additive_surrogate(X, logits, meta, held_out, out, maximum=10000):
    """An actual additive Chebyshev logit surrogate; never replaces the classifier."""
    if X.shape[1]>256:
        result={"status":"not_applicable","reason":"Full-feature reference is not a low-dimensional interpretation model"}
        atomic_json(out/"symbolic_surrogate.json",result)
        return result
    ti, vi = np.flatnonzero(train_mask(meta, held_out)), np.flatnonzero(validation_mask(meta, held_out))
    if maximum < 2:
        raise ValueError("symbolic_samples must be >=2")
    ti = ti[np.linspace(0, len(ti)-1, min(maximum, len(ti)), dtype=int)]
    scale = np.maximum(np.quantile(np.abs(X[ti]), .99, axis=0), 1e-6)

    def design(values, degree):
        basis = np.polynomial.chebyshev.chebvander(np.asarray(values, dtype=np.float64)/scale, degree)[..., 1:]
        return np.column_stack((np.ones(len(values)), basis.reshape(len(values), -1)))

    fits = []
    for degree in range(1, 6):
        if X.shape[1]*degree + 1 >= len(ti):
            continue
        train_design = design(X[ti], degree)
        penalty = np.eye(train_design.shape[1])*1e-4
        penalty[0, 0] = 0
        coefficient = np.linalg.solve(train_design.T @ train_design + penalty, train_design.T @ logits[ti])
        errors = []
        for start in range(0, len(vi), 4096):
            idx = vi[start:start+4096]
            errors.append(design(X[idx], degree) @ coefficient - logits[idx])
        val_rmse = float(np.sqrt(np.mean(np.concatenate(errors)**2)))
        fits.append({"degree": degree, "coefficients": coefficient, "val_rmse": val_rmse,
                     "train_rmse": float(np.sqrt(np.mean((train_design @ coefficient-logits[ti])**2)))})
    if not fits:
        result = {"status": "unavailable", "reason": "too few training rows for additive surrogate"}
        atomic_json(out / "symbolic_surrogate.json", result)
        return result
    selected = min(fits, key=lambda f: (f["val_rmse"], f["degree"]))
    # Scores are computed only after degree selection; each test row is retained.
    test_indices = np.flatnonzero((meta.split == "test").to_numpy())
    predictions = np.concatenate([design(X[idx], selected["degree"]) @ selected["coefficients"]
                                 for idx in np.array_split(test_indices, max(1, int(np.ceil(len(test_indices)/4096))))])
    truth = logits[test_indices]
    surrogate_video=video_predictions(meta.loc[meta.split=="test"],predictions)
    actual_video=video_predictions(meta.loc[meta.split=="test"],truth)
    fidelity=[]
    for (dataset,condition),table in surrogate_video.groupby(["dataset","condition"]):
        original=actual_video[(actual_video.dataset==dataset)&(actual_video.condition==condition)]
        table=table.set_index("video_id").sort_index()
        original=original.set_index("video_id").sort_index()
        if table.label.nunique()==2:
            auc_original=roc_auc_score(original.label,original.probability)
            auc_surrogate=roc_auc_score(table.label,table.probability)
            fidelity.append({"dataset":dataset,"condition":condition,"classifier_auc":float(auc_original),
                             "surrogate_auc":float(auc_surrogate),"auc_drop":float(auc_original-auc_surrogate),
                             "mean_absolute_probability_error":float(np.mean(np.abs(table.probability-original.probability)))})
    variance = float(np.sum((truth-truth.mean())**2))
    result = {"status": "computed", "basis": "additive Chebyshev T1..Td per shared input coordinate, plus intercept",
              "scope": "diagnostic logit surrogate; not edge symbolification or classifier replacement",
              "fit": "train clean seen logits only; degree selected by FF++ validation clean seen logit RMSE",
              "train_rows": len(ti), "test_rows": len(test_indices), "degree": selected["degree"],
              "train_rmse": selected["train_rmse"], "val_rmse": selected["val_rmse"],
              "test_rmse": float(np.sqrt(np.mean((predictions-truth)**2))),
              "test_r2": 1-float(np.sum((predictions-truth)**2))/variance if variance > 1e-12 else None,
              "test_decision_agreement": float(np.mean((predictions >= 0) == (truth >= 0))),
              "scale": scale.tolist(), "coefficients": selected["coefficients"].tolist(),
              "validation_candidates": [{k: v for k, v in fit.items() if k != "coefficients"} for fit in fits],
              "test_video_fidelity": fidelity,
              "hypothesis_pass_inferred": False}
    atomic_json(out / "symbolic_surrogate.json", result)
    return {k: v for k, v in result.items() if k not in {"scale", "coefficients", "validation_candidates"}}


def evaluate_run(model, initial_state: dict, X: np.ndarray, meta: pd.DataFrame, coords,
                 cfg, out: Path, held_out=None) -> dict[str, Any]:
    """Evaluate one trained head; write auditable scores and diagnostic artifacts."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    X = np.asarray(X, dtype=np.float32)
    meta = meta.reset_index(drop=True)
    if X.ndim != 2 or len(X) != len(meta) or not np.isfinite(X).all():
        raise ValueError("Require finite feature rows aligned with metadata")
    if not (meta.split == "test").any():
        raise ValueError("No test rows to evaluate")
    device = str(cfg.train.device)
    model = model.to(device).float().eval()
    logits = predict(model, X, device, batch=4096)
    vm = validation_mask(meta, held_out)
    temperature, calibration = fit_video_temperature(meta.loc[vm], logits[vm])
    video = video_predictions(meta, logits, temperature)
    frames = meta.copy()
    frames["logit"], frames["probability"], frames["calibrated"] = logits, expit(logits), expit(logits/temperature)
    video.to_csv(out / "scores.csv", index=False)
    frames.loc[frames.split == "test"].to_csv(out / "frame_scores.csv", index=False)
    vv = (video.dataset == "ffpp") & (video.split == "val") & (video.condition == "clean")
    if held_out is not None:
        vv &= video.method != held_out
    thresholds = {}
    for level, table in [("frame", frames.loc[vm]), ("video", video.loc[vv])]:
        thresholds[level] = {name: validation_threshold(table.label.to_numpy(), table[column].to_numpy())
                             for name, column in [("raw", "probability"), ("temperature", "calibrated")]}
    records, bins = [], []
    for level, table in [("frame", frames), ("video", video)]:
        new_records, new_bins = _metric_records(table, level, thresholds)
        records.extend(new_records)
        bins.extend(new_bins)
    pd.DataFrame(records).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(bins).to_csv(out / "calibration.csv", index=False)
    atomic_json(out / "temperature.json", {**calibration, "thresholds": thresholds,
                                            "target_validation_fpr": .05})
    pd_result = _pd(model, X, meta, coords, cfg, device, out, held_out)
    intervention = _interventions(model, X, meta, coords, device, out)
    # This separate audit predicts changes from frozen source-validation curves.
    # The original intervention endpoint calls the full head on hybrid inputs;
    # neither endpoint is substituted for the other in the primary verdict.
    from .response_audit import audit_responses
    response_audit = audit_responses(model, X, meta, coords, cfg, out, held_out, logits=logits)
    if X.shape[1]>256:
        openset={"metrics":[],"reason":"Full-feature reference evaluates clean classification; dense covariance novelty omitted"}
        pd.DataFrame(columns=VIDEO_KEYS).to_csv(out/"openset_scores.csv",index=False)
    else:
        openset = _openset(model, initial_state, X, meta, logits, device, out, held_out)
    surrogate = _additive_surrogate(X, logits, meta, held_out, out, int(cfg.eval.get("symbolic_samples", 10000)))
    synthetic = bool(cfg.get("experiment", {}).get("synthetic", False))
    repeats = int(cfg.eval.get("benchmark_repeats", 50 if synthetic else 500))
    warmup = int(cfg.eval.get("benchmark_warmup", 20 if synthetic else 100))
    compute = {"params": count_params(model), "input_features": X.shape[1], "dtype": "float32",
               "latency_scope": "head only; frozen CLIP and preprocessing excluded"}
    for size in (1, 256):
        batch = torch.as_tensor(np.tile(X[:1], (size, 1)), device=device, dtype=torch.float32)
        compute[f"latency_batch{size}"] = benchmark(model, batch, warmup=warmup, repeats=repeats)
    return {"eval": records, "interp": {"pd": pd_result, "intervention": intervention,
            "response_audit": response_audit, "symbolic": surrogate},
            "openset": openset, "compute": compute, "temperature": {**calibration, "thresholds": thresholds}}
