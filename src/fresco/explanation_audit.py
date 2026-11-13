"""Replay a frozen explanation ladder on identical, source-grid-covered pairs.

A = additive global PD, B = sum of local clean-anchored ICE effects,
C = joint covariate response with clean representation, D = observed response.
The telescoping remainders B-A, C-B, D-C diagnose context averaging,
covariate interactions, and omitted representation change/interactions. They
are model diagnostics, not identified physical causal effects or edge attributions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import average_precision_score, roc_auc_score

from .heads import make_head
from .interpret import response_curves
from .replay_audit import restore_coordinates, select_runs, verify_replayed_logits
from .response_audit import (COVARIATE_NAMES, CURVE_COVERAGE_RULE, CURVE_COVERAGE_VERSION,
                             curve_delta, source_curve_grids)
from .run import load_cache
from .training import train_mask, validation_mask, video_predictions
from .utils import atomic_json, digest_file, environment, seed_everything


KEYS = ["dataset", "condition", "video_id", "label", "method", "scenario", "group_id"]
PREDICTORS = ("global_pd", "local_ice", "joint_covariate", "zero_change")
REMAINDERS = ("context", "covariate_interaction", "representation_remainder")
NOTES = [
    "Exploratory follow-up after inspecting Celeb-DF++; no source coordinates or head weights are fitted.",
    "All A/B/C/zero errors use exactly the same PD-covered paired frames and equal-video weighting.",
    "Coverage describes the frozen marginal curve grids, not classifier validity or joint manifold support.",
    "B and C use observed changed covariates and hybrid inputs; this is model diagnosis, not physical causality.",
    "B-A includes background/context averaging and finite-grid PD interpolation error; it is not an isolated context effect.",
    "A/B/C explain their own head's observed D; normalized errors across heads have different denominators.",
    "Squared remainder magnitudes are not additive error shares: cross-products and cancellation matter.",
    "Framewise scores averaged within videos do not constitute temporal feature learning.",
    "No confidence intervals, equivalence, architecture superiority or generator-population inference are claimed here.",
]


@torch.inference_mode()
def _predict(model, values, device):
    output = model(torch.as_tensor(np.array(values, dtype=np.float32), device=device)).reshape(-1)
    result = output.float().cpu().numpy().astype(np.float64)
    if result.shape != (len(values),) or not np.isfinite(result).all():
        raise ValueError("Expected finite single-logit predictions")
    return result


def _ladder_batch(model, clean, changed, k, grids, curves, device="cpu", clean_logits=None,
                  changed_logits=None):
    """Evaluate the ladder; return finite B/C/D and unavailable A outside its grid.

    Covariates are the trailing coordinates in the same order as saved source
    coordinates. An empty block has exactly zero response, but no PD explanation
    is claimed for a classifier trained without observed covariates.
    """
    a, b = np.asarray(clean), np.asarray(changed)
    if a.ndim != 2 or a.shape != b.shape or not 0 < k <= a.shape[1] or not len(a):
        raise ValueError("Require aligned nonempty clean/changed rows and valid embedding dimension")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite paired coordinates")
    # Hybrid effects use a same-batch clean prediction. Reusing a previously
    # batched clean logit can create a tiny false covariate effect in FP32,
    # especially when an unchanged hybrid is bitwise identical to the anchor.
    baseline = _predict(model, a, device)
    observed_clean = baseline if clean_logits is None else np.asarray(clean_logits, float)
    altered = _predict(model, b, device) if changed_logits is None else np.asarray(changed_logits, float)
    if (observed_clean.shape != (len(a),) or altered.shape != observed_clean.shape
            or not np.isfinite(observed_clean).all() or not np.isfinite(altered).all()):
        raise ValueError("Finite aligned per-frame logits required")
    global_pd, covered = curve_delta(a[:, k:], b[:, k:], grids, curves)
    local = np.zeros(len(a), dtype=np.float64)
    for index in range(k, a.shape[1]):
        hybrid = a.copy()
        hybrid[:, index] = b[:, index]
        effect = _predict(model, hybrid, device)-baseline
        effect[a[:, index] == b[:, index]] = 0.
        local += effect
    hybrid = a.copy()
    hybrid[:, k:] = b[:, k:]
    joint = _predict(model, hybrid, device)-baseline
    joint[np.all(a[:, k:] == b[:, k:], axis=1)] = 0.
    actual = altered-observed_clean
    return {"global_pd": global_pd, "local_ice": local, "joint_covariate": joint,
            "zero_change": np.zeros(len(a)), "actual": actual, "covered": covered,
            "context": local-global_pd, "covariate_interaction": joint-local,
            "representation_remainder": actual-joint}


def ladder_batch(model, clean, changed, k, grids, curves, device="cpu", clean_logits=None,
                 changed_logits=None):
    """Evaluate deterministic frozen-model effects while preserving caller mode."""
    was_training = model.training
    model.eval()
    try:
        return _ladder_batch(model, clean, changed, k, grids, curves, device,
                             clean_logits, changed_logits)
    finally:
        model.train(was_training)


def fit_source_curves(model, X, meta, coords, *, device="cpu", maximum=512, points=31,
                      batch=4096, held_out=None):
    """Freeze grids and PD using seen FF++ train/validation only."""
    k = len(coords.zmean)
    if X.shape[1] == k:
        return [], [], pd.DataFrame(columns=["feature", "feature_index", "grid", "response"]), []
    if maximum < 1 or points < 2 or batch < 1:
        raise ValueError("Positive batch/background size and at least two grid points required")
    seen = set(meta.loc[train_mask(meta, held_out), "method"])
    background_mask = validation_mask(meta, held_out) & meta.method.isin(seen).to_numpy()
    rows = np.flatnonzero(background_mask)
    if not len(rows):
        raise ValueError("No source-only validation background")
    rows = rows[np.linspace(0, len(rows)-1, min(maximum, len(rows)), dtype=int)]
    background = torch.as_tensor(np.array(X[rows]), dtype=torch.float32, device=device)
    grids = source_curve_grids(X, meta, coords, points, held_out)
    indices = np.asarray(getattr(coords, "covariate_indices", np.arange(4)), dtype=int)
    if len(indices) != X.shape[1]-k:
        raise ValueError("Saved covariate selection and transformed input disagree")
    records, curves = [], []
    for j, grid in enumerate(grids):
        result = response_curves(model, background, [k+j],
            torch.tensor(grid, dtype=torch.float32, device=device), batch_size=min(batch, maximum))
        curve = result["curves"][str(k+j)]["mean"].astype(np.float64)
        curves.append(curve)
        records.extend({"feature": COVARIATE_NAMES[indices[j]], "feature_index": k+j,
            "raw_covariate_index": int(indices[j]), "grid": float(g), "response": float(v),
            "background_rows": len(rows)} for g, v in zip(grid, curve))
    return grids, curves, pd.DataFrame(records), rows.tolist()


def _paired_rows(meta):
    keys = ["dataset", "video_id", "frame_idx"]
    attributes = ["label", "method", "scenario", "group_id"]
    clean = meta.loc[(meta.split == "test") & (meta.condition == "clean"), keys+attributes].copy()
    clean["clean_row"] = clean.index
    changed = meta.loc[(meta.split == "test") & (meta.condition != "clean")].copy()
    changed["changed_row"] = changed.index
    if clean.duplicated(keys).any() or changed.duplicated(keys+["condition"]).any():
        raise ValueError("Duplicate paired clean/degraded frame keys")
    paired = changed.merge(clean, on=keys, how="left", suffixes=("", "_clean"), validate="many_to_one")
    if paired.clean_row.isna().any():
        raise ValueError("Every changed test frame requires its clean counterpart")
    for attr in attributes:
        if not paired[attr].equals(paired[attr+"_clean"]):
            raise ValueError(f"Paired frame provenance mismatch: {attr}")
    paired["clean_row"] = paired.clean_row.astype(np.int64)
    return paired


def pair_statistics(model, X, meta, k, grids, curves, *, device="cpu", batch=4096,
                    epsilon=1e-8, logits=None):
    """Aggregate video x condition sufficient statistics, retaining common coverage."""
    if batch < 1 or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Positive batch and finite positive response epsilon required")
    paired = _paired_rows(meta)
    pieces = []
    for start in range(0, len(paired), batch):
        part = paired.iloc[start:start+batch]
        a, b = part.clean_row.to_numpy(), part.changed_row.to_numpy()
        result = ladder_batch(model, X[a], X[b], k, grids, curves, device,
            None if logits is None else logits[a], None if logits is None else logits[b])
        covered, total = result["covered"], result["actual"]
        nonzero = covered & (np.abs(total) > epsilon)
        rows = part[KEYS].copy()
        rows["paired_frames"], rows["covered_frames"] = 1, covered.astype(int)
        rows["sign_trials"] = nonzero.astype(int)
        rows["near_zero_total_frames"] = (covered & ~nonzero).astype(int)
        rows["total_sse_all"], rows["total_sse"] = total**2, np.where(covered, total**2, 0.)
        for name in PREDICTORS:
            prediction = result[name]
            error = np.where(covered, prediction-total, 0.)
            rows[name+"_error_sse"] = error**2
            rows[name+"_absolute_error_sum"] = np.abs(error)
            rows[name+"_signed_error_sum"] = error
            rows[name+"_predicted_sse"] = np.where(covered, prediction**2, 0.)
            rows[name+"_sign_agree_count"] = (nonzero & (np.sign(prediction) == np.sign(total))).astype(int)
        for name in REMAINDERS:
            value = np.where(covered, result[name], 0.)
            rows[name+"_sse"], rows[name+"_signed_sum"] = value**2, value
        for i, first in enumerate(REMAINDERS):
            for second in REMAINDERS[i+1:]:
                rows[first+"_x_"+second] = np.where(covered, result[first]*result[second], 0.)
        columns = [name for name in rows if name not in KEYS]
        pieces.append(rows.groupby(KEYS, dropna=False, sort=False, as_index=False)[columns].sum())
    if not pieces:
        return pd.DataFrame(columns=KEYS+["paired_frames", "covered_frames"])
    table = pd.concat(pieces, ignore_index=True)
    return table.groupby(KEYS, dropna=False, sort=True, as_index=False).sum(numeric_only=True)


def summarize_ladder(table, epsilon=1e-8):
    """Same covered-pair denominator and equal-video weighting for all predictors."""
    if table.empty:
        return {"status": "unavailable", "reason": "no paired degradations"}
    pairs, covered = int(table.paired_frames.sum()), int(table.covered_frames.sum())
    result = {"paired_frames": pairs, "covered_frames": covered, "coverage": covered/pairs,
        "videos": int(table[["dataset", "video_id"]].drop_duplicates().shape[0]),
        "coverage_rule_version": CURVE_COVERAGE_VERSION, "coverage_rule": CURVE_COVERAGE_RULE,
        "weighting": "equal videos; all covered frame-condition pairs within a video share its mass"}
    if not covered:
        return {**result, "status": "unavailable", "reason": "no PD-covered pairs", "predictors": {}}
    columns = [c for c in table if c not in KEYS]
    video = table.groupby(["dataset", "video_id"])[columns].sum()
    all_rms = float(np.sqrt((video.total_sse_all/video.paired_frames).mean()))
    video = video.loc[video.covered_frames > 0]
    total_mse = float((video.total_sse/video.covered_frames).mean())
    result.update({"status": "computed" if pairs == covered else "partial_coverage",
        "covered_videos": len(video), "total_rms_covered": np.sqrt(total_mse),
        "total_rms_all": all_rms, "near_zero_total_frames": int(table.near_zero_total_frames.sum()),
        "predictors": {}, "remainders": {}})
    trials = int(table.sign_trials.sum())
    for name in PREDICTORS:
        mse = float((video[name+"_error_sse"]/video.covered_frames).mean())
        result["predictors"][name] = {"gap": np.sqrt(mse/total_mse) if total_mse > epsilon**2 else None,
            "rmse": np.sqrt(mse), "mae_equal_video": float((video[name+"_absolute_error_sum"]/video.covered_frames).mean()),
            "mae_frame_weighted": float(table[name+"_absolute_error_sum"].sum()/covered),
            "sign_agreement": float(table[name+"_sign_agree_count"].sum()/trials) if trials else None,
            "sign_trials": trials}
    for name in REMAINDERS:
        result["remainders"][name] = {"rms_equal_video": float(np.sqrt((video[name+"_sse"]/video.covered_frames).mean())),
            "signed_mean_equal_video": float((video[name+"_signed_sum"]/video.covered_frames).mean())}
    cross = sum(table[a+"_x_"+b].sum() for i, a in enumerate(REMAINDERS) for b in REMAINDERS[i+1:])
    reconstructed = sum(table[name+"_sse"].sum() for name in REMAINDERS)+2*cross
    result["remainder_sse_identity_absolute_error"] = abs(float(reconstructed-table.global_pd_error_sse.sum()))
    if total_mse <= epsilon**2:
        result["normalized_gap_status"] = "unavailable: near-zero observed response"
    return result


def _strata(table, epsilon):
    result = {}
    for label, keys in [("dataset", ["dataset"]), ("condition", ["dataset", "condition"]),
                        ("generator", ["dataset", "method"]), ("class", ["dataset", "label"]),
                        ("scenario", ["dataset", "scenario"])]:
        rows = []
        for group, part in table.groupby(keys, dropna=False, sort=True):
            values = group if isinstance(group, tuple) else (group,)
            rows.append({**dict(zip(keys, values)), **summarize_ladder(part, epsilon)})
        result[label] = rows
    return result


def classification_by_group(video):
    """Raw, uncalibrated video discrimination; generator subsets share real controls."""
    records = []
    for (dataset, condition), part in video.groupby(["dataset", "condition"], sort=True):
        methods = sorted(part.loc[part.label == 1, "method"].unique())
        for method in ["all"]+methods:
            group = part if method == "all" else part.loc[(part.label == 0) | (part.method == method)]
            y, score = group.label.to_numpy(), group.probability.to_numpy()
            records.append({"dataset": dataset, "condition": condition, "method": method,
                "videos": len(group), "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum()),
                "fake_prevalence_ap_reference": float(y.mean()),
                "auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
                "ap": float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else None})
    return records


def observed_response_statistics(meta, logits, batch=4096):
    """All-pair observed response magnitudes, also meaningful without covariates."""
    paired, pieces = _paired_rows(meta), []
    for start in range(0, len(paired), batch):
        part = paired.iloc[start:start+batch]
        total = logits[part.changed_row.to_numpy()]-logits[part.clean_row.to_numpy()]
        rows = part[KEYS].copy()
        rows["paired_frames"], rows["observed_response_sse"] = 1, total**2
        rows["observed_response_signed_sum"] = total
        pieces.append(rows.groupby(KEYS, dropna=False, sort=False, as_index=False).sum(numeric_only=True))
    if not pieces:
        return pd.DataFrame(columns=KEYS+["paired_frames", "observed_response_sse", "observed_response_signed_sum"])
    table = pd.concat(pieces, ignore_index=True).groupby(KEYS, dropna=False, sort=True, as_index=False).sum(numeric_only=True)
    table["observed_response_rms"] = np.sqrt(table.observed_response_sse/table.paired_frames)
    table["observed_response_mean"] = table.observed_response_signed_sum/table.paired_frames
    return table


def _stream_coordinates(coords, features, covariates, rows, path, batch):
    """Disk-backed transformed coordinates; peak feature RAM is one batch."""
    n_cov = len(getattr(coords, "covariate_indices", np.arange(4))) if coords.use_covariates else 0
    shape = (len(rows), len(coords.zmean)+n_cov)
    needed = int(np.prod(shape))*np.dtype(np.float32).itemsize
    if shutil.disk_usage(path.parent).free < needed+64*2**20:
        raise ValueError(f"Insufficient temporary disk space for transformed coordinates ({needed} bytes)")
    X = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    try:
        for begin in range(0, len(rows), batch):
            selected = rows[begin:begin+batch]
            X[begin:begin+len(selected)] = coords.transform(features[selected], covariates[selected])
        X.flush()
    except Exception:
        X._mmap.close()
        raise
    return X, needed


def audit_saved(experiment, cache, output, *, heads=("kan_spline", "mlp_gelu"), budget="params",
                k=16, covariates=True, seeds=None, held_out=None, device="cpu",
                allow_additional_conditions=False, batch=4096):
    """Verify original inputs/predictions, then evaluate saved heads on paired probes."""
    experiment, cache, output = (Path(p).resolve() for p in [experiment, cache, output])
    if output.exists() or any(output.is_relative_to(p) or p.is_relative_to(output) for p in [experiment, cache]):
        raise ValueError("Use a new output directory outside the original experiment/cache")
    if batch < 1:
        raise ValueError("Positive batch size required")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; select --device cpu explicitly")
    seed_everything(2026)
    manifest_path = experiment/"manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cfg = OmegaConf.create(manifest["config"])
    declared = list(cfg.experiment.seeds)
    selected = declared if seeds is None else list(seeds)
    if not selected or len(set(selected)) != len(selected) or not set(selected).issubset(declared):
        raise ValueError("Select unique seeds from the original declared seeds")
    if not heads or len(set(heads)) != len(heads):
        raise ValueError("Select unique saved heads")
    common, runs = select_runs(experiment, heads, budget, k, covariates, held_out, selected)
    cache_manifest = json.loads((cache/"manifest.json").read_text(encoding="utf-8"))
    for artifact in ["features.npy", "covariates.npy", "metadata.csv"]:
        expected = manifest.get("cache", {}).get(artifact)
        if not expected or cache_manifest.get(artifact.split(".")[0]+"_sha256") != expected:
            raise ValueError(f"Cache does not match the original experiment: {artifact}")
    features, quality, all_meta = load_cache(cache)
    original_conditions = list(cfg.experiment.get("conditions") or all_meta.condition.unique())
    keep = np.ones(len(all_meta), dtype=bool) if allow_additional_conditions else all_meta.condition.isin(original_conditions).to_numpy()
    rows = np.flatnonzero(keep)
    meta = all_meta.loc[keep].reset_index(drop=True)
    original_keep = meta.condition.isin(original_conditions).to_numpy()
    coords_path = common/"coordinates.npz"
    coords = restore_coordinates(coords_path)
    if len(coords.zmean) != k or bool(coords.use_covariates) != covariates:
        raise ValueError("Saved coordinate identity disagrees with requested runs")
    _paired_rows(meta)  # Fail on broken physical pairing before creating artifacts.
    torch.set_num_threads(int(cfg.train.get("threads", 4)))
    output.mkdir(parents=True)
    temporary = output/"_transformed_coordinates.npy"
    X = None
    provenance = {"status": "RUNNING", "analysis_role": "EXPLORATORY_FOLLOWUP",
        "original_experiment": str(experiment), "cache": str(cache), "retrained": False,
        "original_outputs_modified": False, "original_manifest_sha256": digest_file(manifest_path),
        "coordinates_sha256": digest_file(coords_path), "cache_hashes": manifest["cache"],
        "declared_seeds": declared, "selected_seeds": selected,
        "complete_declared_seed_coverage": set(selected) == set(declared),
        "original_conditions": original_conditions, "evaluated_conditions": sorted(meta.condition.unique()),
        "allow_additional_conditions": allow_additional_conditions, "environment": environment(), "runs": [],
        "notes": NOTES}
    atomic_json(output/"provenance.json", provenance)
    try:
        X, size = _stream_coordinates(coords, features, quality, rows, temporary, batch)
        provenance["temporary_coordinate_bytes"] = size
        settings = cfg.get("eval", {})
        epsilon = float(settings.get("response_zero_epsilon", 1e-8))
        records = []
        for run_path, identity in runs:
            print(f"EXPLANATION LADDER {run_path}", flush=True)
            checkpoint = torch.load(run_path/"checkpoint.pt", map_location="cpu", weights_only=True)
            if checkpoint.get("head") != identity["head"] or checkpoint.get("in_features") != X.shape[1]:
                raise ValueError("Saved head identity/input dimension mismatch")
            bound = checkpoint.get("coordinates_sha256")
            if bound is not None and bound != provenance["coordinates_sha256"]:
                raise ValueError("Saved coordinates checksum does not match checkpoint")
            matching = checkpoint["matching"]
            model = make_head(identity["head"], X.shape[1], int(matching.get("hidden", 1)),
                              **matching.get("head_kwargs", {})).to(device).float().eval()
            model.load_state_dict(checkpoint["state"], strict=True)
            logits = np.empty(len(meta), dtype=np.float64)
            for begin in range(0, len(meta), batch):
                logits[begin:begin+batch] = _predict(model, X[begin:begin+batch], device)
            validation = verify_replayed_logits(run_path, meta.loc[original_keep].reset_index(drop=True),
                                                logits[original_keep])
            dest = output/identity["head"]/f"seed{identity['seed']}"
            dest.mkdir(parents=True)
            grids, curves, curve_table, background = fit_source_curves(model, X, meta, coords,
                device=device, maximum=int(settings.get("interpretation_samples", 512)),
                points=int(settings.get("grid_points", 31)), batch=batch, held_out=held_out)
            curve_table.to_csv(dest/"curve_predictor.csv", index=False)
            statistics = (pair_statistics(model, X, meta, k, grids, curves, device=device,
                                          batch=batch, epsilon=epsilon, logits=logits)
                          if covariates else pd.DataFrame(columns=KEYS+["paired_frames", "covered_frames"]))
            statistics.to_csv(dest/"ladder_by_video_condition.csv", index=False)
            observed_response_statistics(meta, logits, batch).to_csv(dest/"observed_response_by_video_condition.csv", index=False)
            video = video_predictions(meta.loc[meta.split == "test"], logits[(meta.split == "test").to_numpy()])
            video.drop(columns=["calibrated"]).to_csv(dest/"video_scores.csv", index=False)
            discrimination = classification_by_group(video)
            pd.DataFrame(discrimination).to_csv(dest/"classification_by_generator.csv", index=False)
            result = {"run": identity, "scope": ("identical PD-covered pairs for A/B/C/zero" if covariates
                                                 else "classification and observed responses; covariate ladder not applicable"),
                "covariate_ladder_applicable": bool(covariates),
                "without_covariates_note": None if covariates else "A/B/C are zero by absence of the covariate block; this is not a PD explanation test",
                "source_background_cache_rows": rows[np.asarray(background, dtype=int)].tolist(),
                "summary": (summarize_ladder(statistics, epsilon) if covariates else
                            {"status": "not_applicable", "reason": "observed covariates disabled"}),
                "strata": _strata(statistics, epsilon) if covariates else {}, "classification": discrimination}
            atomic_json(dest/"ladder.json", result)
            records.append(result)
            provenance["runs"].append({"run": identity, "checkpoint_sha256": digest_file(run_path/"checkpoint.pt"),
                                        "original_logit_reproduction": validation})
            atomic_json(output/"provenance.json", provenance)
            del model
        atomic_json(output/"audit.json", {"status": "COMPUTED", "analysis_role": "EXPLORATORY_FOLLOWUP",
            "runs": records, "notes": NOTES})
        provenance["status"] = "COMPUTED"
        atomic_json(output/"provenance.json", provenance)
        (output/"REPORT.md").write_text("\n".join(["# Frozen explanation ladder", "",
            "A: global PD; B: local ICE sum; C: joint covariate response; D: actual paired response.",
            "Context = B-A; covariate interaction = C-B; representation remainder = D-C.",
            "These signed remainders sum to D-A. Their squared magnitudes do not add without cross-products.",
            "", *["- "+note for note in NOTES], "", f"Completed saved runs: {len(records)}.",
            "See audit.json, each ladder.json and ladder_by_video_condition.csv for measurements.", ""]), encoding="utf-8")
    except Exception as exc:
        provenance.update({"status": "FAILED", "error": str(exc)})
        atomic_json(output/"provenance.json", provenance)
        raise
    finally:
        if X is not None:
            X.flush()
            X._mmap.close()
        if temporary.exists():
            temporary.unlink()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heads", nargs="+", default=["kan_spline", "mlp_gelu"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--budget", choices=["params", "time"], default="params")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--without-covariates", action="store_true")
    parser.add_argument("--held-out")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--allow-additional-conditions", action="store_true",
                        help="Evaluate additional conditions already in the exact original cache; original saved logits must still reproduce")
    args = parser.parse_args()
    result = audit_saved(args.experiment, args.cache, args.output, heads=args.heads, budget=args.budget,
        k=args.k, covariates=not args.without_covariates, seeds=args.seeds, held_out=args.held_out,
        device=args.device, allow_additional_conditions=args.allow_additional_conditions, batch=args.batch_size)
    print(f"REPORT: {result/'REPORT.md'}")


if __name__ == "__main__":
    main()
