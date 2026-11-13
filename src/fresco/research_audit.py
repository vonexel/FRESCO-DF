"""Read-only retrospective comparison of saved FRESCO runs.

Original metrics, reports, completion records and LATEST pointers are never
rewritten. The resampling below treats training seeds and test clusters as
crossed factors; it does not estimate uncertainty over new training datasets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from .statistics import _auc_plan, _weighted_auc


class AuditInputError(ValueError):
    """A saved contrast is incomplete or cannot be paired without selection."""


def _arrays(first, second, groups):
    a, b = np.atleast_2d(np.asarray(first, float)), np.atleast_2d(np.asarray(second, float))
    if (a.ndim != 2 or b.ndim != 2 or a.shape != b.shape or a.shape[1] == 0
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("Require aligned, nonempty, finite [seed,video] arrays")
    g = np.asarray(groups)
    if g.ndim != 1 or len(g) != a.shape[1] or pd.isna(g).any():
        raise ValueError("Require one nonmissing group per paired video")
    unique, inverse = np.unique(g.astype(str), return_inverse=True)
    return a, b, unique, inverse


def _interval(values, requested, description):
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    result = {"resamples_valid": len(finite), "resamples_requested": requested,
              "uncertainty": description}
    if len(finite) < max(1, .9 * requested):
        return {**result, "status": "INCONCLUSIVE", "reason": "more than 10% degenerate draws"}
    q = np.quantile(finite, [.025, .05, .95, .975])
    return {**result, "status": "COMPUTED", "ci95": [float(q[0]), float(q[3])],
            "ci90": [float(q[1]), float(q[2])]}


def _crossed_bootstrap(statistic, seeds, inverse, clusters, resamples, seed):
    if resamples < 1:
        raise ValueError("resamples must be positive")
    baseline = statistic(np.ones(len(inverse)))
    if len(baseline) != seeds or not np.isfinite(baseline).all():
        return {"status": "INCONCLUSIVE", "reason": "undefined observed seed statistic"}
    result = {"status": "COMPUTED", "difference": float(np.mean(baseline)),
              "seed_differences": baseline.tolist(), "seed_count": seeds,
              "seed_difference_sd": float(np.std(baseline, ddof=1)) if seeds > 1 else None,
              "videos": len(inverse), "clusters": clusters,
              "estimand": "mean paired single-seed difference; no score ensemble",
              "scope": "conditional on supplied source training/validation data, selected hyperparameters and current test generator mixture",
              "interval_method": "approximate exploratory percentile bootstrap; nominal coverage is not guaranteed with ten seeds"}
    if clusters < 5:
        return {**result, "status": "INCONCLUSIVE", "reason": "fewer than five test clusters"}
    rng = np.random.default_rng(seed)
    seed_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    conditional, crossed = [], []
    for _ in range(resamples):
        counts = np.bincount(rng.integers(clusters, size=clusters), minlength=clusters)
        differences = statistic(counts[inverse])
        # The SAME sampled seed indices are used for both heads. The statistic
        # already contains their paired differences; seeds are not re-paired.
        selected = seed_rng.integers(seeds, size=seeds)
        conditional.append(np.mean(differences))
        crossed.append(np.mean(differences[selected]))
    result["conditional_clusters"] = _interval(
        conditional, resamples, "test clusters conditional on the fixed training seeds")
    result["crossed_seeds_clusters"] = (_interval(
        crossed, resamples, "paired training seeds and test clusters sampled independently")
        if seeds > 1 else {"status": "INCONCLUSIVE", "reason": "only one training seed"})
    if result["conditional_clusters"]["status"] != "COMPUTED":
        result["status"] = "INCONCLUSIVE"
    result["minimum_seed_count_met"] = seeds >= 10
    result["confirmatory_verdict"] = False
    return result


def paired_auc_uncertainty(labels, first, second, groups, *, resamples=10000, seed=2026):
    """Paired AUROC contrasts with data-only and crossed seed/data intervals."""
    a, b, unique, inverse = _arrays(first, second, groups)
    y = np.asarray(labels)
    if y.shape != (a.shape[1],) or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("Require aligned binary labels with both classes")
    plans = [(_auc_plan(y, aa), _auc_plan(y, bb)) for aa, bb in zip(a, b)]

    def statistic(weights):
        return np.array([_weighted_auc(pa, weights) - _weighted_auc(pb, weights) for pa, pb in plans])

    result = _crossed_bootstrap(statistic, len(a), inverse, len(unique), resamples, seed)
    weights = np.ones(len(y))
    result.update({"head_a_mean": float(np.mean([_weighted_auc(p[0], weights) for p in plans])),
                   "head_b_mean": float(np.mean([_weighted_auc(p[1], weights) for p in plans])),
                   "direction": "higher AUROC is better; difference is head_a minus head_b"})
    return result


def paired_gap_uncertainty(error_a, total_a, error_b, total_b, groups, *,
                           resamples=10000, seed=2026, min_total_rms=1e-6):
    """G = sqrt(mean_video(error SSE/n) / mean_video(total SSE/n)), then mean seeds.

    Inputs are per-video *mean squares*, not per-video relative errors. No
    pooling over frames or averaging of individual-video ratios is performed.
    """
    ea, eb, unique, inverse = _arrays(error_a, error_b, groups)
    ta, tb, _, _ = _arrays(total_a, total_b, groups)
    if ea.shape != ta.shape or any((v < 0).any() for v in [ea, eb, ta, tb]):
        raise ValueError("Require nonnegative aligned error and total mean squares")
    if not np.isfinite(min_total_rms) or min_total_rms < 0:
        raise ValueError("min_total_rms must be finite and nonnegative")

    def gap(error, total, weights):
        denominator = total @ weights / weights.sum()
        numerator = error @ weights / weights.sum()
        return np.sqrt(np.divide(numerator, denominator, out=np.full(len(error), np.nan),
                                 where=denominator > min_total_rms**2))

    def statistic(weights):
        return gap(ea, ta, weights) - gap(eb, tb, weights)

    result = _crossed_bootstrap(statistic, len(ea), inverse, len(unique), resamples, seed)
    w = np.ones(ea.shape[1])
    result.update({"head_a_mean": float(np.mean(gap(ea, ta, w))),
                   "head_b_mean": float(np.mean(gap(eb, tb, w))),
                   "head_a_rmse": np.sqrt(ea.mean(1)).tolist(),
                   "head_b_rmse": np.sqrt(eb.mean(1)).tolist(),
                   "head_a_total_rms": np.sqrt(ta.mean(1)).tolist(),
                   "head_b_total_rms": np.sqrt(tb.mean(1)).tolist(),
                   "min_total_rms": min_total_rms,
                   "direction": "lower G_block is better; difference is head_a minus head_b"})
    return result


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def _json(path, value):
    def clean(item):
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple, np.ndarray)):
            return [clean(v) for v in item]
        if isinstance(item, np.generic):
            return clean(item.item())
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _read(path, inventory, completion=None):
    path = Path(path)
    if not path.is_file():
        raise AuditInputError(f"Missing required saved artifact: {path}")
    sha = _sha(path)
    expected = (completion or {}).get("files", {}).get(path.name)
    if completion is not None and (expected is None or expected != sha):
        raise AuditInputError(f"Saved artifact no longer matches its DONE checksum: {path}")
    inventory[str(path.resolve())] = {"sha256": sha, "bytes": path.stat().st_size}
    return pd.read_csv(path, keep_default_na=False)


def _aligned(tables, columns, identity):
    """Fail closed instead of silently selecting the intersection of available videos."""
    reference = None
    values = []
    for description, table in tables:
        required = {"video_id", *columns, *identity}
        if not required.issubset(table.columns):
            raise AuditInputError(f"{description}: missing columns {sorted(required-set(table.columns))}")
        if table.empty or table.video_id.duplicated().any():
            raise AuditInputError(f"{description}: empty population or duplicate video ids")
        table = table.set_index("video_id").sort_index()
        current = table[list(identity)]
        if reference is None:
            reference = current
        elif not current.equals(reference):
            raise AuditInputError(f"{description}: video, label, method or cluster pairing differs")
        array = table[list(columns)].to_numpy(float)
        if not np.isfinite(array).all():
            raise AuditInputError(f"{description}: nonfinite paired values")
        values.append(array)
    return reference, np.stack(values)


def _video_response(table):
    if table.empty:
        return table
    required = {"video_id", "label", "method", "group_id", "error_sse", "total_sse", "frame_count"}
    if not required.issubset(table.columns):
        raise AuditInputError(f"Intervention artifact lacks columns {sorted(required-set(table.columns))}")
    if (table.frame_count <= 0).any():
        raise AuditInputError("Intervention frame counts must be positive")
    numeric = table[["error_sse", "total_sse", "frame_count"]].to_numpy(float)
    if not np.isfinite(numeric).all() or (numeric < 0).any():
        raise AuditInputError("Invalid intervention sums of squares or frame counts")
    for column in ["label", "method", "group_id"]:
        if (table.groupby("video_id")[column].nunique(dropna=False) != 1).any():
            raise AuditInputError(f"A video has inconsistent {column}")
    grouped = table.groupby("video_id", as_index=False).agg(
        label=("label", "first"), method=("method", "first"), group_id=("group_id", "first"),
        error_sse=("error_sse", "sum"), total_sse=("total_sse", "sum"), frame_count=("frame_count", "sum"))
    grouped["error_mean_square"] = grouped.error_sse / grouped.frame_count
    grouped["total_mean_square"] = grouped.total_sse / grouped.frame_count
    return grouped


def _strata(table, head, seed, min_total_rms):
    """Descriptive strata; method rows contain that method only, not shared real controls."""
    if table.empty:
        return []
    table = table.copy()
    table["family"] = table.condition.str.extract(r"^(jpeg|blur|resize)", expand=False).fillna("other")
    scopes = [("dataset", ["dataset"]), ("family", ["dataset", "family"]),
              ("condition", ["dataset", "condition"]), ("method", ["dataset", "method"]),
              ("class", ["dataset", "label"])]
    records = []
    for scope, keys in scopes:
        for group_key, group in table.groupby(keys, sort=True):
            group_key = group_key if isinstance(group_key, tuple) else (group_key,)
            v = _video_response(group)
            e, t = v.error_mean_square.mean(), v.total_mean_square.mean()
            record = {"scope": scope, "dataset": group_key[0], "stratum": str(group_key[-1]) if len(keys)>1 else "all",
                      "head": head, "seed": seed, "videos": len(v), "frame_conditions": int(v.frame_count.sum()),
                      "rmse": np.sqrt(e), "total_rms": np.sqrt(t),
                      "G_block": np.sqrt(e/t) if t > min_total_rms**2 else np.nan,
                      "status": "COMPUTED" if t > min_total_rms**2 else "UNAVAILABLE_NEAR_ZERO_RESPONSE"}
            if {"total", "covariate"}.issubset(group.columns):
                # Original intervention.csv has video-condition means, not
                # framewise signs. Never present this as frame sign agreement.
                mean = group.groupby("video_id")[["total", "covariate"]].mean()
                valid = mean.total.abs() > min_total_rms
                record["video_mean_sign_agreement"] = (
                    np.mean(np.sign(mean.loc[valid, "total"]) == np.sign(mean.loc[valid, "covariate"]))
                    if valid.any() else np.nan)
            if "covariate_train_support_fraction" in group:
                support = group.assign(_supported=group.covariate_train_support_fraction*group.frame_count)
                s = support.groupby("video_id")[["_supported", "frame_count"]].sum()
                record["train_quantile_box_fraction"] = (s._supported/s.frame_count).mean()
            records.append(record)
    return records


def _convergence(path, head, seed, budget, cfg, inventory, completion):
    table = _read(path / "train_log.csv", inventory, completion)
    required = {"epoch", "loss", "val_video_auc", "wall_s"}
    if table.empty or not required.issubset(table.columns):
        raise AuditInputError(f"Invalid saved training history: {path}")
    if not np.isfinite(table[list(required)].to_numpy(float)).all():
        raise AuditInputError(f"Nonfinite saved training history: {path}")
    best = int(np.argmax(table.val_video_auc.to_numpy(float)))
    window = min(5, len(table))
    train = cfg.get("train", {})
    fixed_steps = budget == "params" and train.get("max_steps") is not None
    if fixed_steps:
        columns = ["step", "examples_processed", "epoch_fraction"]
        if not set(columns).issubset(table.columns):
            raise AuditInputError(f"Fixed-step training history lacks step/exposure columns: {path}")
        numeric = table[columns].to_numpy(float)
        if (not np.isfinite(numeric).all() or (numeric <= 0).any()
                or not (np.diff(table.step.to_numpy(float)) > 0).all()
                or not (np.diff(table.examples_processed.to_numpy(float)) > 0).all()
                or not np.equal(numeric[:, :2], np.floor(numeric[:, :2])).all()):
            raise AuditInputError(f"Invalid fixed-step training exposure history: {path}")
    step_observed = int(table.step.iloc[-1]) if "step" in table else None
    return {"head": head, "seed": seed, "budget": budget,
            "schedule": "fixed_steps" if fixed_steps else ("epochs" if budget == "params" else "wall_clock"),
            "validation_checks": len(table), "window_validation_checks": window,
            "epochs_observed": None if fixed_steps else len(table),
            "best_epoch": None if fixed_steps else int(table.epoch.iloc[best]),
            "optimizer_steps_observed": step_observed,
            "best_step": int(table.step.iloc[best]) if "step" in table else None,
            "last_epoch_fraction": float(table.epoch_fraction.iloc[-1]) if "epoch_fraction" in table else None,
            "best_epoch_fraction": float(table.epoch_fraction.iloc[best]) if "epoch_fraction" in table else None,
            "examples_processed": int(table.examples_processed.iloc[-1]) if "examples_processed" in table else None,
            "best_validation_auc": float(table.val_video_auc.iloc[best]),
            "last_validation_auc": float(table.val_video_auc.iloc[-1]), "train_wall_s": float(table.wall_s.iloc[-1]),
            "loss_change_last_window": float(table.loss.iloc[-1]-table.loss.iloc[-window]),
            "window_epochs": None if fixed_steps else window, "best_in_last_window": best >= len(table)-window,
            "configured_max_epochs": None if fixed_steps else train.get("epochs"),
            "hit_epoch_limit": (len(table) >= int(train.get("epochs", 0)))
                               if budget == "params" and not fixed_steps else None,
            "configured_max_steps": train.get("max_steps") if fixed_steps else None,
            "hit_step_limit": step_observed >= int(train["max_steps"]) if fixed_steps else None,
            "interpretation": ("descriptive trajectory over validation checks at optimizer steps, not epochs; "
                               "no convergence or undertraining proof" if fixed_steps else
                               "descriptive trajectory; no convergence or undertraining proof")}


def audit_experiment(experiment, output, *, head_a="kan_spline", head_b="mlp_gelu", budget="params",
                     k=16, covariates=True, held_out=None, dataset="celebdfpp", resamples=10000,
                     seed=2026, min_total_rms=1e-6):
    """Write a new exploratory audit beside, never inside, a saved experiment."""
    experiment, output = Path(experiment).resolve(), Path(output).resolve()
    if output == experiment or output.is_relative_to(experiment):
        raise ValueError("Audit output must be outside the original experiment")
    output.mkdir(parents=True, exist_ok=False)
    inventory, response_rows, convergence = {}, [], []
    result = {"analysis_status": "EXPLORATORY_RETROSPECTIVE", "status": "INCONCLUSIVE",
              "contrast": {"head_a": head_a, "head_b": head_b, "budget": budget, "k": k,
                           "covariates": covariates, "held_out": held_out, "dataset": dataset},
              "original_experiment": str(experiment), "original_artifacts_modified": False,
              "notes": ["The target benchmark has already been examined; this is not a new preregistered test.",
                        "Intervals are conditional on the supplied training dataset and training recipe.",
                        "G_block tests covariate-block sufficiency; it does not use or validate the saved PD curves.",
                        "Head timing excludes CLIP, crop extraction and feature-cache preparation.",
                        "Marginal train quantile box support is not spline-grid or joint-manifold support."]}
    try:
        manifest_path = experiment / "manifest.json"
        if not manifest_path.is_file():
            raise AuditInputError("Saved experiment manifest is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        inventory[str(manifest_path)] = {"sha256": _sha(manifest_path), "bytes": manifest_path.stat().st_size}
        cfg = manifest["config"]
        seeds = list(cfg["experiment"]["seeds"])
        if not seeds or len(seeds) != len(set(seeds)):
            raise AuditInputError("Configured seeds must be nonempty and unique")
        result.update({"seeds": seeds, "synthetic": bool(manifest.get("synthetic", False)),
                       "training_source_fingerprint": manifest.get("source")})
        scores, interventions = {head_a: [], head_b: []}, {head_a: [], head_b: []}
        if head_a == head_b:
            raise AuditInputError("Choose two different head names")
        for head in [head_a, head_b]:
            base = experiment/f"k{k}_cov{int(covariates)}"/f"heldout_{held_out or 'none'}"/head/budget
            actual = {int(p.name[4:]) for p in base.glob("seed*") if p.is_dir() and p.name[4:].isdigit()}
            if actual != set(seeds):
                raise AuditInputError(f"{head}: configured and saved seed sets differ ({seeds} versus {sorted(actual)})")
            for training_seed in seeds:
                path = base/f"seed{training_seed}"
                done = path/"DONE.json"
                if not done.is_file():
                    raise AuditInputError(f"Run is not marked complete: {path}")
                completion = json.loads(done.read_text(encoding="utf-8"))
                inventory[str(done)] = {"sha256": _sha(done), "bytes": done.stat().st_size}
                score = _read(path/"scores.csv", inventory, completion)
                score_columns = {"dataset", "split", "condition", "video_id", "label", "method",
                                 "group_id", "probability"}
                if not score_columns.issubset(score.columns):
                    raise AuditInputError(f"{path}: scores missing columns {sorted(score_columns-set(score.columns))}")
                selected = score.loc[(score.dataset==dataset)&(score.split=="test")&(score.condition=="clean")].copy()
                probability = selected.probability.to_numpy(dtype=float)
                if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
                    raise AuditInputError(f"{path}: saved probabilities must be finite and lie in [0,1]")
                if dataset=="ffpp" and held_out is not None:
                    selected = selected.loc[(selected.label==0)|(selected.method==held_out)]
                scores[head].append((f"{head}/seed{training_seed}", selected))
                intervention = _read(path/"intervention.csv", inventory, completion)
                intervention_columns = {"dataset", "condition", "video_id", "label", "method", "group_id",
                                        "error_sse", "total_sse", "frame_count"}
                if len(intervention) and not intervention_columns.issubset(intervention.columns):
                    raise AuditInputError(f"{path}: intervention missing columns {sorted(intervention_columns-set(intervention.columns))}")
                response_rows.extend(_strata(intervention, head, training_seed, min_total_rms))
                selected = intervention.loc[intervention.dataset==dataset].copy() if len(intervention) else intervention
                if dataset=="ffpp" and held_out is not None and len(selected):
                    selected = selected.loc[(selected.label==0)|(selected.method==held_out)]
                # Compare every configured image condition, not merely each
                # model's available intersection. Full paired row keys checked.
                interventions[head].append((f"{head}/seed{training_seed}", selected))
                convergence.append(_convergence(path, head, training_seed, budget, cfg, inventory, completion))
        reference, score_arrays = _aligned(scores[head_a]+scores[head_b], ["probability"], ["label", "method", "group_id"])
        n = len(seeds)
        result["auroc"] = paired_auc_uncertainty(reference.label.to_numpy(), score_arrays[:n, :, 0],
            score_arrays[n:, :, 0], reference.group_id.to_numpy(), resamples=resamples, seed=seed)
        if not covariates:
            result["G_block"] = {"status": "NOT_APPLICABLE", "reason": "covariates disabled"}
        else:
            paired = interventions[head_a]+interventions[head_b]
            key_reference = None
            video_tables = []
            expected_conditions = set(cfg.get("features", {}).get("conditions", [])) - {"clean"}
            expected_conditions = (set(cfg["experiment"].get("conditions", [])) - {"clean"}
                                   if cfg["experiment"].get("conditions") else expected_conditions)
            for description, table in paired:
                if table.empty or table.duplicated(["video_id", "condition"]).any():
                    raise AuditInputError(f"{description}: empty or duplicate intervention pairs")
                keys = table[["video_id", "condition", "label", "method", "group_id", "frame_count"]].sort_values(
                    ["video_id", "condition"]).reset_index(drop=True)
                if key_reference is None:
                    key_reference = keys
                elif not keys.equals(key_reference):
                    raise AuditInputError(f"{description}: paired frame/condition coverage differs")
                if expected_conditions and any(set(v)!=expected_conditions for _, v in table.groupby("video_id").condition):
                    raise AuditInputError(f"{description}: incomplete predeclared degradation conditions")
                video_tables.append((description, _video_response(table)))
            g_reference, arrays = _aligned(video_tables, ["error_mean_square", "total_mean_square"],
                                            ["label", "method", "group_id"])
            if not g_reference.equals(reference):
                raise AuditInputError("Clean-score and intervention video populations differ")
            result["G_block"] = paired_gap_uncertainty(arrays[:n, :, 0], arrays[:n, :, 1],
                arrays[n:, :, 0], arrays[n:, :, 1], reference.group_id.to_numpy(),
                resamples=resamples, seed=seed, min_total_rms=min_total_rms)
        result["status"] = "COMPUTED" if result["auroc"]["status"]=="COMPUTED" else "INCONCLUSIVE"
        if result["G_block"]["status"] == "INCONCLUSIVE":
            result["status"] = "INCONCLUSIVE"
    except (AuditInputError, KeyError, ValueError) as exc:
        result["reason"] = str(exc)
        result["status"] = "INCONCLUSIVE"
    response = pd.DataFrame(response_rows)
    response.to_csv(output/"response_by_seed.csv", index=False)
    if len(response):
        columns = [c for c in ["G_block", "rmse", "total_rms", "video_mean_sign_agreement", "train_quantile_box_fraction"] if c in response]
        summary = response.groupby(["scope", "dataset", "stratum", "head"])[columns].agg(["mean", "std", "count"])
        summary.columns = ["_".join(c) for c in summary.columns]
        summary.reset_index().to_csv(output/"stratified_response.csv", index=False)
    else:
        pd.DataFrame(columns=["scope", "dataset", "stratum", "head"]).to_csv(output/"stratified_response.csv", index=False)
    pd.DataFrame(convergence).to_csv(output/"convergence.csv", index=False)
    _json(output/"audit.json", result)
    _json(output/"provenance.json", {"analysis_source_sha256": _sha(Path(__file__)),
          "statistics_source_sha256": _sha(Path(__file__).with_name("statistics.py")),
          "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
          "resamples": resamples, "resampling_seed": seed, "inputs": inventory})
    lines = ["# Retrospective FRESCO research audit", "", f"Status: **{result['status']}**.", "",
             f"Contrast: `{head_a} minus {head_b}`, {budget}, k{k}, covariates={covariates}, target={dataset}.", "",
             "This is an exploratory analysis after examining the target results. It preserves the original experiment.", ""]
    if result.get("reason"):
        lines.extend([f"Pairing limitation: {result['reason']}", ""])
    for endpoint in ["auroc", "G_block"]:
        value = result.get(endpoint, {})
        lines.extend([f"## {endpoint}", "", f"Status: {value.get('status', 'UNAVAILABLE')}", ""])
        if "difference" in value:
            lines.extend([f"Head A mean: {value.get('head_a_mean')}; head B mean: {value.get('head_b_mean')}; paired difference: {value['difference']}.",
                          f"Seed-difference SD: {value.get('seed_difference_sd')}; seeds: {value.get('seed_count')}.", ""])
        for interval in ["conditional_clusters", "crossed_seeds_clusters"]:
            item = value.get(interval, {})
            if item:
                lines.extend([f"{interval}: {item.get('status')}; 95% CI={item.get('ci95')}; {item.get('uncertainty', item.get('reason'))}.", ""])
    lines.extend(["## Interpretation limits", "", *[f"- {note}" for note in result["notes"]], "",
                  "`stratified_response.csv` contains means, training-seed SDs and counts, not inferential intervals.",
                  "Method strata contain the indicated method only; they do not add shared real controls.",
                  "Video-mean sign agreement is not framewise sign agreement. G near a zero response is unavailable.",
                  "Training histories diagnose trajectories; reaching the epoch cap alone does not prove undertraining.",
                  "No primary hypothesis verdict is replaced by this audit; absence of significance is not equivalence."])
    (output/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory outside the original experiment")
    parser.add_argument("--head-a", default="kan_spline")
    parser.add_argument("--head-b", default="mlp_gelu")
    parser.add_argument("--budget", choices=["params", "time"], default="params")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--covariates", choices=["true", "false"], default="true")
    parser.add_argument("--held-out", default="none")
    parser.add_argument("--dataset", default="celebdfpp")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-total-rms", type=float, default=1e-6)
    args = parser.parse_args(argv)
    if args.resamples < 1:
        parser.error("--resamples must be positive")
    values = vars(args)
    values["covariates"] = args.covariates == "true"
    values["held_out"] = None if args.held_out.lower()=="none" else args.held_out
    result = audit_experiment(**values)
    print(f"Audit {result['status']}: {(args.output/'REPORT.md').resolve()}")
    return 0 if result["status"] == "COMPUTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
