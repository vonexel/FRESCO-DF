"""Offline exploratory AUROC audit of architecture x covariates x source recipe.

Every bootstrap draw shares test-cluster and paired-seed indices across all
eight cells. This diagnoses recipe interactions, not a physical causal mechanism.
Original experiment artifacts and primary verdicts are never rewritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .research_audit import AuditInputError, _aligned, _interval, _json, _read, _sha
from .statistics import _auc_plan, _weighted_auc


RECIPES = ("clean", "augmented")
HEADS = ("kan_spline", "mlp_gelu")
CELLS = tuple(f"{recipe}/{head}/cov{cov}" for recipe in RECIPES for head in HEADS for cov in (0, 1))


def factorial_contrasts():
    """Fixed contrasts in cell order recipe, head, covariates (off, on)."""
    names, rows = [], []
    for recipe_index, recipe in enumerate(RECIPES):
        offset = 4*recipe_index
        for cov in (0, 1):
            row = np.zeros(8)
            row[offset+cov], row[offset+2+cov] = 1, -1
            names.append(f"{recipe}: KAN minus MLP, cov{cov}")
            rows.append(row)
        for head_index, head in enumerate(HEADS):
            row = np.zeros(8)
            index = offset+2*head_index
            row[index+1], row[index] = 1, -1
            names.append(f"{recipe}: covariate effect, {head}")
            rows.append(row)
        row = np.zeros(8)
        row[offset:offset+4] = [-1, 1, 1, -1]
        names.append(f"{recipe}: architecture x covariates interaction")
        rows.append(row)
    names.append("augmentation change in architecture x covariates interaction")
    rows.append(rows[9]-rows[4])
    return names, np.stack(rows)


def factorial_auc_uncertainty(labels, scores, groups, *, resamples=10000, seed=2026):
    """Mean single-seed AUROC contrasts; scores are [2,2,2,seed,video].

    The eight AUROCs for each seed are computed once per cluster draw, then all
    eleven contrasts are obtained by a fixed linear map. No score ensemble is
    formed and seed indices are never re-paired independently between cells.
    """
    y, values, group = np.asarray(labels), np.asarray(scores, float), np.asarray(groups)
    if (values.ndim != 5 or values.shape[:3] != (2, 2, 2) or min(values.shape[3:]) < 1
            or not np.isfinite(values).all()):
        raise ValueError("Require finite [2 recipes,2 heads,2 covariates,seeds,videos] scores")
    seeds, videos = values.shape[3:]
    if y.shape != (videos,) or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("Require both binary classes aligned with all videos")
    if group.shape != (videos,) or pd.isna(group).any():
        raise ValueError("Require one nonmissing test cluster per video")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    unique, inverse = np.unique(group.astype(str), return_inverse=True)
    scores_flat = values.reshape(8, seeds, videos)
    plans = [[_auc_plan(y, values) for values in cell] for cell in scores_flat]
    names, contrasts = factorial_contrasts()

    def aucs(weights):
        return np.array([[_weighted_auc(plan, weights) for plan in cell] for cell in plans])

    observed_auc = aucs(np.ones(videos))
    observed = contrasts @ observed_auc
    result = {"status": "COMPUTED", "seed_count": seeds, "videos": videos, "clusters": len(unique),
              "analysis_status": "EXPLORATORY_RETROSPECTIVE",
              "estimand": "mean single-seed AUROC; no score ensemble",
              "scope": "conditional on supplied source training data and declared training recipes",
              "cell_order": list(CELLS), "cells": [], "contrasts": [],
              "minimum_seed_count_met": seeds >= 10, "confirmatory_verdict": False}
    for name, row in zip(CELLS, observed_auc):
        result["cells"].append({"cell": name, "mean_auc": float(row.mean()), "seed_aucs": row.tolist(),
            "seed_sd": float(row.std(ddof=1)) if seeds > 1 else None})
    conditional = np.full((resamples, len(names)), np.nan)
    crossed = np.full_like(conditional, np.nan)
    if len(unique) >= 5:
        rng = np.random.default_rng(seed)
        seed_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        for draw in range(resamples):
            weights = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))[inverse]
            # A degenerate one-class draw is unavailable in every contrast.
            cell_aucs = aucs(weights)
            if not np.isfinite(cell_aucs).all():
                continue
            differences = contrasts @ cell_aucs
            selected = seed_rng.integers(seeds, size=seeds)
            conditional[draw] = differences.mean(1)
            crossed[draw] = differences[:, selected].mean(1)
    else:
        result.update(status="INCONCLUSIVE", reason="fewer than five independent test clusters")
    for index, name in enumerate(names):
        row = observed[index]
        item = {"contrast": name, "difference": float(row.mean()), "seed_differences": row.tolist(),
                "seed_difference_sd": float(row.std(ddof=1)) if seeds > 1 else None,
                "coefficients": contrasts[index].tolist(),
                "conditional_clusters": _interval(conditional[:, index], resamples,
                    "test clusters conditional on fixed paired training seeds"),
                "crossed_seeds_clusters": _interval(crossed[:, index], resamples,
                    "paired training seeds and test clusters sampled independently") if seeds > 1 else {
                    "status": "INCONCLUSIVE", "reason": "only one paired training seed"}}
        if item["conditional_clusters"]["status"] != "COMPUTED":
            result["status"] = "INCONCLUSIVE"
        result["contrasts"].append(item)
    result["primary_diagnostic"] = "augmentation change in architecture x covariates interaction"
    result["multiple_comparisons"] = "eleven fixed exploratory contrasts; intervals are unadjusted"
    result["direction"] = ("AUROC higher is better; negative architecture x covariates interaction means "
                           "the covariate effect is less favorable for KAN than MLP")
    return result


def _saved_json(path, inventory, completion=None):
    path = Path(path)
    if not path.is_file():
        raise AuditInputError(f"Missing required saved artifact: {path}")
    digest = _sha(path)
    if completion is not None:
        expected = completion.get("files", {}).get(path.name)
        if expected is None or digest != expected:
            raise AuditInputError(f"Missing or mismatched DONE checksum: {path}")
    inventory[str(path.resolve())] = {"sha256": digest, "bytes": path.stat().st_size}
    return json.loads(path.read_text(encoding="utf-8"))


def _coordinates(path, inventory):
    if not path.is_file():
        raise AuditInputError(f"Missing shared fitted coordinates: {path}")
    inventory[str(path.resolve())] = {"sha256": _sha(path), "bytes": path.stat().st_size}
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files if name != "use_covariates"}


def _same_coordinates(reference, actual):
    if set(reference) != set(actual):
        return False
    return all(reference[name].shape == actual[name].shape
               and np.array_equal(reference[name], actual[name]) for name in reference)


def _exposure_control(train, budget):
    """Describe the declared control without upgrading epoch matching to steps."""
    if budget == "time":
        return {"status": "DECLARED_NOT_VERIFIED", "criterion": "wall clock within 10% per run",
                "mode": "wall_clock", "optimizer_exposure_matched": False}
    steps, interval = train.get("max_steps"), train.get("validation_every_steps")
    if steps is None and interval is None:
        return {"status": "NOT_MATCHED", "mode": "legacy_epochs", "optimizer_exposure_matched": False,
                "criterion": "legacy epoch recipe; row counts and update exposure can differ",
                "limitation": "The augmentation contrast also includes differences in optimizer exposure."}
    for name, value in [("max_steps", steps), ("validation_every_steps", interval),
                        ("batch_size", train.get("batch_size"))]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise AuditInputError(f"Fixed-step factorial control requires positive integer {name}")
    validation_steps = list(range(interval, steps+1, interval))
    if not validation_steps or validation_steps[-1] != steps:
        validation_steps.append(steps)
    return {"status": "DECLARED_NOT_VERIFIED", "mode": "fixed_steps",
            "optimizer_exposure_matched": False, "criterion": "optimizer steps, examples and validation schedule",
            "max_steps": steps, "batch_size": train["batch_size"],
            "examples_processed": steps*train["batch_size"], "validation_every_steps": interval,
            "validation_steps": validation_steps}


def _validate_exposure(training, control, run):
    if control["mode"] != "fixed_steps":
        return
    expected = {"schedule": "fixed_steps", "optimizer_steps": control["max_steps"],
                "requested_max_steps": control["max_steps"], "batch_size": control["batch_size"],
                "examples_processed": control["examples_processed"],
                "validation_every_steps": control["validation_every_steps"],
                "validation_steps": control["validation_steps"]}
    for name, value in expected.items():
        actual = training.get(name)
        valid = actual == value
        if isinstance(value, int):
            valid = valid and isinstance(actual, int) and not isinstance(actual, bool)
        elif isinstance(value, list):
            valid = valid and isinstance(actual, list) and all(
                isinstance(v, int) and not isinstance(v, bool) for v in actual)
        if not valid:
            raise AuditInputError(f"Fixed-step exposure mismatch for {name}: {run}")


def audit_factorial(clean, augmented, output, *, k=16, budget="params", dataset="celebdfpp",
                    resamples=10000, seed=2026):
    """Strictly pair completed factorial runs and write a separate new report."""
    paths = [Path(clean).resolve(), Path(augmented).resolve()]
    output = Path(output).resolve()
    if paths[0] == paths[1]:
        raise ValueError("Clean and augmented recipes must be distinct saved experiments")
    if any(output == path or output.is_relative_to(path) for path in paths):
        raise ValueError("Output must be outside both original experiments")
    if budget not in {"params", "time"} or k < 1 or resamples < 1:
        raise ValueError("Valid budget, positive k and resamples required")
    output.mkdir(parents=True, exist_ok=False)
    inventory = {}
    result = {"status": "INCONCLUSIVE", "analysis_status": "EXPLORATORY_RETROSPECTIVE",
              "original_artifacts_modified": False, "experiments": dict(zip(RECIPES, map(str, paths))),
              "dataset": dataset, "condition": "clean", "k": k, "budget": budget,
              "notes": ["Celeb-DF++ has already been inspected; this extension is exploratory.",
                        "Interactions concern training recipes and AUROC, not a physical causal mechanism.",
                        "No G contrast is computed when quality covariates are disabled.",
                        "Intervals do not include uncertainty over newly sampled source training datasets.",
                        "Common cache and clean-fitted coordinates are required across all eight cells."]}
    try:
        manifests = [_saved_json(path/"manifest.json", inventory) for path in paths]
        configs = [manifest["config"] for manifest in manifests]
        seeds = list(configs[0]["experiment"]["seeds"])
        if (not seeds or len(set(seeds)) != len(seeds)
                or list(configs[1]["experiment"]["seeds"]) != seeds):
            raise AuditInputError("Clean and augmented declared seed orders/sets must match exactly")
        caches = [manifest.get("cache", {}) for manifest in manifests]
        required_cache = {"features.npy", "covariates.npy", "metadata.csv"}
        if (not all(required_cache.issubset(cache) for cache in caches)
                or any(caches[0][name] != caches[1][name] for name in required_cache)):
            raise AuditInputError("Recipes must use exactly the same feature/covariate/metadata cache")
        if manifests[0].get("source") != manifests[1].get("source") or not manifests[0].get("source"):
            raise AuditInputError("Recipes must share the same training source fingerprint")
        recipe_conditions = [list(config.get("train", {}).get("conditions", ["clean"])) for config in configs]
        if recipe_conditions[0] != ["clean"] or ("clean" not in recipe_conditions[1]
                or len(recipe_conditions[1]) < 2 or len(set(recipe_conditions[1])) != len(recipe_conditions[1])):
            raise AuditInputError("Expected clean-only versus explicit clean-plus-degradation training recipes")
        source_conditions = [set(config["features"].get("source_conditions", [])) for config in configs]
        # Resolved configs can store source cache conditions under experiment.
        for index, config in enumerate(configs):
            source_conditions[index] |= set(config["experiment"].get("source_conditions", []))
        if not all(set(recipe_conditions[1]).issubset(conditions) for conditions in source_conditions):
            raise AuditInputError("Both cache recipes must declare every augmented source condition")
        train_settings = [{key: value for key, value in config["train"].items() if key != "conditions"}
                          for config in configs]
        if train_settings[0] != train_settings[1] or configs[0].get("model") != configs[1].get("model"):
            raise AuditInputError("Training/model settings differ beyond the declared condition recipe")
        target = configs[0]["train"].get("target_params")
        if budget == "params" and (not isinstance(target, int) or target < 1):
            raise AuditInputError("Parameter series requires an explicit common positive target_params")
        control = _exposure_control(configs[0]["train"], budget)
        result["training_exposure_control"] = control
        if control["status"] == "NOT_MATCHED":
            result["notes"].append(control["limitation"])
        all_tables, coordinate_reference, video_reference = [], None, None
        for recipe_index, (recipe, experiment, config) in enumerate(zip(RECIPES, paths, configs)):
            for head in HEADS:
                for cov in (0, 1):
                    common = experiment/f"k{k}_cov{cov}"/"heldout_none"
                    fit = _saved_json(common/"fit_provenance.json", inventory)
                    if (fit.get("dataset") != "ffpp" or fit.get("split") != "train"
                            or fit.get("coordinate_fit_conditions") != ["clean"]
                            or fit.get("held_out_method") is not None
                            or fit.get("head_training_conditions") != recipe_conditions[recipe_index]):
                        raise AuditInputError(f"Invalid source coordinate/training provenance: {common}")
                    if video_reference is None:
                        video_reference = fit.get("video_ids")
                    if not video_reference or fit.get("video_ids") != video_reference:
                        raise AuditInputError("Source clean training video populations differ across cells")
                    coordinates = _coordinates(common/"coordinates.npz", inventory)
                    if len(coordinates.get("zmean", [])) != k:
                        raise AuditInputError("Saved coordinate dimension differs from requested k")
                    if coordinate_reference is None:
                        coordinate_reference = coordinates
                    elif not _same_coordinates(coordinate_reference, coordinates):
                        raise AuditInputError("Clean-fitted PCA/scaler coordinates differ across factorial cells")
                    base = common/head/budget
                    actual = {int(p.name[4:]) for p in base.glob("seed*") if p.is_dir() and p.name[4:].isdigit()}
                    if actual != set(seeds):
                        raise AuditInputError(f"Missing or unexpected seed cells: {recipe}/{head}/cov{cov}")
                    hpo_done = _saved_json(base/"DONE.json", inventory)
                    hpo = _saved_json(base/"hpo.json", inventory, hpo_done)
                    if (hpo.get("selection") != "FF++ seen-method validation video AUC only"
                            or hpo.get("selection_seed") != seeds[0]):
                        raise AuditInputError("HPO must use the declared source-only validation selection")
                    for training_seed in seeds:
                        run = base/f"seed{training_seed}"
                        done = _saved_json(run/"DONE.json", inventory)
                        if "scores.csv" not in done.get("files", {}):
                            raise AuditInputError(f"scores.csv lacks a completion checksum: {run}")
                        metrics = _saved_json(run/"metrics.json", inventory, done)
                        matching, training = metrics["matching"], metrics["training"]
                        expected = {"head": head, "budget": budget, "k": k,
                                    "covariates": bool(cov), "held_out": None, "seed": training_seed}
                        if any(metrics["run"].get(key) != value for key, value in expected.items()):
                            raise AuditInputError(f"Saved run identity differs from factorial cell: {run}")
                        if (training.get("training_conditions") != recipe_conditions[recipe_index]
                                or training.get("validation_selection") != "FF++ clean seen-method video AUROC only"):
                            raise AuditInputError(f"Run training/validation recipe provenance differs: {run}")
                        if not matching.get("feasible"):
                            raise AuditInputError(f"Infeasible matched budget: {run}")
                        if budget == "params" and matching.get("target_params") != target:
                            raise AuditInputError(f"Parameter targets differ across factorial cells: {run}")
                        _validate_exposure(training, control, run)
                        if budget == "time" and (training.get("time_budget_relative_error") is None
                                or training["time_budget_relative_error"] > .10):
                            raise AuditInputError(f"Training wall-time match exceeds its 10% tolerance: {run}")
                        scores = _read(run/"scores.csv", inventory, done)
                        required = {"dataset", "split", "condition", "video_id", "label", "method",
                                    "group_id", "probability", "frames"}
                        if not required.issubset(scores):
                            raise AuditInputError(f"Missing score columns: {run}")
                        selected = scores.loc[(scores.dataset == dataset) & (scores.split == "test")
                                              & (scores.condition == "clean")].copy()
                        all_tables.append((f"{recipe}/{head}/cov{cov}/seed{training_seed}", selected))
        reference, arrays = _aligned(all_tables, ["probability"], ["label", "method", "group_id", "frames"])
        shape = (2, 2, 2, len(seeds), len(reference))
        values = arrays[:, :, 0].reshape(shape)
        if (values < 0).any() or (values > 1).any():
            raise AuditInputError("Saved probabilities must lie in [0,1]")
        analysis = factorial_auc_uncertainty(reference.label.to_numpy(), values, reference.group_id.to_numpy(),
                                             resamples=resamples, seed=seed)
        result.update(analysis, seeds=seeds, target_params=target, training_conditions=dict(zip(RECIPES, recipe_conditions)))
        if control["mode"] != "legacy_epochs":
            control["status"] = "VERIFIED"
            control["optimizer_exposure_matched"] = control["mode"] == "fixed_steps"
    except (AuditInputError, KeyError, TypeError, ValueError, OSError) as exc:
        result.update(status="INCONCLUSIVE", reason=str(exc))
    _json(output/"factorial_audit.json", result)
    _json(output/"provenance.json", {"analysis_source_sha256": _sha(Path(__file__)),
        "statistics_source_sha256": _sha(Path(__file__).with_name("statistics.py")),
        "research_audit_source_sha256": _sha(Path(__file__).with_name("research_audit.py")),
        "resamples": resamples, "resampling_seed": seed, "inputs": inventory})
    pd.DataFrame(result.get("cells", [])).to_csv(output/"cells.csv", index=False)
    rows = []
    for item in result.get("contrasts", []):
        row = {key: item[key] for key in ["contrast", "difference", "seed_difference_sd"]}
        for kind in ["conditional_clusters", "crossed_seeds_clusters"]:
            interval = item[kind]
            row[kind+"_status"] = interval["status"]
            row[kind+"_ci95_low"], row[kind+"_ci95_high"] = interval.get("ci95", [None, None])
        rows.append(row)
    pd.DataFrame(rows).to_csv(output/"contrasts.csv", index=False)
    lines = ["# Exploratory factorial FRESCO audit", "", f"Status: **{result['status']}**.", "",
        "Two heads x covariates off/on x clean/augmented source training. Endpoint: target clean video AUROC.", "",
        "All contrasts use identical video populations and paired training seeds; every bootstrap draw is shared across cells.", ""]
    if result.get("reason"):
        lines.extend([f"Input limitation: {result['reason']}", ""])
    if result.get("training_exposure_control"):
        control = result["training_exposure_control"]
        lines.extend([f"Training exposure control: **{control['status']}**; {control['criterion']}.", ""])
    for item in result.get("contrasts", []):
        lines.extend([f"- {item['contrast']}: {item['difference']:.6f}; "
            f"data-only 95% CI {item['conditional_clusters'].get('ci95')}; "
            f"crossed seed/cluster 95% CI {item['crossed_seeds_clusters'].get('ci95')}."])
    lines.extend(["", *[f"- {note}" for note in result["notes"]], "",
        "Intervals for eleven exploratory contrasts are unadjusted. Failure to show a difference is not equivalence.",
        "A change of the architecture x covariate interaction does not identify a physical causal pathway."])
    (output/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--augmented", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--budget", choices=["params", "time"], default="params")
    parser.add_argument("--dataset", default="celebdfpp")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    if args.k < 1 or args.resamples < 1:
        parser.error("--k and --resamples must be positive")
    result = audit_factorial(**vars(args))
    print(f"Factorial audit {result['status']}: {(args.output/'REPORT.md').resolve()}")
    return 0 if result["status"] == "COMPUTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
