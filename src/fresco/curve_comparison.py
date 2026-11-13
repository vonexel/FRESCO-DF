"""Exploratory, coverage-matched comparison of frozen response-curve predictors.

The reference predicts zero logit change and therefore has G_curve = 1 whenever
the observed response has nonzero energy. All comparisons use the same covered
frame-condition pairs; no comparison with full-population G_block is made.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .research_audit import AuditInputError, _json, _sha, paired_gap_uncertainty


HEADS = ("kan_spline", "mlp_gelu")
KEYS = ["dataset", "video_id", "condition"]
COUNTS = ["paired_frames", "covered_frames"]


def _read_table(path, inventory):
    if not path.is_file():
        raise AuditInputError(f"Missing curve response artifact: {path}")
    table = pd.read_csv(path, keep_default_na=False)
    inventory[str(path)] = {"sha256": _sha(path), "bytes": path.stat().st_size}
    required = {*KEYS, "group_id", *COUNTS, "error_sse", "total_sse"}
    if not required.issubset(table.columns):
        raise AuditInputError(f"Missing curve columns: {sorted(required-set(table.columns))}")
    if table.empty or table.duplicated(KEYS).any():
        raise AuditInputError("Curve populations must be nonempty with unique video-condition keys")
    for name in [*COUNTS, "error_sse", "total_sse"]:
        values = pd.to_numeric(table[name], errors="raise").to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise AuditInputError("Curve counts and sums of squares must be finite and nonnegative")
        table[name] = values
    for name in COUNTS:
        if not np.equal(table[name], np.floor(table[name])).all():
            raise AuditInputError("Curve frame counts must be integers")
        table[name] = table[name].astype(np.int64)
    if (table.paired_frames <= 0).any() or (table.covered_frames > table.paired_frames).any():
        raise AuditInputError("Covered counts must be bounded by positive paired counts")
    absent = table.covered_frames == 0
    if (table.loc[absent, ["error_sse", "total_sse"]] != 0).any().any():
        raise AuditInputError("Uncovered observations must not contribute error or response energy")
    if table.group_id.eq("").any():
        raise AuditInputError("Every video needs a source/identity cluster")
    if table.groupby(["dataset", "video_id"]).group_id.nunique().gt(1).any():
        raise AuditInputError("A video has inconsistent source/identity clusters")
    table = table.sort_values(KEYS).reset_index(drop=True)
    identity = [*KEYS, "group_id", *COUNTS]
    identity += [name for name in ["label", "method"] if name in table]
    return table, table[identity]


def _video_vectors(table):
    videos = table.groupby(["video_id", "group_id"], as_index=False)[[
        "error_sse", "total_sse", "covered_frames"]].sum().sort_values("video_id")
    videos = videos.loc[videos.covered_frames > 0].reset_index(drop=True)
    return videos, (videos.error_sse / videos.covered_frames).to_numpy(), (
        videos.total_sse / videos.covered_frames).to_numpy()


def _contrast(error_a, total_a, error_b, total_b, groups, resamples, seed, label):
    result = paired_gap_uncertainty(error_a, total_a, error_b, total_b, groups,
                                    resamples=resamples, seed=seed)
    result["direction"] = f"{label}; negative favors the first predictor; lower G_curve is better"
    result["endpoint"] = "mean seed-specific equal-video G_curve on common covered pairs"
    result["analysis_role"] = "EXPLORATORY_COVERED_SUBSET"
    result["interval_method"] = "approximate percentile bootstrap of paired seeds and test clusters"
    result["minimum_seed_count_met"] = result.pop(
        "confirmatory_seed_minimum_met", result.get("minimum_seed_count_met", False))
    result.setdefault("difference", None)
    return result


def compare_curves(directory, output, *, resamples=10000, seed=2026):
    """Write new comparison artifacts; preserve input replay and original runs.

    Missing observations or unequal coverage are errors rather than grounds for
    silently intersecting the available populations. Only videos with zero
    common curve coverage are excluded from the conditional gap estimand.
    """
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if output == directory or output.is_relative_to(directory):
        raise ValueError("Use a new comparison directory outside the input curve audit")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    output.mkdir(parents=True, exist_ok=False)
    inventory = {}
    result = {"status": "INCONCLUSIVE", "analysis_role": "EXPLORATORY_COVERED_SUBSET",
              "input_curve_audit": str(directory), "original_artifacts_modified": False,
              "zero_change_reference_gap": 1., "datasets": [], "notes": [
                  "Changing coordinates require both endpoints inside frozen source grids; unchanged finite coordinates contribute exactly zero.",
                  "Estimates condition on the common covered pairs and videos; this subset need not represent the full test.",
                  "G_curve is not compared numerically with full-population G_block.",
                  "Each head is evaluated against its own observed logit change; response normalization differs between heads.",
                  "The zero-change predictor has G=1 for each valid seed; mean G>1 concerns mean normalized gaps, not pooled unnormalized MSE across seeds.",
                  "Intervals are approximate and exploratory after inspecting Celeb-DF++; they are conditional on the source training data and current generator mix.",
                  "Three contrasts per dataset are exploratory with unadjusted intervals; they do not establish universal architecture superiority.",
                  "The original G_block threshold 0.30 is not a success rule for G_curve."]}
    csv_rows = []
    try:
        objects = {}
        for name in ["audit.json", "provenance.json"]:
            path = directory/name
            objects[name] = json.loads(path.read_text(encoding="utf-8"))
            inventory[str(path)] = {"sha256": _sha(path), "bytes": path.stat().st_size}
        audit, provenance = objects["audit.json"], objects["provenance.json"]
        declared = provenance.get("declared_seeds")
        selected = provenance.get("selected_seeds")
        if (not isinstance(declared, list) or not declared or len(declared) != len(set(declared))
                or selected != declared or not provenance.get("complete_declared_seed_coverage")):
            raise AuditInputError("All unique declared training seeds must be present in the curve audit")
        expected = {(head, training_seed) for head in HEADS for training_seed in declared}
        actual = [(row["run"]["head"], row["run"]["seed"]) for row in audit["runs"]]
        if len(actual) != len(expected) or set(actual) != expected:
            raise AuditInputError("Curve audit must contain exactly both heads and all declared seeds")
        tables, reference = {}, None
        for head in HEADS:
            for training_seed in declared:
                table, identity = _read_table(directory/head/f"seed{training_seed}"/"curve_intervention.csv", inventory)
                if reference is None:
                    reference = identity
                elif not identity.equals(reference):
                    raise AuditInputError("Paired frame-condition populations or curve coverage differ across heads/seeds")
                tables[(head, training_seed)] = table
        result["seeds"] = declared
        for dataset, identity in reference.groupby("dataset", sort=True):
            paired = int(identity.paired_frames.sum())
            covered = int(identity.covered_frames.sum())
            population = {"dataset": dataset, "paired_frame_conditions": paired,
                          "covered_frame_conditions": covered, "coverage": covered/paired,
                          "videos": int(identity.video_id.nunique()),
                          "covered_videos": int(identity.loc[identity.covered_frames > 0, "video_id"].nunique()),
                          "coverage_complete": covered == paired,
                          "scope": "common covered pairs only; videos with zero coverage excluded"}
            if not covered:
                result["datasets"].append({**population, "status": "UNAVAILABLE", "reason": "no common covered pairs"})
                continue
            errors, totals, base_videos = {}, {}, None
            for head in HEADS:
                es, ts = [], []
                for training_seed in declared:
                    table = tables[(head, training_seed)]
                    video, e, t = _video_vectors(table.loc[table.dataset == dataset])
                    if base_videos is None:
                        base_videos = video[["video_id", "group_id", "covered_frames"]]
                    elif not video[["video_id", "group_id", "covered_frames"]].equals(base_videos):
                        raise AuditInputError("Covered video populations differ after aggregation")
                    es.append(e)
                    ts.append(t)
                errors[head], totals[head] = np.stack(es), np.stack(ts)
            groups = base_videos.group_id.to_numpy()
            comparisons = {"kan_minus_mlp": _contrast(errors[HEADS[0]], totals[HEADS[0]],
                errors[HEADS[1]], totals[HEADS[1]], groups, resamples, seed, "KAN G_curve minus MLP G_curve")}
            for head in HEADS:
                comparisons[head+"_minus_zero_change"] = _contrast(errors[head], totals[head],
                    totals[head], totals[head], groups, resamples, seed, head+" G_curve minus 1")
            measured = {**population, "status": "COMPUTED", "clusters": len(np.unique(groups)),
                        "comparisons": comparisons, "heads": {}}
            for head in HEADS:
                measured["heads"][head] = {"mean_absolute_rmse": float(np.sqrt(errors[head].mean(1)).mean()),
                    "seed_absolute_rmse": np.sqrt(errors[head].mean(1)).tolist(),
                    "zero_change_rmse_mean": float(np.sqrt(totals[head].mean(1)).mean())}
            for name, contrast in comparisons.items():
                if contrast["status"] != "COMPUTED":
                    measured["status"] = "INCONCLUSIVE"
                row = {"dataset": dataset, "comparison": name, "difference": contrast["difference"],
                       "first_gap_mean": contrast["head_a_mean"], "reference_gap_mean": contrast["head_b_mean"],
                       "coverage": population["coverage"], "covered_videos": population["covered_videos"]}
                for kind in ["conditional_clusters", "crossed_seeds_clusters"]:
                    interval = contrast.get(kind, {})
                    row[kind+"_status"] = interval.get("status", "UNAVAILABLE")
                    row[kind+"_low"], row[kind+"_high"] = interval.get("ci95", [None, None])
                csv_rows.append(row)
            result["datasets"].append(measured)
        result["status"] = ("COMPUTED" if result["datasets"] and all(
            row["status"] == "COMPUTED" for row in result["datasets"]) else "INCONCLUSIVE")
    except (AuditInputError, KeyError, TypeError, ValueError, OSError) as exc:
        result.update(status="INCONCLUSIVE", reason=str(exc))
    _json(output/"comparison.json", result)
    _json(output/"provenance.json", {"analysis_source_sha256": _sha(Path(__file__)),
          "research_audit_source_sha256": _sha(Path(__file__).with_name("research_audit.py")),
          "resamples": resamples, "resampling_seed": seed, "inputs": inventory})
    pd.DataFrame(csv_rows).to_csv(output/"comparisons.csv", index=False)
    lines = ["# Coverage-matched curve predictor comparison", "", f"Status: **{result['status']}**.", "",
             "Exploratory approximate percentile intervals; lower G_curve is better.", ""]
    if result.get("reason"):
        lines += ["Input limitation: "+result["reason"], ""]
    for population in result["datasets"]:
        lines += [f"## {population['dataset']}", "",
                  f"Coverage: {population['covered_frame_conditions']}/{population['paired_frame_conditions']} "
                  f"({population['coverage']:.4%}); covered videos: {population['covered_videos']}/{population['videos']}.", ""]
        if "comparisons" not in population:
            lines += [population.get("reason", "Unavailable"), ""]
            continue
        lines += ["| Contrast | Difference | 95% fixed-seed cluster CI | 95% seed x cluster CI |",
                  "|---|---:|---|---|"]
        for name, contrast in population["comparisons"].items():
            difference = contrast["difference"]
            text = f"{difference:.6f}" if difference is not None and np.isfinite(difference) else "unavailable"
            lines.append(f"|{name}|{text}|"
                         f"{contrast.get('conditional_clusters', {}).get('ci95')}|"
                         f"{contrast.get('crossed_seeds_clusters', {}).get('ci95')}|")
        lines += ["", "The zero-change reference has G_curve=1. Values above one are worse than predicting no change "
                  "on this covered subset; uncertainty of the difference must be read alongside the point estimate.", ""]
    lines += ["## Limits", "", *["- "+note for note in result["notes"]], ""]
    (output/"REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    if args.resamples < 1:
        parser.error("--resamples must be positive")
    result = compare_curves(args.curves, args.output, resamples=args.resamples, seed=args.seed)
    print(f"Curve comparison {result['status']}: {(args.output/'REPORT.md').resolve()}")
    return 0 if result["status"] == "COMPUTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
