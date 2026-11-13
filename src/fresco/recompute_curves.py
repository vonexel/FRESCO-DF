"""Recompute curve-pair eligibility from frozen curves and saved frame logits.

No encoder, head, PCA or response curve is fitted or evaluated. This correction
uses only original metadata, quality covariates and verified recorded logits.
The previous audit and original experiment remain immutable inputs.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .response_audit import (COVARIATE_NAMES, CURVE_COVERAGE_RULE, CURVE_COVERAGE_VERSION,
                             CURVE_POPULATION_WARNING, _curve_pairs, summarize_curve_response)
from .utils import atomic_json, digest_file


def saved_logits(run_path, meta, *, expected_hash):
    """Read only row/logit columns after verifying the original completion record."""
    path = Path(run_path) / "frame_scores.csv"
    completion = json.loads((Path(run_path) / "DONE.json").read_text(encoding="utf-8"))
    if (not expected_hash or completion.get("files", {}).get(path.name) != expected_hash
            or digest_file(path) != expected_hash):
        raise ValueError(f"Saved test logits lack matching verified provenance: {path}")
    mask = (meta.split == "test").to_numpy()
    expected = meta.loc[mask, "row"].to_numpy()
    logits = np.full(len(meta), np.nan, dtype=np.float64)
    offset = 0
    for chunk in pd.read_csv(path, usecols=["row", "logit"], chunksize=131072):
        rows = chunk.row.to_numpy()
        stop = offset + len(rows)
        if not np.array_equal(rows, expected[offset:stop]):
            raise ValueError("Original frame-score ordering/population differs from the verified cache")
        values = chunk.logit.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Original frame logits must be finite")
        logits[rows] = values
        offset = stop
    if offset != len(expected):
        raise ValueError("Original frame scores are incomplete")
    return logits


def population_summaries(table, unmatched, epsilon):
    result = {**summarize_curve_response(table, epsilon), "unmatched_changed_frames": unmatched}
    for name, keys in [("by_dataset", ["dataset"]),
                       ("by_dataset_condition", ["dataset", "condition"]),
                       ("by_dataset_method", ["dataset", "method"]),
                       ("by_dataset_class", ["dataset", "label"])]:
        rows = []
        for key, group in table.groupby(keys, sort=True):
            values = key if isinstance(key, tuple) else (key,)
            rows.append({**dict(zip(keys, values)), **summarize_curve_response(group, epsilon)})
        result[name] = rows
    return result


def recompute_curves(previous, experiment, cache, output):
    previous, experiment, cache, output = (Path(p).resolve() for p in [previous, experiment, cache, output])
    if output.exists() or any(output.is_relative_to(p) or p.is_relative_to(output)
                              for p in [previous, experiment, cache]):
        raise ValueError("Use a new output directory outside all original inputs")
    old_provenance = json.loads((previous / "provenance.json").read_text(encoding="utf-8"))
    manifest_path = experiment / "manifest.json"
    if digest_file(manifest_path) != old_provenance["original_manifest_sha256"]:
        raise ValueError("Original experiment manifest changed since the verified replay")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache_manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    inventory = {str(manifest_path): digest_file(manifest_path)}
    for name in ["metadata.csv", "covariates.npy"]:
        path = cache / name
        expected = manifest["cache"][name]
        if (cache_manifest.get(name.split(".")[0]+"_sha256") != expected
                or digest_file(path) != expected):
            raise ValueError(f"The cache does not match the original experiment: {path}")
        inventory[str(path)] = expected
    meta = pd.read_csv(cache / "metadata.csv", keep_default_na=False)
    if not np.array_equal(meta.row, np.arange(len(meta))):
        raise ValueError("Original metadata row identifiers must match array positions")
    if ((meta.dataset != "ffpp") & (meta.split != "test")).any():
        raise ValueError("Target rows must remain test-only")
    if meta.duplicated(["dataset", "video_id", "frame_idx", "condition"]).any():
        raise ValueError("Duplicate frame-condition metadata")
    quality = np.load(cache / "covariates.npy", allow_pickle=False)
    if quality.shape != (len(meta), 4) or not np.isfinite(quality).all():
        raise ValueError("Finite [metadata rows,4] quality covariates required")
    prior = json.loads((previous / "audit.json").read_text(encoding="utf-8"))
    inventory[str(previous / "audit.json")] = digest_file(previous / "audit.json")
    proofs = {(r["run"]["head"], r["run"]["seed"]): r for r in old_provenance["runs"]}
    identities = [(r["run"]["head"], r["run"]["seed"]) for r in prior["runs"]]
    if len(identities) != len(set(identities)) or set(identities) != set(proofs):
        raise ValueError("Previous measured runs and verified provenance disagree")
    cfg = manifest["config"]
    epsilon = float(cfg.get("eval", {}).get("response_zero_epsilon", 1e-8))
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Positive finite response epsilon required")
    output.mkdir(parents=True)
    provenance = {**copy.deepcopy(old_provenance), "previous_audit": str(previous),
        "coverage_rule_version": CURVE_COVERAGE_VERSION, "coverage_rule": CURVE_COVERAGE_RULE,
        "correction": "unchanged finite coordinates cancel exactly without out-of-grid evaluation",
        "no_model_inference": True, "no_curve_refit": True,
        "features_array_used": False, "inputs_sha256": inventory,
        "analysis_source_sha256": digest_file(Path(__file__)),
        "response_audit_source_sha256": digest_file(Path(__file__).with_name("response_audit.py")),
        "runs": []}
    records, coordinates = [], {}
    for record in prior["runs"]:
        identity = record["run"]
        head, seed = identity["head"], identity["seed"]
        print(f"CORRECT COVERAGE {head} seed{seed}", flush=True)
        key = (identity["k"], identity["covariates"], identity["held_out"])
        if not key[1]:
            raise ValueError("Saved-curve correction requires enabled covariates")
        common = experiment / f"k{key[0]}_cov1" / f"heldout_{key[2] or 'none'}"
        coordinate_path = common / "coordinates.npz"
        if digest_file(coordinate_path) != old_provenance["coordinates_sha256"]:
            raise ValueError("Coordinates changed since the verified replay")
        if key not in coordinates:
            with np.load(coordinate_path, allow_pickle=False) as fitted:
                standardized = ((quality-fitted["cmean"])/fitted["cstd"]).astype(np.float32)
            coordinates[key] = standardized
            inventory[str(coordinate_path)] = digest_file(coordinate_path)
        source = previous / head / f"seed{seed}"
        predictor_path = source / "curve_predictor.csv"
        predictor = pd.read_csv(predictor_path)
        inventory[str(predictor_path)] = digest_file(predictor_path)
        grids, responses = [], []
        if set(predictor.feature) != set(COVARIATE_NAMES):
            raise ValueError("Frozen predictor must contain all four expected quality curves")
        for j, name in enumerate(COVARIATE_NAMES):
            curve = predictor.loc[predictor.feature == name].sort_values("grid")
            if not curve.feature_index.eq(key[0]+j).all():
                raise ValueError("Frozen predictor feature identity differs from saved coordinates")
            grids.append(curve.grid.to_numpy(float))
            responses.append(curve.response.to_numpy(float))
        run = common / head / identity["budget"] / f"seed{seed}"
        proof = proofs[(head, seed)]
        if digest_file(run / "checkpoint.pt") != proof["checkpoint_sha256"]:
            raise ValueError("Original checkpoint changed since the verified replay")
        frame_hash = proof["replay_validation"].get("frame_scores_sha256")
        if frame_hash is None:
            frame_hash = json.loads((run / "DONE.json").read_text())["files"]["frame_scores.csv"]
        logits = saved_logits(run, meta, expected_hash=frame_hash)
        inventory[str(run / "frame_scores.csv")] = frame_hash
        # Only four covariates are needed: paired full-head responses are already
        # measured in logits. No synthetic embedding values are introduced.
        table, unmatched = _curve_pairs(None, coordinates[key], meta, 0, grids, responses,
                                        "cpu", 16384, epsilon, logits=logits)
        audit = copy.deepcopy(record["audit"])
        audit["curve_response"] = population_summaries(table, unmatched, epsilon)
        audit["curve_predictor"].update(coverage_rule_version=CURVE_COVERAGE_VERSION,
                                       coverage_rule=CURVE_COVERAGE_RULE)
        destination = output / head / f"seed{seed}"
        destination.mkdir(parents=True)
        table.to_csv(destination / "curve_intervention.csv", index=False)
        for name in ["curve_predictor.csv", "response_support.csv", "spline_support.csv"]:
            shutil.copy2(source / name, destination / name)
        atomic_json(destination / "response_audit.json", {"run": identity, "audit": audit})
        records.append({"run": identity, "audit": audit})
        provenance["runs"].append({**proof, "frozen_predictor_sha256": inventory[str(predictor_path)]})
        atomic_json(output / "provenance.json", provenance)
    result = {"status": "EXPLORATORY_FOLLOWUP", "coverage_rule_version": CURVE_COVERAGE_VERSION,
              "previous_audit": str(previous), "runs": records}
    atomic_json(output / "audit.json", result)
    rows = []
    for record in records:
        rows += [{"head": record["run"]["head"], "seed": record["run"]["seed"], **row}
                 for row in record["audit"]["curve_response"]["by_dataset"]]
    pd.DataFrame(rows).to_csv(output / "by_seed.csv", index=False)
    lines = ["# Corrected frozen-curve audit", "", f"Coverage rule version: {CURVE_COVERAGE_VERSION}.",
             "Unchanged finite coordinates contribute exactly zero without interpolation; only changing coordinates require both endpoints in their source grid.",
             "The previous all-coordinate eligibility rule was unnecessarily strict. Previous measurements are preserved as superseded inputs.",
             "", "No encoder/head inference, retraining, PCA or curve fitting was performed.",
             "All logits are verified original observations; curves and covariates remain unchanged.",
             "", CURVE_POPULATION_WARNING,
             "Coverage depends on the declared marginal curve grid, not joint support or an intrinsic model domain.",
             "A zero-change predictor has G_curve=1 on this same covered population when its denominator is defined.",
             "The original main G_block hypothesis assessment is unaffected.", "",
             "See by_seed.csv, audit.json and the separate curve_comparison report for paired intervals.", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["previous", "experiment", "cache", "output"]:
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    output = recompute_curves(**vars(args))
    print(f"REPORT: {output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
