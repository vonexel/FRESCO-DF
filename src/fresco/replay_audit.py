"""Evaluate new response diagnostics on saved heads without retraining or rewriting runs.

The output directory must be new and outside the original experiment/cache.
The supplied cache must match the original recorded artifact hashes. No source
coordinates, temperatures, heads or hyperparameters are fitted in this command.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from .heads import make_head
from .response_audit import audit_responses
from .run import load_cache
from .training import Coordinates, predict
from .utils import atomic_json, digest_file, environment


def restore_coordinates(path: Path) -> Coordinates:
    coords = Coordinates()
    with np.load(path, allow_pickle=False) as saved:
        for name in saved.files:
            value = saved[name]
            setattr(coords, name, value.item() if value.ndim == 0 else value)
    required = {"center", "components", "zmean", "zstd", "cmean", "cstd", "clow", "chigh",
                "use_covariates"}
    if not required.issubset(vars(coords)):
        raise ValueError("Saved coordinates are incomplete; do not refit them on test data")
    # Pre-native-reference checkpoints used PCA for all dimensions.
    if not hasattr(coords, "native"):
        coords.native = False
    return coords


def select_runs(experiment: Path, heads, budget, k, covariates, held_out, seeds):
    common = experiment / f"k{k}_cov{int(covariates)}" / f"heldout_{held_out or 'none'}"
    selected = []
    for head in heads:
        for seed in seeds:
            path = common / head / budget / f"seed{seed}"
            metrics_path, checkpoint_path = path / "metrics.json", path / "checkpoint.pt"
            if not all(p.is_file() for p in [metrics_path, checkpoint_path, path / "DONE.json"]):
                raise ValueError(f"Missing completed saved run: {path}")
            details = json.loads(metrics_path.read_text(encoding="utf-8"))
            run = details["run"]
            wanted = {"head": head, "budget": budget, "k": k, "covariates": covariates,
                      "held_out": held_out, "seed": seed}
            if any(run.get(key) != value for key, value in wanted.items()):
                raise ValueError(f"Saved run identity mismatch: {path}")
            marker = json.loads((path / "DONE.json").read_text(encoding="utf-8"))
            recorded = marker.get("files", {})
            for name in ["metrics.json", "checkpoint.pt"]:
                if recorded.get(name) != digest_file(path / name):
                    raise ValueError(f"Saved run checksum mismatch: {path / name}")
            selected.append((path, run))
    return common, selected


def verify_replayed_logits(run_path: Path, meta, logits, *, rtol=1e-4, atol=2e-5):
    """Legacy coordinates are accepted only after reproducing all saved test logits.

    Old checkpoints do not bind their PCA/scaler checksum. A matching shape is
    insufficient, so verify the recorded frame artifact and every paired frame.
    """
    columns = ["dataset", "video_id", "frame_idx", "condition"]
    path = run_path / "frame_scores.csv"
    marker = json.loads((run_path / "DONE.json").read_text(encoding="utf-8"))
    expected_hash = marker.get("files", {}).get(path.name)
    if not expected_hash or digest_file(path) != expected_hash:
        raise ValueError("Legacy replay requires verified saved frame_scores.csv")
    mask = (meta.split == "test").to_numpy()
    expected_keys = meta.loc[mask, columns].reset_index(drop=True)
    predicted = np.asarray(logits)[mask]
    offset, maximum = 0, 0.
    for chunk in pd.read_csv(path, usecols=columns+["logit"], keep_default_na=False, chunksize=131072):
        end = offset+len(chunk)
        keys = chunk[columns].reset_index(drop=True)
        if end > len(expected_keys) or not keys.equals(expected_keys.iloc[offset:end].reset_index(drop=True)):
            raise ValueError("Saved and replayed frame keys/order differ")
        recorded = chunk.logit.to_numpy(dtype=float)
        current = predicted[offset:end]
        if not np.isfinite(recorded).all() or not np.allclose(current, recorded, rtol=rtol, atol=atol):
            raise ValueError("Saved coordinates/head do not reproduce the original test logits")
        maximum = max(maximum, float(np.max(np.abs(current-recorded))))
        offset = end
    if offset != len(expected_keys):
        raise ValueError("Saved frame artifact has incomplete test coverage")
    return {"evidence": "all saved test-frame logits reproduced", "frames": offset,
            "maximum_absolute_error": maximum, "rtol": rtol, "atol": atol,
            "frame_scores_sha256": expected_hash}


def replay(experiment: Path, cache: Path, output: Path, *, heads=("kan_spline", "mlp_gelu"),
           budget="params", k=16, covariates=True, held_out=None, seeds=None, device="cpu"):
    experiment, cache, output = (Path(p).resolve() for p in [experiment, cache, output])
    if output.exists() or any(output.is_relative_to(p) or p.is_relative_to(output)
                              for p in [experiment, cache]):
        raise ValueError("Use a new output directory outside the saved experiment and cache")
    manifest_path = experiment / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cfg = OmegaConf.create(manifest["config"])
    declared = list(cfg.experiment.seeds)
    seeds = declared if seeds is None else list(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds).issubset(declared):
        raise ValueError("Select a nonempty unique subset of the original declared seeds")
    if not heads or len(set(heads)) != len(heads):
        raise ValueError("Select unique saved heads")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; select --device cpu explicitly")
    common, selected = select_runs(experiment, heads, budget, k, covariates, held_out, seeds)
    cache_manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    for artifact in ["features.npy", "covariates.npy", "metadata.csv"]:
        key = artifact.split(".")[0] + "_sha256"
        expected = manifest.get("cache", {}).get(artifact)
        if not expected or cache_manifest.get(key) != expected:
            raise ValueError(f"Cache does not match the original experiment: {artifact}")
    features, quality, meta = load_cache(cache)
    if cfg.experiment.get("conditions"):
        mask = meta.condition.isin(list(cfg.experiment.conditions)).to_numpy()
        features, quality, meta = features[mask], quality[mask], meta.loc[mask].reset_index(drop=True)
    coords_path = common / "coordinates.npz"
    coords = restore_coordinates(coords_path)
    if len(coords.zmean) != k or bool(coords.use_covariates) != covariates:
        raise ValueError("Saved coordinate dimension/covariates disagree with requested runs")
    X = coords.transform(features, quality)
    cfg.train.device = device
    torch.set_num_threads(int(cfg.train.get("threads", 4)))
    output.mkdir(parents=True)
    provenance = {"status": "EXPLORATORY_FOLLOWUP", "original_experiment": str(experiment),
                  "original_manifest_sha256": digest_file(manifest_path),
                  "coordinates_sha256": digest_file(coords_path), "cache": manifest["cache"],
                  "selected_seeds": seeds, "declared_seeds": declared,
                  "complete_declared_seed_coverage": set(seeds) == set(declared),
                  "source_only_curve_fit": True, "retrained": False,
                  "original_outputs_modified": False, "environment": environment(), "runs": []}
    atomic_json(output / "provenance.json", provenance)
    records = []
    for run_path, run in selected:
        print(f"REPLAY AUDIT {run_path}", flush=True)
        checkpoint = torch.load(run_path / "checkpoint.pt", map_location="cpu", weights_only=True)
        if checkpoint.get("head") != run["head"] or checkpoint.get("in_features") != X.shape[1]:
            raise ValueError(f"Checkpoint head/input mismatch: {run_path}")
        matching = checkpoint["matching"]
        coordinate_digest = checkpoint.get("coordinates_sha256")
        if coordinate_digest is not None and coordinate_digest != digest_file(coords_path):
            raise ValueError("Coordinates do not match the checksum bound to the checkpoint")
        model = make_head(run["head"], X.shape[1], int(matching.get("hidden", 1)),
                          **matching.get("head_kwargs", {}))
        model.load_state_dict(checkpoint["state"], strict=True)
        model = model.to(device).float().eval()
        destination = output / run["head"] / f"seed{run['seed']}"
        destination.mkdir(parents=True)
        logits = predict(model, X, device, batch=int(cfg.eval.get("audit_batch_size", 4096)))
        validation = ({"evidence": "coordinate checksum bound to saved checkpoint"}
                      if coordinate_digest is not None else verify_replayed_logits(run_path, meta, logits))
        result = audit_responses(model, X, meta, coords, cfg, destination, held_out, logits=logits)
        atomic_json(destination / "response_audit.json", {"run": run, "audit": result})
        records.append({"run": run, "audit": result})
        provenance["runs"].append({"run": run, "checkpoint_sha256": digest_file(run_path / "checkpoint.pt"),
                                  "replay_validation": validation})
        atomic_json(output / "provenance.json", provenance)
        del model
    atomic_json(output / "audit.json", {"status": "EXPLORATORY_FOLLOWUP", "runs": records})
    lines = ["# Frozen curve prediction audit", "", "Exploratory follow-up after inspecting the target benchmark.",
             "No heads, source coordinates or hyperparameters were refitted. Original results are preserved.",
             "", "G_curve uses source-validation PD curves; G_block uses the original hybrid full-head response.",
             "Out-of-grid curve predictions are unavailable; report coverage alongside covered-only errors.",
             "Unchanged finite covariates cancel exactly without interpolation, even outside a grid.",
             "Covered-only G_curve and full-test G_block use different populations and must not be subtracted.",
             "Quantile-box excursions and actual spline-layer clipping are separate diagnostics.", "",
             f"Completed saved-head evaluations: {len(records)}; seeds: {seeds}.",
             f"Complete original seed coverage: {set(seeds) == set(declared)}.", "",
             "See audit.json and each head/seed's CSV files for measured coverage and errors.", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
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
    parser.add_argument("--held-out", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = replay(args.experiment, args.cache, args.output, heads=args.heads, seeds=args.seeds,
                    budget=args.budget, k=args.k, covariates=not args.without_covariates,
                    held_out=args.held_out, device=args.device)
    print(f"REPORT: {result / 'REPORT.md'}")


if __name__ == "__main__":
    main()
