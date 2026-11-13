from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from fresco.heads import make_head
from fresco.replay_audit import restore_coordinates
from fresco.utils import atomic_json, environment
from .inventory import sha256
from .paired import assert_source_splits
from .preflight import SEEDS, inspect
from .provenance import snapshot_code, workspace_provenance
from .symbolic import bounded_attempt


def _sample_source(cache, limit):
    pieces = []
    fields = ["row", "dataset", "video_id", "frame_idx", "split", "group_id", "condition"]
    for part in pd.read_csv(cache / "metadata.csv", usecols=fields, keep_default_na=False, chunksize=131072):
        selected = part.loc[(part.dataset == "ffpp") & part.split.isin(["train", "val"])
                            & (part.condition == "clean")]
        if not selected.empty:
            pieces.append(selected)
    if not pieces:
        raise ValueError("No clean source-train/validation rows")
    meta = pd.concat(pieces, ignore_index=True)
    rng = np.random.default_rng(2026)
    samples = {}
    for split in ("train", "val"):
        eligible = meta.loc[meta.split == split]
        if len(eligible) < 2:
            raise ValueError(f"Insufficient {split} source rows")
        position = np.sort(rng.choice(len(eligible), size=min(limit, len(eligible)), replace=False))
        samples[split] = eligible.iloc[position].reset_index(drop=True)
    assert_source_splits(samples["train"], samples["val"])
    features = np.load(cache / "features.npy", mmap_mode="r")
    quality = np.load(cache / "covariates.npy", mmap_mode="r")
    return samples, features, quality


def _sample_ids(meta):
    ids = [f"{r.dataset}\0{r.video_id}\0{r.frame_idx}\0{r.condition}"
           for r in meta.itertuples(index=False)]
    return sorted(hashlib.sha256(s.encode()).hexdigest() for s in ids)


def run_symbolic(cfg, cache, main, replay, output_root, run_id):
    cache, main = Path(cache).resolve(), Path(main).resolve()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run-id may contain only letters, digits, underscores and hyphens")
    output_root = Path(output_root).resolve()
    out = output_root / "explanations" / run_id / "symbolic"
    if cache.is_relative_to(out) or main.is_relative_to(out) or out.is_relative_to(cache) or out.is_relative_to(main):
        raise ValueError("Output must not overlap a historical cache or experiment")
    preflight = inspect(cache, main, replay_provenance=Path(replay) if replay else None, full_hash=True, scan=True)
    if (preflight["issues"] or any(t["status"] != "CHECKSUM_AND_REPLAY_VERIFIED"
                                    for t in preflight["teachers"])):
        raise ValueError("Original input preflight or all-frame replay binding failed: " +
                         json.dumps(preflight["issues"] + preflight["warnings"]))
    options = cfg.symbolic
    samples, features, quality = _sample_source(cache, int(options.max_rows_per_split))
    coordinates_path = main / "k16_cov1" / "heldout_none" / "coordinates.npz"
    coords = restore_coordinates(coordinates_path)
    transformed = {}
    for split, meta in samples.items():
        indices = meta.row.to_numpy(dtype=np.int64)
        transformed[split] = coords.transform(features[indices], quality[indices])
    sample_ids = {name: _sample_ids(meta) for name, meta in samples.items()}
    code_dir = Path(__file__).resolve().parent
    code_hashes = {p.name: sha256(p) for p in sorted(code_dir.glob("*.py")) if not p.name.startswith("test_")}
    runtime = environment()
    identity = {"analysis": "symbolic", "config": OmegaConf.to_container(cfg, resolve=True),
                "extension_code_hashes": code_hashes, "runtime": runtime,
                "dependency_files": {name: sha256(Path.cwd()/name) for name in ("uv.lock", "pyproject.toml")},
                "main_manifest_sha256": preflight["main_manifest_sha256"],
                "coordinates_sha256": preflight["coordinates_sha256"],
                "array_hashes": {k: v["expected_sha256"] for k, v in preflight["array_checks"].items()},
                "sample_ids_sha256": {name: hashlib.sha256("\n".join(ids).encode()).hexdigest()
                                       for name, ids in sample_ids.items()},
                "teacher_hashes": {str(t["seed"]): t["checkpoint_sha256"] for t in preflight["teachers"]
                                   if t["head"] == "kan_spline"}}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("signature") != signature:
            raise ValueError("Existing run has a different signature; use another run-id")
    else:
        atomic_json(manifest_path, {"signature": signature, "identity": identity, "environment": runtime,
                                   "workspace": workspace_provenance(),
                                   "teacher_training": False,
                                   "notice": "source-only edge fits; no whole-model symbolification evaluated"})
        OmegaConf.save(cfg, out / "resolved.yaml", resolve=True)
        atomic_json(out / "sample_ids_sha256.json", sample_ids)
        atomic_json(out / "preflight.json", preflight)
    snapshot_code(out/"source_snapshot", code_dir, identity["extension_code_hashes"])
    summaries = []
    for seed in SEEDS:
        seed_path = out / f"seed{seed}.json"
        marker_path = out / f"seed{seed}.done.json"
        if seed_path.exists() and marker_path.exists():
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if marker.get("signature") != signature or marker.get("sha256") != sha256(seed_path):
                raise ValueError(f"Saved seed {seed} failed resume verification")
            record = json.loads(seed_path.read_text(encoding="utf-8"))
        else:
            try:
                cp = main / "k16_cov1" / "heldout_none" / "kan_spline" / "params" / f"seed{seed}" / "checkpoint.pt"
                checkpoint = torch.load(cp, map_location="cpu", weights_only=True)
                matching = checkpoint["matching"]
                model = make_head("kan_spline", checkpoint["in_features"], int(matching["hidden"]),
                                  **matching["head_kwargs"])
                model.load_state_dict(checkpoint["state"], strict=True)
                record = bounded_attempt(model, transformed["train"], transformed["val"],
                                         train_meta=samples["train"], val_meta=samples["val"],
                                         tolerance=float(options.edge_validation_rmse_tolerance),
                                         max_rows=int(options.max_rows_per_split), vocabulary=tuple(options.vocabulary))
            except Exception as exc:  # noqa: BLE001 - record every failed seed and continue the ten-seed audit.
                record = {"status": "FAILED", "reason": f"{type(exc).__name__}: {exc}"}
            record["seed"] = seed
            atomic_json(seed_path, record)
            atomic_json(marker_path, {"signature": signature, "sha256": sha256(seed_path)})
        summaries.append({k: record.get(k) for k in ("seed", "status", "reason", "accepted_edges", "total_edges",
                                                      "pruned_edges", "pruned_clone_validation_rmse",
                                                      "elapsed_seconds")})
    failed = sum(s["status"] == "FAILED" for s in summaries)
    overall = "BLOCKED" if failed == len(SEEDS) else ("PARTIAL_FAILURE" if failed else "FULL_SYMBOLIFICATION_NOT_EVALUATED")
    atomic_json(out / "metrics.json", {"status": overall, "seeds": summaries,
                                      "success_denominator": 10, "timestamp_utc": time.time()})
    pd.DataFrame(summaries).to_csv(out / "metrics.csv", index=False)
    (out / "REPORT.md").write_text(
        "# Source-only KAN edge attempt\n\n"
        f"Attempted seeds: {len(summaries)}/10; failed: {failed}. No original weights were modified.\n\n"
        "Actual spline edge values were approximated by a bounded vocabulary on source train; "
        "candidate selection and zero-edge pruning used source validation only. "
        "No whole-model symbolic conversion or held-out detector evaluation was performed: "
        "FULL_SYMBOLIFICATION_NOT_EVALUATED. Do not interpret edge fits as whole-model fidelity.\n\n"
        "## Limitations\n\nOnly source validation of a pruned clone is measured; no target result, "
        "complexity-fidelity frontier, publication figures, or KAN-vs-MLP structural claim is available.\n",
        encoding="utf-8")
    return out