from __future__ import annotations


import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from fresco.heads import make_head
from fresco.replay_audit import restore_coordinates
from fresco.utils import atomic_json, environment
from .inventory import sha256
from .preflight import inspect
from .provenance import snapshot_code, workspace_provenance


def digest_id(*parts):
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode("utf-8")).hexdigest()


class FrozenInputs:
    """Original data, PCA and teachers; no fitted state is changed."""
    def __init__(self, cache: Path, main: Path, replay: Path, *, verify=True):
        self.cache, self.main = Path(cache).resolve(), Path(main).resolve()
        self.preflight = inspect(self.cache, self.main, replay_provenance=replay,
                                 full_hash=verify, scan=verify)
        if self.preflight["issues"] or any(
                t["status"] != "CHECKSUM_AND_REPLAY_VERIFIED" for t in self.preflight["teachers"]):
            raise ValueError("Original inputs or checkpoint replay provenance did not pass preflight: "
                             + json.dumps(self.preflight["issues"] + self.preflight["warnings"]))
        categories = {name: "category" for name in ("dataset", "video_id", "split", "method",
                                                       "scenario", "group_id", "condition")}
        self.meta = pd.read_csv(self.cache / "metadata.csv", dtype=categories, keep_default_na=False)
        self.features = np.load(self.cache / "features.npy", mmap_mode="r")
        self.quality = np.load(self.cache / "covariates.npy", mmap_mode="r")
        if (len(self.meta) != len(self.features) or len(self.meta) != len(self.quality)
                or not np.array_equal(self.meta.row, np.arange(len(self.meta)))):
            raise ValueError("Original feature arrays and metadata do not align")
        common = self.main / "k16_cov1" / "heldout_none"
        self.coords = restore_coordinates(common / "coordinates.npz")
        if len(self.coords.zmean) != 16 or not self.coords.use_covariates:
            raise ValueError("Original frozen coordinates must be PCA16 with covariates")
        self.common = common

    def pairs(self, split="test", dataset=None):
        """Real observed clean/degraded pairs with their original array row indices."""
        data = self.meta.loc[self.meta.split == split]
        if dataset is not None:
            data = data.loc[data.dataset == dataset]
        keys = ["dataset", "video_id", "frame_idx"]
        clean = data.loc[data.condition == "clean", keys+["row"]].rename(columns={"row": "clean_row"})
        changed = data.loc[data.condition != "clean",
                           keys+["row", "label", "method", "scenario", "group_id", "condition"]].rename(
                               columns={"row": "changed_row"})
        if clean.duplicated(keys).any() or changed.duplicated(keys+["condition"]).any():
            raise ValueError("Duplicate clean/degraded frame keys")
        paired = changed.merge(clean, how="left", on=keys, validate="many_to_one", sort=False)
        if paired.clean_row.isna().any() or len(paired) != len(changed):
            raise ValueError("A transformed frame lacks its original clean anchor")
        paired["clean_row"] = paired.clean_row.astype(np.int64)
        return paired.reset_index(drop=True)

    def coordinates(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or not len(indices):
            raise ValueError("Nonempty feature row indices required")
        return self.coords.transform(self.features[indices], self.quality[indices])

    def standardized_quality(self, indices):
        raw = np.asarray(self.quality[np.asarray(indices, dtype=np.int64)], dtype=np.float32)
        return ((raw - np.asarray(self.coords.cmean, dtype=np.float32))
                / np.asarray(self.coords.cstd, dtype=np.float32)).astype(np.float32)

    def source_background(self, maximum=2048):
        rows = self.meta.loc[(self.meta.dataset == "ffpp") & (self.meta.split == "train")
                             & (self.meta.condition == "clean")]
        if len(rows) < 2:
            raise ValueError("Insufficient source training rows")
        rng = np.random.default_rng(2026)
        chosen = np.sort(rng.choice(len(rows), min(maximum, len(rows)), replace=False))
        selected = rows.iloc[chosen].reset_index(drop=True)
        return selected, self.coordinates(selected.row.to_numpy())

    def teacher(self, head, seed, device):
        path = self.common / head / "params" / f"seed{seed}" / "checkpoint.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("head") != head or checkpoint.get("in_features") != 20:
            raise ValueError("Saved head identity or input dimension mismatch")
        match = checkpoint["matching"]
        model = make_head(head, 20, int(match.get("hidden", 1)), **match.get("head_kwargs", {}))
        model.load_state_dict(checkpoint["state"], strict=True)
        return model.to(device).float().eval()


def create_run(experiment, run_id, cfg, inputs: FrozenInputs, extra=None):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run-id may contain only letters, digits, underscore and hyphen")
    root = (Path.cwd() / "outputs" / "extensions" / experiment).resolve()
    output = (root / run_id).resolve()
    if not output.is_relative_to(root) or any(output.is_relative_to(p) or p.is_relative_to(output)
                                            for p in (inputs.cache, inputs.main)):
        raise ValueError("Extension output overlaps protected original data")
    module_dir = Path(__file__).resolve().parent
    active_modules = ({"__main__.py", "runtime.py", "preflight.py", "response_run.py", "visuals.py", "paired.py",
                       "inventory.py", "figure_manifest.py", "provenance.py"} if experiment == "response" else
                      {"__main__.py", "runtime.py", "preflight.py", "representation_run.py",
                       "representation_visuals.py", "reuse.py", "inventory.py", "figure_manifest.py",
                       "source_pairs.py", "source_cache.py", "factorial_cache.py", "paired.py",
                       "historical_reference.py", "provenance.py"})
    identity = {"experiment": experiment, "config": OmegaConf.to_container(cfg, resolve=True),
                "cache_hashes": {k: v["expected_sha256"] for k, v in inputs.preflight["array_checks"].items()},
                "main_manifest_sha256": inputs.preflight["main_manifest_sha256"],
                "coordinates_sha256": inputs.preflight["coordinates_sha256"],
                "teacher_hashes": [r["checkpoint_sha256"] for r in inputs.preflight["teachers"]],
                "code": {p.name: sha256(p) for p in sorted(module_dir.glob("*.py"))
                         if not p.name.startswith("test_") and (active_modules is None or p.name in active_modules)},
                "runtime": environment(), "extra": extra or {}}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest = output / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")).get("signature") != signature:
            raise ValueError("Existing extension run has a different signature; choose a new run-id")
    else:
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(manifest, {"signature": signature, "identity": identity, "workspace": workspace_provenance(),
                               "legacy_verdict": "NOT_SUPPORTED; unchanged"})
        OmegaConf.save(cfg, output / "resolved.yaml", resolve=True)
        atomic_json(output / "preflight.json", inputs.preflight)
    snapshot_code(output/"source_snapshot", module_dir, identity["code"])
    return output, signature