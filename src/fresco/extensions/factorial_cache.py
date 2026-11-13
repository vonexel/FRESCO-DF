from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from fresco.heads import make_head
from fresco.replay_audit import restore_coordinates
from fresco.training import predict
from .inventory import sha256
from .paired import assert_source_splits
from .source_cache import CONDITIONS


_CHECKPOINT = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
_FIELDS = ("faces_sha256", "crop_content_sha256", "encoder", "pretrained", "feature", "covariates",
           "conditions", "dtype", "activation_policy", "torch_version", "open_clip_torch_version")
_KEYS = ["dataset", "video_id", "frame_idx"]
_ATTRS = ["split", "label", "method", "scenario", "group_id"]
_TEST_FIELDS = [*_KEYS, "condition", *_ATTRS]


def _provenance(original, historical):
    for field in _FIELDS:
        left, right = original.get(field), historical.get(field)
        if left is None or right is None or left != right:
            raise ValueError(f"Factorial cache provenance mismatch or missing: {field}")
    if original.get("encoder") != "ViT-L-14" or original.get("pretrained") != "openai":
        raise ValueError("Factorial cache must use the original OpenAI ViT-L-14 encoder")
    for name, manifest in (("original", original), ("factorial", historical)):
        p = manifest.get("encoder_provenance", {})
        if (p.get("checkpoint_sha256") != _CHECKPOINT or not p.get("projection_consistency", {}).get("passed")
                or p.get("visual_trainable_parameter_count") != 0
                or p.get("whole_clip_trainable_parameter_count") != 0):
            raise ValueError(f"{name} cache lacks verified frozen CLIP encoder provenance")
    if historical.get("source_only") is True or set(historical.get("source_conditions", [])) != {"clean", *CONDITIONS}:
        raise ValueError("Factorial cache must be the complete historical clean/augmented cache")


def _test_population(original, historical):
    a = original.loc[original.split == "test", _TEST_FIELDS].reset_index(drop=True)
    b = historical.loc[historical.split == "test", _TEST_FIELDS].reset_index(drop=True)
    if len(a) != len(b):
        raise ValueError("Factorial and original held-out frame populations differ in size")
    for start in range(0, len(a), 65536):
        x, y = a.iloc[start:start + 65536], b.iloc[start:start + 65536]
        for field in _TEST_FIELDS:
            if not np.array_equal(x[field].to_numpy(), y[field].to_numpy()):
                raise ValueError(f"Factorial and original held-out metadata differ: {field}")
    return len(a)


def _replay_factorial_samples(meta, features, quality, reuse, device):
    """Reproduce saved test-frame logits for one seed in each old factorial cell."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable for historical factorial replay")
    columns = ["row", "dataset", "video_id", "frame_idx", "condition", "split", "label", "method", "group_id"]
    results = []
    for recipe in ("clean", "augmented"):
        experiment = Path(reuse[recipe]["experiment"])
        for cov in (0, 1):
            common = experiment / f"k16_cov{cov}" / "heldout_none"
            coordinate_file = common / "coordinates.npz"
            coordinate_hash = sha256(coordinate_file)
            coords = restore_coordinates(coordinate_file)
            if len(coords.zmean) != 16 or bool(coords.use_covariates) != bool(cov):
                raise ValueError("Historical factorial coordinate dimension/covariates differ")
            for head in ("kan_spline", "mlp_gelu"):
                run = common / head / "params" / "seed0"
                marker = json.loads((run / "DONE.json").read_text(encoding="utf-8"))["files"]
                frame_file = run / "frame_scores.csv"
                if not marker.get("checkpoint.pt") or sha256(run / "checkpoint.pt") != marker["checkpoint.pt"]:
                    raise ValueError(f"Historical factorial checkpoint failed SHA-256: {recipe}/{head}/cov{cov}")
                if not marker.get("frame_scores.csv") or sha256(frame_file) != marker["frame_scores.csv"]:
                    raise ValueError(f"Historical factorial frame scores failed SHA-256: {recipe}/{head}/cov{cov}")
                saved = pd.read_csv(frame_file, usecols=columns + ["logit"], nrows=4096, keep_default_na=False)
                if saved.empty or not np.isfinite(saved.logit.to_numpy()).all():
                    raise ValueError("Historical factorial replay requires finite saved test logits")
                indices = saved.row.to_numpy(dtype=np.int64)
                if (indices < 0).any() or (indices >= len(meta)).any():
                    raise ValueError("Historical factorial replay frame rows are out of range")
                cache_rows = meta.iloc[indices].reset_index(drop=True)
                if not (cache_rows.split == "test").all():
                    raise ValueError("Historical factorial replay sample is not exclusively held-out data")
                for field in columns:
                    if not np.array_equal(saved[field].to_numpy(), cache_rows[field].to_numpy()):
                        raise ValueError(f"Historical factorial replay frame identity differs: {field}")
                checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
                if (checkpoint.get("head") != head or checkpoint.get("coordinates_sha256") != coordinate_hash
                        or checkpoint.get("in_features") != 16 + 4*cov):
                    raise ValueError("Historical factorial replay checkpoint identity differs")
                match = checkpoint["matching"]
                model = make_head(head, checkpoint["in_features"], int(match.get("hidden", 1)),
                                  **match.get("head_kwargs", {}))
                model.load_state_dict(checkpoint["state"], strict=True)
                model = model.to(device).float().eval()
                x = coords.transform(features[indices], quality[indices])
                current = predict(model, x, device)
                recorded = saved.logit.to_numpy(dtype=float)
                if not np.isfinite(current).all() or not np.allclose(current, recorded, rtol=1e-4, atol=2e-5):
                    raise ValueError(f"Historical factorial saved logits do not replay: {recipe}/{head}/cov{cov}")
                results.append({"recipe": recipe, "head": head, "covariates": bool(cov), "seed": 0,
                                "test_frames": len(saved), "rtol": 1e-4, "atol": 2e-5,
                                "max_abs_logit_difference": float(np.max(np.abs(current-recorded))),
                                "frame_scores_sha256": marker["frame_scores.csv"]})
                del model
    return results


def audit_factorial_cache(inputs, directory: Path, reuse, *, device="cpu"):
    """Audit one self-contained factorial cache without claiming baseline identity."""
    if set(reuse) != {"clean", "augmented"} or any(
            report.get("status") != "POSSIBLE_MATCH_REVIEW_REQUIRED" for report in reuse.values()):
        raise ValueError("Both exact historical k16 reuse audits must pass first")
    base, directory = Path(inputs.cache).resolve(), Path(directory).resolve()
    if directory == base or directory.is_relative_to(base) or base.is_relative_to(directory):
        raise ValueError("Factorial cache must be separate from the original cache")
    original = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    historical = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    _provenance(original, historical)
    for name in ("features.npy", "covariates.npy", "metadata.csv"):
        field = name.split(".")[0] + "_sha256"
        check = inputs.preflight.get("array_checks", {}).get(name, {})
        if check.get("status") != "VERIFIED" or check.get("expected_sha256") != original.get(field):
            raise ValueError(f"Original cache is not byte-verified: {name}")
        if not historical.get(field) or sha256(directory / name) != historical[field]:
            raise ValueError(f"Historical factorial cache checksum mismatch: {name}")

    meta = pd.read_csv(directory / "metadata.csv", keep_default_na=False)
    features = np.load(directory / "features.npy", mmap_mode="r")
    quality = np.load(directory / "covariates.npy", mmap_mode="r")
    if (features.shape != (len(meta), 1024) or quality.shape != (len(meta), 4)
            or features.dtype != np.float32 or quality.dtype != np.float32
            or not np.array_equal(meta.row.to_numpy(), np.arange(len(meta)))):
        raise ValueError("Historical factorial cache arrays/metadata do not align")
    if (inputs.features.shape != (len(inputs.meta), 1024) or inputs.quality.shape != (len(inputs.meta), 4)
            or inputs.features.dtype != np.float32 or inputs.quality.dtype != np.float32):
        raise ValueError("Original cache arrays/metadata do not align")
    if ((meta.dataset != "ffpp") & (meta.split != "test")).any():
        raise ValueError("Historical factorial cache contains non-test target data")
    if not meta.split.isin(["train", "val", "test"]).all():
        raise ValueError("Historical factorial cache contains an unknown split")
    held_out = _test_population(inputs.meta, meta)

    base_rows = inputs.meta.loc[(inputs.meta.dataset == "ffpp") & inputs.meta.split.isin(["train", "val"])
                                & (inputs.meta.condition == "clean")]
    source = meta.loc[(meta.dataset == "ffpp") & meta.split.isin(["train", "val"])].copy()
    if base_rows.empty or len(source) != len(base_rows) * (len(CONDITIONS) + 1):
        raise ValueError("Historical factorial cache has incomplete source transformations")
    if not source.condition.isin(["clean", *CONDITIONS]).all() or source.duplicated(_KEYS + ["condition"]).any():
        raise ValueError("Historical factorial cache has extra or duplicate source transformations")
    anchors = base_rows[_KEYS + ["row", *_ATTRS]].rename(
        columns={"row": "base_row", **{name: name + "_base" for name in _ATTRS}})
    paired = source.merge(anchors, on=_KEYS, how="left", validate="many_to_one", sort=False)
    if paired.base_row.isna().any():
        raise ValueError("Historical factorial source frame is missing its original identity")
    for field in _ATTRS:
        if not np.array_equal(paired[field].astype(str).to_numpy(), paired[field + "_base"].astype(str).to_numpy()):
            raise ValueError(f"Historical factorial source provenance differs: {field}")
    counts = paired.groupby("base_row").condition.nunique()
    if len(counts) != len(base_rows) or not counts.eq(len(CONDITIONS) + 1).all():
        raise ValueError("Historical factorial source transformations are incomplete")
    assert_source_splits(paired.loc[paired.split == "train"], paired.loc[paired.split == "val"])
    for split in ("train", "val"):
        if set(paired.loc[paired.split == split, "label"].unique()) != {0, 1}:
            raise ValueError(f"Historical factorial {split} must contain both source labels")

    clean = paired.loc[paired.condition == "clean"]
    total, values_outside, rows_outside, max_abs, sum_abs = 0, 0, 0, 0., 0.
    max_quality_abs = 0.
    for start in range(0, len(clean), 4096):
        part = clean.iloc[start:start + 4096]
        old_idx, base_idx = part.row.to_numpy(dtype=np.int64), part.base_row.to_numpy(dtype=np.int64)
        x, reference = features[old_idx], inputs.features[base_idx]
        q, q_reference = quality[old_idx], inputs.quality[base_idx]
        if not all(np.isfinite(array).all() for array in (x, reference, q, q_reference)):
            raise ValueError("Factorial/original clean source values are nonfinite")
        if not np.allclose(q, q_reference, rtol=2e-4, atol=2e-4):
            raise ValueError("Historical and original source quality covariates differ beyond tolerance")
        diff = np.abs(x - reference)
        within = np.isclose(x, reference, rtol=2e-4, atol=2e-4)
        total += diff.size
        values_outside += int((~within).sum())
        rows_outside += int((~within).any(axis=1).sum())
        max_abs = max(max_abs, float(diff.max()))
        sum_abs += float(np.sum(diff, dtype=np.float64))
        max_quality_abs = max(max_quality_abs, float(np.abs(q-q_reference).max()))

    replays = _replay_factorial_samples(meta, features, quality, reuse, device)
    expected = {(recipe, head, bool(cov)) for recipe in ("clean", "augmented")
                for head in ("kan_spline", "mlp_gelu") for cov in (0, 1)}
    actual = {(item["recipe"], item["head"], item["covariates"]) for item in replays}
    if len(replays) != 8 or actual != expected or any(item.get("seed") != 0 or item.get("test_frames", 0) < 1
                                                       for item in replays):
        raise ValueError("Historical factorial sampled replay coverage is incomplete")
    return {"status": "VERIFIED_WITHIN_HISTORICAL_FACTORIAL_CACHE",
            "manifest_sha256": sha256(directory / "manifest.json"),
            "source_pairs": len(paired) - len(clean), "source_clean_frames": len(clean),
            "held_out_frame_conditions_identical": held_out,
            "original_cache_features": {"identity": "DIFFERENT" if values_outside else "WITHIN_TOLERANCE",
                                        "rtol": 2e-4, "atol": 2e-4,
                                        "values_outside_tolerance": values_outside,
                                        "rows_outside_tolerance": rows_outside,
                                        "total_values": total, "max_abs_difference": max_abs,
                                        "mean_abs_difference": sum_abs / total},
            "source_quality_max_abs_difference": max_quality_abs,
            "encoder_devices": {"original": original["encoder_provenance"].get("device_name"),
                                "factorial": historical["encoder_provenance"].get("device_name")},
            "sampled_test_logit_replays": replays,
            "limitation": "The factorial uses the historical combined cache for every k16/k128 cell. "
                          "The original baseline uses its own verified cache and remains separate; "
                          "numerical cache differences do not establish their cause."}