from __future__ import annotations


import json
from pathlib import Path
import numpy as np
import pandas as pd
from .inventory import sha256
from .paired import assert_source_splits
from .source_cache import CONDITIONS


def load_source_pairs(inputs, directory: Path):
    directory = Path(directory).resolve()
    if directory == inputs.cache or directory.is_relative_to(inputs.cache) or inputs.cache.is_relative_to(directory):
        raise ValueError("Transformed source pairs require a separate cache")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    encoder = manifest.get("encoder_provenance", {})
    if (encoder.get("checkpoint_sha256") != "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
            or not encoder.get("projection_consistency", {}).get("passed")):
        raise ValueError("Transformed pairs lack the verified original frozen CLIP encoder")
    for name in ("features.npy", "covariates.npy", "metadata.csv"):
        expected = manifest.get(name.split(".")[0]+"_sha256")
        if not expected or sha256(directory/name) != expected:
            raise ValueError(f"Transformed source-cache checksum mismatch: {name}")
    features = np.load(directory / "features.npy", mmap_mode="r")
    quality = np.load(directory / "covariates.npy", mmap_mode="r")
    meta = pd.read_csv(directory / "metadata.csv", keep_default_na=False)
    if (features.shape != (len(meta), 1024) or quality.shape != (len(meta), 4)
            or not np.array_equal(meta.row.to_numpy(), np.arange(len(meta)))):
        raise ValueError("Source cache array shapes or row indices disagree")
    original_hashes = {name.split(".")[0]: value["expected_sha256"]
                       for name, value in inputs.preflight["array_checks"].items()}
    source = inputs.meta.loc[(inputs.meta.dataset == "ffpp") & inputs.meta.split.isin(["train", "val"])
                             & (inputs.meta.condition == "clean")]
    if manifest.get("source_only") is True:
        if manifest.get("identity", {}).get("original_cache_hashes") != original_hashes:
            raise ValueError("Source cache was not built from this exact original cache")
        if set(meta.condition) != set(CONDITIONS):
            raise ValueError("Source-only cache must contain exactly the three declared transformations")
        markers = sorted((directory / "chunks").glob("*.json"))
        if not markers:
            raise ValueError("Source-only cache lacks per-chunk encoder provenance")
        devices = set()
        for marker_path in markers:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            prov = marker.get("encoder_provenance", {})
            if (marker.get("signature") != manifest.get("signature")
                    or not prov.get("projection_consistency", {}).get("passed")
                    or prov.get("checkpoint_sha256") != encoder.get("checkpoint_sha256")):
                raise ValueError("Source chunk has mismatched encoder provenance")
            devices.add(prov.get("device_name"))
        if len(devices) != 1 or (manifest.get("identity", {}).get("device", "").startswith("cuda")
                                 and None in devices):
            raise ValueError("Source cache mixes GPU identities or lacks a recorded GPU name")
        rows = meta.copy()
        rows["source_row"] = rows.row
    else:
        if (manifest.get("encoder") != "ViT-L-14" or manifest.get("pretrained") != "openai"
                or manifest.get("encoder_provenance", {}).get("checkpoint_sha256")
                != "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"):
            raise ValueError("Historical transformed cache does not use the verified original CLIP weights")
        source_conditions = manifest.get("source_conditions", [])
        if set(source_conditions) != {"clean", *CONDITIONS}:
            raise ValueError("Historical cache source transforms differ from the frozen recipe")
        clean = meta.loc[(meta.dataset == "ffpp") & meta.split.isin(["train", "val"])
                         & (meta.condition == "clean")]
        keys = ["dataset", "video_id", "frame_idx"]
        base = source[keys+["row"]].copy().rename(columns={"row": "base_row"})
        clean = clean.merge(base, on=keys, how="left", validate="one_to_one")
        if clean.base_row.isna().any() or len(clean) != len(base):
            raise ValueError("Historical source clean frames differ from original source")
        for start in range(0, len(clean), 4096):
            part = clean.iloc[start:start+4096]
            if (not np.allclose(features[part.row.to_numpy()], inputs.features[part.base_row.to_numpy()],
                                rtol=2e-4, atol=2e-4)
                    or not np.allclose(quality[part.row.to_numpy()], inputs.quality[part.base_row.to_numpy()],
                                       rtol=2e-4, atol=2e-4)):
                raise ValueError("Historical and original clean source rows differ beyond declared tolerance")
        rows = meta.loc[(meta.dataset == "ffpp") & meta.split.isin(["train", "val"])
                        & meta.condition.isin(CONDITIONS)].copy().rename(columns={"row": "source_row"})
        rows = rows.merge(base, on=keys, how="left", validate="many_to_one")
    if rows.empty or rows.base_row.isna().any() or not rows.condition.isin(CONDITIONS).all():
        raise ValueError("Missing source clean anchors or declared transformations")
    rows["base_row"] = rows.base_row.astype(np.int64)
    anchor = inputs.meta.iloc[rows.base_row.to_numpy()].reset_index(drop=True)
    rows = rows.reset_index(drop=True)
    for name in ("dataset", "video_id", "frame_idx", "split", "label", "method", "group_id"):
        if not np.array_equal(rows[name].astype(str).to_numpy(), anchor[name].astype(str).to_numpy()):
            raise ValueError(f"Source pair provenance disagrees for {name}")
    if not (anchor.condition == "clean").all() or not (rows.dataset == "ffpp").all():
        raise ValueError("Only FF++ clean anchors may be used for source fitting")
    keys = ["base_row", "condition"]
    if rows.duplicated(keys).any() or len(rows) != len(source)*len(CONDITIONS):
        raise ValueError("Incomplete or duplicated source train/validation intervention pairs")
    counts = rows.groupby("base_row").condition.nunique()
    if len(counts) != len(source) or not counts.eq(len(CONDITIONS)).all():
        raise ValueError("Every source clean frame must have each declared transform")
    assert_source_splits(rows.loc[rows.split == "train"], rows.loc[rows.split == "val"])
    return {"path": directory, "manifest": manifest, "metadata": rows,
            "features": features, "covariates": quality,
            "manifest_sha256": sha256(directory / "manifest.json")}


def transform_source_pairs(inputs, source, batch=4096):
    meta = source["metadata"]
    clean = np.empty((len(meta), 20), dtype=np.float32)
    changed = np.empty_like(clean)
    raw_quality = np.empty((len(meta), 8), dtype=np.float32)
    for start in range(0, len(meta), batch):
        part = meta.iloc[start:start+batch]
        base = part.base_row.to_numpy(dtype=np.int64)
        transformed = part.source_row.to_numpy(dtype=np.int64)
        clean[start:start+len(part)] = inputs.coordinates(base)
        changed[start:start+len(part)] = inputs.coords.transform(source["features"][transformed],
                                                                  source["covariates"][transformed])
        raw_quality[start:start+len(part), :4] = inputs.quality[base]
        raw_quality[start:start+len(part), 4:] = source["covariates"][transformed]
    return clean, changed, raw_quality