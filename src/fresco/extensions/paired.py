from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd


def assert_source_splits(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    required = {"dataset", "split", "video_id", "group_id"}
    if not required.issubset(train.columns) or not required.issubset(validation.columns):
        raise ValueError("Source rows require dataset, split, video_id and group_id")
    if train.empty or validation.empty:
        raise ValueError("Nonempty source train and validation are required")
    if not ((train.dataset == "ffpp") & (train.split == "train")).all():
        raise ValueError("Fitting rows must be FF++ train only")
    if not ((validation.dataset == "ffpp") & (validation.split == "val")).all():
        raise ValueError("Selection rows must be FF++ validation only")
    if set(train.video_id) & set(validation.video_id) or set(train.group_id) & set(validation.group_id):
        raise ValueError("Source train and validation video/group leakage")
    if train.group_id.isna().any() or validation.group_id.isna().any():
        raise ValueError("Missing source group identifier")


def paired_rows(meta: pd.DataFrame, *, split: str, dataset: str | None = None) -> pd.DataFrame:
    """Join real clean/degraded rows; fail on missing or inconsistent pairs"""
    keys = ["dataset", "video_id", "frame_idx"]
    attributes = ["label", "method", "group_id", "split"]
    required = set(keys + attributes + ["condition"])
    if not required.issubset(meta.columns):
        raise ValueError(f"Missing paired metadata columns: {sorted(required-set(meta.columns))}")
    rows = meta.loc[meta.split == split].copy()
    if dataset is not None:
        rows = rows.loc[rows.dataset == dataset].copy()
    rows["source_row"] = rows.index
    if rows.duplicated(keys + ["condition"]).any():
        raise ValueError("Duplicate frame-condition key")
    clean = rows.loc[rows.condition == "clean", keys + attributes + ["source_row"]]
    changed = rows.loc[rows.condition != "clean"]
    result = changed.merge(clean, on=keys, how="left", suffixes=("", "_clean"), validate="many_to_one")
    if result.source_row_clean.isna().any():
        raise ValueError("Changed frame without its clean anchor")
    for attribute in attributes:
        if not result[attribute].equals(result[attribute + "_clean"]):
            raise ValueError(f"Paired metadata differs for {attribute}")
    result = result.rename(columns={"source_row": "changed_row", "source_row_clean": "clean_row"})
    result["clean_row"] = result.clean_row.astype(np.int64)
    return result


def sample_hash(frame: pd.DataFrame) -> str:
    columns = ["dataset", "video_id", "frame_idx", "condition"]
    if not set(columns).issubset(frame.columns) or frame.duplicated(columns).any():
        raise ValueError("Sample IDs must be unique and include dataset, video, frame and condition")
    # JSON encoding is unambiguous even when an ID contains a separator
    payload = frame[columns].sort_values(columns).to_json(orient="records", force_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def common_mask(*masks) -> np.ndarray:
    if not masks:
        raise ValueError("At least one support mask required")
    arrays = [np.asarray(mask) for mask in masks]
    if any(a.dtype != bool or a.ndim != 1 or a.shape != arrays[0].shape for a in arrays):
        raise ValueError("All masks must be aligned one-dimensional booleans")
    return np.logical_and.reduce(arrays)


def response_metrics(actual, predicted, video_ids, *, mask=None, epsilon=1e-8, sign_tolerance=1e-8):
    y, p, ids = np.asarray(actual, float), np.asarray(predicted, float), np.asarray(video_ids)
    if y.ndim != 1 or p.shape != y.shape or ids.shape != y.shape:
        raise ValueError("Aligned one-dimensional actual, predicted and video IDs required")
    if not np.isfinite(epsilon) or epsilon <= 0 or not np.isfinite(sign_tolerance) or sign_tolerance < 0:
        raise ValueError("Invalid denominator or sign tolerance")
    selected = np.ones(len(y), dtype=bool) if mask is None else np.asarray(mask)
    if selected.dtype != bool or selected.shape != y.shape:
        raise ValueError("Mask must be an aligned boolean array")
    if not np.isfinite(y[selected]).all() or not np.isfinite(p[selected]).all():
        raise ValueError("Selected responses must be finite")
    y, p, ids = y[selected], p[selected], ids[selected]
    if not len(y):
        return {"status": "UNDEFINED", "reason": "empty population", "pairs": 0, "videos": 0}
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inverse]
    w /= w.sum()
    error = p-y
    denominator = float(np.sum(w*y*y))
    mse = float(np.sum(w*error*error))
    nonzero = np.abs(y) > sign_tolerance
    denom_sign = float(w[nonzero].sum())
    return {"status": "COMPUTED" if denominator > epsilon**2 else "UNDEFINED",
            "reason": None if denominator > epsilon**2 else "near-zero response denominator",
            "pairs": len(y), "videos": len(counts), "weighting": "equal video",
            "denominator_mean_square": denominator, "error_mean_square": mse,
            "G": float(np.sqrt(mse/denominator)) if denominator > epsilon**2 else None,
            "rmse": float(np.sqrt(mse)), "mae": float(np.sum(w*np.abs(error))),
            "bias": float(np.sum(w*error)), "sign_tolerance": sign_tolerance,
            "sign_excluded": int((~nonzero).sum()), "sign_trials": int(nonzero.sum()),
            "sign_agreement": (float(np.sum(w[nonzero]*(np.sign(y[nonzero]) == np.sign(p[nonzero])))
                                     / denom_sign) if denom_sign else None)}