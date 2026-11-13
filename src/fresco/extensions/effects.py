from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Effect:
    grid: np.ndarray
    values: np.ndarray
    occupied: np.ndarray
    counts: np.ndarray
    kind: str


def _check(source, source_meta, index, bins):
    x = np.asarray(source, dtype=float)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Finite nonempty source matrix required")
    if (not hasattr(source_meta, "columns") or not {"dataset", "split"}.issubset(source_meta.columns)
            or len(source_meta) != len(x)
            or not ((source_meta.dataset == "ffpp") & (source_meta.split == "train")).all()):
        raise ValueError("Effect fitting accepts verified FF++ source training rows only")
    if not 0 <= index < x.shape[1] or bins < 1:
        raise ValueError("Invalid feature index or bin count")
    return x


def fit_pd(predict, source, splits, index, *, bins=20):
    x = _check(source, splits, index, bins)
    grid = np.unique(np.quantile(x[:, index], np.linspace(0, 1, bins+1)))
    if len(grid) == 1:
        return Effect(grid, np.zeros(1), np.ones(0, bool), np.zeros(0, int), "pd")
    values = []
    for point in grid:
        changed = x.copy()
        changed[:, index] = point
        value = np.asarray(predict(changed), dtype=float).reshape(-1)
        if len(value) != len(x) or not np.isfinite(value).all():
            raise ValueError("Teacher returned invalid PD predictions")
        values.append(value.mean())
    values = np.asarray(values)
    return Effect(grid, values-values.mean(), np.ones(len(grid)-1, bool),
                  np.histogram(x[:, index], bins=grid)[0], "pd")


def fit_ale(predict, source, splits, index, *, bins=20, min_bin_count=5):
    x = _check(source, splits, index, bins)
    if min_bin_count < 1:
        raise ValueError("min_bin_count must be positive")
    grid = np.unique(np.quantile(x[:, index], np.linspace(0, 1, bins+1)))
    if len(grid) == 1:
        return Effect(grid, np.zeros(1), np.ones(0, bool), np.zeros(0, int), "ale")
    membership = np.clip(np.searchsorted(grid, x[:, index], side="right")-1, 0, len(grid)-2)
    counts = np.bincount(membership, minlength=len(grid)-1)
    occupied = counts >= min_bin_count
    increments = np.zeros(len(counts))
    for j, n in enumerate(counts):
        if not occupied[j]:
            continue
        lower = x[membership == j].copy()
        upper = lower.copy()
        lower[:, index], upper[:, index] = grid[j], grid[j+1]
        a, b = np.asarray(predict(lower), float).reshape(-1), np.asarray(predict(upper), float).reshape(-1)
        if len(a) != n or len(b) != n or not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise ValueError("Teacher returned invalid ALE predictions")
        increments[j] = (b-a).mean()
    values = np.r_[0., np.cumsum(increments)]
    centered = np.interp(x[:, index], grid, values)
    values -= centered.mean()
    return Effect(grid, values, occupied, counts, "ale")


def effect_delta(effect: Effect, clean, changed):
    a, b = np.asarray(clean, float), np.asarray(changed, float)
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError("Effect endpoints must be aligned one-dimensional arrays")
    finite = np.isfinite(a) & np.isfinite(b)
    unchanged = finite & (a == b)
    covered = unchanged.copy()
    delta = np.zeros(len(a))
    if len(effect.grid) > 1:
        inside = (finite & ~unchanged & (a >= effect.grid[0]) & (a <= effect.grid[-1])
                  & (b >= effect.grid[0]) & (b <= effect.grid[-1]))
        ia = np.clip(np.searchsorted(effect.grid, np.minimum(a, b), side="right")-1, 0, len(effect.occupied)-1)
        ib = np.clip(np.searchsorted(effect.grid, np.maximum(a, b), side="left")-1, 0, len(effect.occupied)-1)
        bad_prefix = np.r_[0, np.cumsum(~effect.occupied)]
        inside &= (bad_prefix[ib+1]-bad_prefix[ia]) == 0
        covered |= inside
        delta[inside] = np.interp(b[inside], effect.grid, effect.values)-np.interp(a[inside], effect.grid, effect.values)
    delta[~covered] = np.nan
    return delta, covered


def additive_delta(effects, clean, changed):
    a, b = np.asarray(clean), np.asarray(changed)
    if a.ndim != 2 or a.shape != b.shape or a.shape[1] != len(effects):
        raise ValueError("One effect is required for every aligned quality feature")
    predictions = [effect_delta(effect, a[:, i], b[:, i]) for i, effect in enumerate(effects)]
    covered = np.logical_and.reduce([item[1] for item in predictions])
    delta = np.sum([item[0] for item in predictions], axis=0)
    delta[~covered] = np.nan
    return delta, covered