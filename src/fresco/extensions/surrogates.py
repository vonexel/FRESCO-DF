from __future__ import annotations

import time
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from .paired import assert_source_splits


class ZeroResponse:
    def predict(self, X):
        return np.zeros(len(X), dtype=float)


def fit_surrogates(train_x, train_y, train_meta, val_x, val_y, val_meta, *, seed,
                   include_ebm=True, ebm_interactions=False):
    """Select equal-size, predeclared budgets on group-disjoint source validation.

    Inputs are exactly the four clean and four changed observed covariates.
    EBM is never silently replaced if its separate dependency is unavailable.
    """
    assert_source_splits(train_meta, val_meta)
    tx, ty, vx, vy = (np.asarray(v, dtype=float) for v in (train_x, train_y, val_x, val_y))
    if (tx.ndim != 2 or vx.ndim != 2 or tx.shape[1] != 8 or vx.shape[1] != 8
            or ty.shape != (len(tx),) or vy.shape != (len(vx),)
            or len(tx) != len(train_meta) or len(vx) != len(val_meta)
            or not all(np.isfinite(a).all() for a in (tx, ty, vx, vy))):
        raise ValueError("Require aligned finite eight-quality-feature source pairs")
    if seed not in range(10):
        raise ValueError("Scientific teacher seed must be 0..9")
    candidates = {
        "ridge": [(str(a), make_pipeline(StandardScaler(), Ridge(alpha=a))) for a in (0.1, 1., 10.)],
        "gam": [(str(a), make_pipeline(SplineTransformer(n_knots=5, degree=3, include_bias=False,
                                                          extrapolation="constant"),
                                        StandardScaler(), Ridge(alpha=a))) for a in (0.1, 1., 10.)],
    }
    blocked = {}
    if include_ebm:
        try:
            from interpret.glassbox import ExplainableBoostingRegressor
        except ImportError as exc:
            blocked["ebm"] = f"BLOCKED_DEPENDENCY: {exc}"
            if ebm_interactions:
                blocked["ebm_interactions"] = blocked["ebm"]
        else:
            def make_ebm(rate, interactions):
                return ExplainableBoostingRegressor(
                    interactions=interactions, max_bins=128, max_interaction_bins=32,
                    max_leaves=2, learning_rate=rate, outer_bags=1, inner_bags=0,
                    validation_size=0, max_rounds=300, min_samples_leaf=10,
                    n_jobs=1, random_state=seed)
            candidates["ebm"] = [(str(rate), make_ebm(rate, 0)) for rate in (0.02, 0.04, 0.08)]
            if ebm_interactions:
                candidates["ebm_interactions"] = [(str(rate), make_ebm(rate, 2))
                                                   for rate in (0.02, 0.04, 0.08)]
    fitted = {"zero": ZeroResponse()}
    metadata = {"zero": {"status": "RUN", "validation_rmse": float(np.sqrt(np.mean(vy**2))),
                          "fit_seconds": 0., "selected": None}}
    for family, options in candidates.items():
        trials = []
        for setting, model in options:
            start = time.perf_counter()
            model.fit(tx, ty)
            prediction = np.asarray(model.predict(vx), dtype=float)
            if prediction.shape != vy.shape or not np.isfinite(prediction).all():
                raise ValueError(f"{family} produced invalid validation predictions")
            trials.append((float(np.sqrt(np.mean((prediction-vy)**2))), setting,
                           time.perf_counter()-start, model))
        best = min(trials, key=lambda v: (v[0], v[1]))
        fitted[family] = best[3]
        metadata[family] = {"status": "RUN", "validation_rmse": best[0], "selected": best[1],
                            "fit_seconds": sum(v[2] for v in trials),
                            "trials": [{"setting": t[1], "validation_rmse": t[0],
                                        "fit_seconds": t[2]} for t in trials],
                            "interactions": 2 if family == "ebm_interactions" else 0}
    for family, reason in blocked.items():
        metadata[family] = {"status": "BLOCKED_DEPENDENCY", "reason": reason}
    return fitted, metadata