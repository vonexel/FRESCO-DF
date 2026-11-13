from __future__ import annotations

from collections.abc import Sequence
from typing import Any
import numpy as np
import torch
from torch import Tensor, nn


def _prediction(model: nn.Module, x: Tensor) -> Tensor:
    output = model(x).reshape(-1)
    if output.shape[0] != x.shape[0]:
        raise ValueError("Expected exactly one logit per row")
    return output


def intervention_decomposition(model: nn.Module, clean_x: Tensor, changed_x: Tensor,
                               covariate_start: int) -> dict[str, np.ndarray]:
    """Exact two-block finite-difference decomposition of paired logit changes.
    total=f(zq,cq)-f(z0,c0); covariate=f(z0,cq)-f(z0,c0);
    embedding=f(zq,c0)-f(z0,c0); interaction=total-covariate-embedding.
    """
    if clean_x.ndim != 2 or clean_x.shape != changed_x.shape:
        raise ValueError("Require clean/changed tensors with the same [N,D] shape")
    if not 0 < covariate_start < clean_x.shape[1]:
        raise ValueError("covariate_start must split two nonempty feature blocks")
    covariate_hybrid, embedding_hybrid = clean_x.clone(), changed_x.clone()
    covariate_hybrid[:, covariate_start:] = changed_x[:, covariate_start:]
    embedding_hybrid[:, covariate_start:] = clean_x[:, covariate_start:]
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            clean = _prediction(model, clean_x)
            changed = _prediction(model, changed_x)
            covariate = _prediction(model, covariate_hybrid) - clean
            embedding = _prediction(model, embedding_hybrid) - clean
            total = changed - clean
            interaction = total - covariate - embedding
            result = {"clean": clean, "changed": changed, "total": total,
                      "covariate": covariate, "embedding": embedding,
                      "interaction": interaction}
            return {key: value.detach().cpu().numpy() for key, value in result.items()}
    finally:
        model.train(was_training)


def response_curves(model: nn.Module, X: Tensor, indices: Sequence[int], grid: Tensor,
                    batch_size: int = 1024) -> dict[str, Any]:
    if X.ndim != 2 or len(X) == 0 or grid.ndim != 1 or len(grid) == 0:
        raise ValueError("Require nonempty [N,D] background and one-dimensional grid")
    if batch_size < 1 or any(i < 0 or i >= X.shape[1] for i in indices):
        raise ValueError("Invalid batch size or feature index")
    grid = grid.to(device=X.device, dtype=X.dtype)
    curves: dict[str, Any] = {}
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            baseline = torch.cat([_prediction(model, chunk) for chunk in X.split(batch_size)])
            for index in indices:
                means, raw_means, lower, upper = [], [], [], []
                for value in grid:
                    rows = []
                    for chunk in X.split(batch_size):
                        changed = chunk.clone()
                        changed[:, index] = value
                        rows.append(_prediction(model, changed))
                    raw = torch.cat(rows)
                    delta = raw - baseline
                    means.append(delta.mean())
                    raw_means.append(raw.mean())
                    lower.append(delta.quantile(0.05))
                    upper.append(delta.quantile(0.95))
                curves[str(index)] = {
                    key: torch.stack(values).cpu().numpy()
                    for key, values in {"mean": means, "raw_mean": raw_means,
                                        "q05": lower, "q95": upper}.items()
                }
    finally:
        model.train(was_training)
    return {"grid": grid.cpu().numpy(), "curves": curves, "background_rows": len(X),
            "scale": "logit", "interval": "individual_response_quantiles"}


def fit_pd_polynomial(train_grid: np.ndarray, train_response: np.ndarray,
                      val_grid: np.ndarray, val_response: np.ndarray,
                      degrees: Sequence[int] = (1, 2, 3, 4, 5)) -> dict[str, Any]:
    tx, ty = np.asarray(train_grid, float).reshape(-1), np.asarray(train_response, float).reshape(-1)
    vx, vy = np.asarray(val_grid, float).reshape(-1), np.asarray(val_response, float).reshape(-1)
    if len(tx) != len(ty) or len(vx) != len(vy) or len(tx) < 2 or len(vx) == 0:
        raise ValueError("Require aligned train/validation grids and responses")
    if not all(np.isfinite(v).all() for v in (tx, ty, vx, vy)):
        raise ValueError("Polynomial data must be finite")
    center, scale = float(tx.mean()), float(tx.std())
    if scale == 0:
        raise ValueError("Training grid must vary")
    xt, xv = (tx - center) / scale, (vx - center) / scale
    candidates = []
    for degree in sorted(set(int(d) for d in degrees)):
        if degree < 1 or degree >= len(np.unique(tx)):
            continue
        coefficient = np.polynomial.polynomial.polyfit(xt, ty, degree)
        train_prediction = np.polynomial.polynomial.polyval(xt, coefficient)
        val_prediction = np.polynomial.polynomial.polyval(xv, coefficient)
        candidates.append({"degree": degree, "coefficients": coefficient.tolist(),
                           "train_rmse": float(np.sqrt(np.mean((train_prediction - ty) ** 2))),
                           "val_rmse": float(np.sqrt(np.mean((val_prediction - vy) ** 2)))})
    if not candidates:
        raise ValueError("No degree is identifiable from the training grid")
    selected = min(candidates, key=lambda item: (item["val_rmse"], item["degree"]))
    return {**selected, "center": center, "scale": scale, "candidates": candidates,
            "scope": "one_dimensional_prediction_response"}


def predict_pd_polynomial(fit: dict[str, Any], grid: np.ndarray) -> np.ndarray:
    x = (np.asarray(grid, float) - fit["center"]) / fit["scale"]
    return np.polynomial.polynomial.polyval(x, fit["coefficients"])


def curve_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    a, b = np.asarray(first, float).reshape(-1), np.asarray(second, float).reshape(-1)
    if a.shape != b.shape or len(a) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Require aligned finite curves with at least two points")
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def plasticity_score(model: nn.Module, initial_model: nn.Module, x: Tensor) -> np.ndarray:
    trained_layer = next((m for m in model.modules() if callable(getattr(m, "edges", None))), None)
    initial_layer = next((m for m in initial_model.modules() if callable(getattr(m, "edges", None))), None)
    if trained_layer is None or initial_layer is None:
        raise ValueError("Plasticity requires explicit first-layer edge functions")
    if type(trained_layer) is not type(initial_layer):
        raise ValueError("Trained and initial edge layers must use the same architecture")
    modes = model.training, initial_model.training
    model.eval()
    initial_model.eval()
    try:
        with torch.inference_mode():
            delta = (trained_layer.edges(x) - initial_layer.edges(x)).abs()
            return delta.flatten(1).quantile(0.5, dim=1).cpu().numpy()
    finally:
        model.train(modes[0])
        initial_model.train(modes[1])