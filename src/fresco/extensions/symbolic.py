from __future__ import annotations

import copy
import time
import numpy as np
import torch
from fresco.heads import SplineEdgeLayer
from .paired import assert_source_splits


def _design(x, family):
    x = np.asarray(x, dtype=np.float64)
    if family == "zero":
        return np.zeros((len(x), 0))
    if family == "constant":
        return np.ones((len(x), 1))
    if family == "linear":
        return np.column_stack((np.ones(len(x)), x))
    if family == "quadratic":
        return np.column_stack((np.ones(len(x)), x, x*x))
    if family == "cubic":
        return np.column_stack((np.ones(len(x)), x, x*x, x*x*x))
    if family == "silu":
        # The teacher's actual residual has precisely this form.
        sigmoid = np.exp(-np.logaddexp(0., -x))
        return np.column_stack((np.ones(len(x)), x*sigmoid))
    raise ValueError(f"Unknown symbolic family: {family}")


def extract_edges(model, inputs):
    """Return real edge values and layer inputs, including knots and biases"""
    if not hasattr(model, "network") or not len(model.network):
        raise ValueError("Expected the project's actual spline-KAN LogitHead")
    layers = list(model.network)
    if any(not isinstance(layer, SplineEdgeLayer) for layer in layers):
        raise ValueError("Every KAN layer must be an actual SplineEdgeLayer")
    x = torch.as_tensor(inputs, dtype=torch.float32)
    if x.ndim != 2 or not len(x) or not torch.isfinite(x).all():
        raise ValueError("Finite nonempty source inputs required")
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("Move the copied teacher to CPU before extracting edges")
    was_training = model.training
    model.eval()
    result = []
    try:
        with torch.inference_mode():
            for index, layer in enumerate(layers):
                edges = layer.edges(x)
                result.append({"layer": index, "inputs": x.numpy().copy(), "edges": edges.numpy().copy(),
                               "knots": layer.knots.cpu().numpy().copy(), "degree": layer.degree,
                               "domain": [layer.low, layer.high], "bias": layer.bias.detach().numpy().copy()})
                x = layer(x)
    finally:
        model.train(was_training)
    return result


def bounded_attempt(model, train_x, val_x, *, train_meta, val_meta, tolerance=0.02, max_rows=2048,
                    vocabulary=("zero", "constant", "linear", "silu", "quadratic", "cubic")):
    """Fit each actual edge with a fixed, bounded vocabulary and source validation.

    Each input uses the first max_rows deterministic observations. Selection is
    the simplest candidate with validation edge RMSE <= tolerance; no target
    samples enter this calculation. This function does not replace network edges.
    """
    assert_source_splits(train_meta, val_meta)
    if len(train_x) != len(train_meta) or len(val_x) != len(val_meta):
        raise ValueError("Each source input must have matching provenance metadata")
    if tolerance <= 0 or max_rows < 2 or not vocabulary:
        raise ValueError("Positive tolerance, at least two rows and a vocabulary are required")
    start = time.perf_counter()
    clone = copy.deepcopy(model).cpu().eval()
    train, validation = extract_edges(clone, np.asarray(train_x)[:max_rows]), extract_edges(clone, np.asarray(val_x)[:max_rows])
    records = []
    for source, holdout in zip(train, validation):
        for output in range(source["edges"].shape[1]):
            for inp in range(source["edges"].shape[2]):
                tx, ty = source["inputs"][:, inp], source["edges"][:, output, inp]
                vx, vy = holdout["inputs"][:, inp], holdout["edges"][:, output, inp]
                candidates = []
                for complexity, family in enumerate(vocabulary):
                    design = _design(tx, family)
                    coefficient = np.linalg.lstsq(design, ty, rcond=None)[0] if design.shape[1] else np.array([])
                    prediction = _design(vx, family) @ coefficient
                    rmse = float(np.sqrt(np.mean((prediction-vy)**2)))
                    candidates.append((rmse, complexity, family, coefficient))
                feasible = [c for c in candidates if c[0] <= tolerance]
                best = min(feasible, key=lambda c: (c[1], c[0])) if feasible else min(candidates, key=lambda c: c[0])
                records.append({"layer": source["layer"], "output": output, "input": inp,
                                "family": best[2], "coefficients": best[3].tolist(),
                                "validation_rmse": best[0], "accepted": bool(feasible),
                                "source_min": float(tx.min()), "source_max": float(tx.max()),
                                "knots": source["knots"].tolist(), "degree": source["degree"],
                                "spline_domain": source["domain"]})
    # zero edges accepted on source validation are actually pruned on a deep copy.
    pruned = copy.deepcopy(clone)
    with torch.no_grad():
        for edge in records:
            if edge["accepted"] and edge["family"] == "zero":
                layer = pruned.network[edge["layer"]]
                layer.base_weight[edge["output"], edge["input"]] = 0
                layer.coefficients[edge["output"], edge["input"], :] = 0
        sample = torch.as_tensor(np.asarray(val_x)[:max_rows], dtype=torch.float32)
        before = clone(sample).cpu().numpy()
        after = pruned(sample).cpu().numpy()
    return {"status": "FULL_SYMBOLIFICATION_NOT_EVALUATED", "records": records,
            "accepted_edges": sum(r["accepted"] for r in records), "total_edges": len(records),
            "tolerance": tolerance, "max_rows": max_rows, "vocabulary": list(vocabulary),
            "pruned_edges": sum(r["accepted"] and r["family"] == "zero" for r in records),
            "pruned_clone_validation_rmse": float(np.sqrt(np.mean((before-after)**2))),
            "pruned_clone_validation_max_abs": float(np.max(np.abs(before-after))),
            "elapsed_seconds": time.perf_counter()-start,
            "reason": "Actual edges were fitted, but no whole-model symbolic clone was evaluated."}