from __future__ import annotations

import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from fresco.heads import SplineEdgeLayer


class HybridSplineLayer(nn.Module):
    """Replace accepted actual edge functions; retain rejected spline edges."""

    def __init__(self, original, records):
        super().__init__()
        if not isinstance(original, SplineEdgeLayer):
            raise TypeError("Expected an actual FRESCO spline edge layer")
        self.original = copy.deepcopy(original)
        shape = tuple(original.base_weight.shape)
        coefficients = np.zeros((5, *shape), dtype=np.float32)  # 1,x,x²,x³,SiLU(x)
        accepted = np.zeros(shape, dtype=bool)
        seen = set()
        for record in records:
            o, i = int(record["output"]), int(record["input"])
            if not (0 <= o < shape[0] and 0 <= i < shape[1]) or (o, i) in seen:
                raise ValueError("Duplicate or invalid actual edge identity")
            seen.add((o, i))
            if not record["accepted"]:
                continue
            accepted[o, i] = True
            family = record["family"]
            value = np.asarray(record["coefficients"], dtype=np.float32)
            expected = {"zero": 0, "constant": 1, "linear": 2, "quadratic": 3, "cubic": 4, "silu": 2}
            if family not in expected or len(value) != expected[family] or not np.isfinite(value).all():
                raise ValueError("Invalid fitted symbolic edge")
            if family == "silu":
                coefficients[0, o, i], coefficients[4, o, i] = value
            elif len(value):
                coefficients[:len(value), o, i] = value
        if len(seen) != shape[0]*shape[1]:
            raise ValueError("Every actual edge must be represented exactly once")
        self.register_buffer("accepted", torch.as_tensor(accepted))
        self.register_buffer("symbolic_coefficients", torch.as_tensor(coefficients))
        self.all_accepted = bool(accepted.all())

    def forward(self, x):
        if x.ndim != 2:
            raise ValueError("Expected [batch, input] KAN layer values")
        c = self.symbolic_coefficients
        z = x.unsqueeze(1)
        symbolic = c[0]+c[1]*z+c[2]*z.square()+c[3]*z.pow(3)+c[4]*F.silu(z)
        # When every edge is symbolified, evaluate the expression graph without
        # calling any original spline edges. The saved copy still retains the
        # old weights as provenance, which is reflected in its model size.
        edges = symbolic if self.all_accepted else torch.where(self.accepted.unsqueeze(0), symbolic,
                                                               self.original.edges(x))
        return edges.sum(-1)+self.original.bias


def build_clones(teacher, records):
    """Create separate pruned and mixed models; never mutate the teacher."""
    if not hasattr(teacher, "network") or any(not isinstance(layer, SplineEdgeLayer) for layer in teacher.network):
        raise ValueError("Expected the original two-layer spline-KAN teacher")
    pruned, hybrid = copy.deepcopy(teacher).eval(), copy.deepcopy(teacher).eval()
    total, accepted, pruned_count = 0, 0, 0
    with torch.no_grad():
        for layer_index, layer in enumerate(teacher.network):
            rows = [r for r in records if r["layer"] == layer_index]
            layer_clone = HybridSplineLayer(layer, rows)
            hybrid.network[layer_index] = layer_clone
            total += len(rows)
            accepted += sum(bool(r["accepted"]) for r in rows)
            for record in rows:
                if record["accepted"] and record["family"] == "zero":
                    o, i = int(record["output"]), int(record["input"])
                    pruned.network[layer_index].base_weight[o, i] = 0
                    pruned.network[layer_index].coefficients[o, i, :] = 0
                    pruned_count += 1
    return pruned, hybrid, {"total_edges": total, "accepted_symbolic_edges": accepted,
                            "retained_spline_edges": total-accepted, "pruned_edges": pruned_count,
                            "full_symbolification": accepted == total}