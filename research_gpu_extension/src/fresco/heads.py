"""Small, original PyTorch classifier heads used by the FRESCO comparison.

These implementations use published mathematical *ideas*, not vendored library
code. ``kan_spline`` follows the edge-function construction of Liu et al.,
https://arxiv.org/abs/2404.19756; ``kan_fast`` uses Gaussian radial basis
functions as in FastKAN, https://arxiv.org/abs/2405.06721. ``kan_rational`` is a
grouped rational-activation control inspired by KAT,
https://arxiv.org/abs/2409.10594, not a reproduction of its architecture or CUDA
kernels. ``kan_wavelet`` is our Mexican-hat edge-function implementation, inspired
by Wav-KAN, https://arxiv.org/abs/2405.12832. None is claimed numerically equivalent
to an upstream package. The frozen representation is supplied by the caller.
"""

from __future__ import annotations

import math
import time
from typing import Any
import torch
from torch import Tensor, nn
from torch.nn import functional as F


HEAD_NAMES = (
    "linear", "mlp_relu", "mlp_gelu", "mlp_bspline", "kan_spline",
    "kan_fast", "kan_rational", "kan_wavelet",
)


def open_uniform_knots(grid_size: int = 8, degree: int = 3, low: float = -3.0, high: float = 3.0) -> Tensor:
    """Open uniform knots: ``grid_size`` intervals, ``grid_size+degree`` bases"""
    if grid_size < 1 or degree < 0 or high <= low:
        raise ValueError("Require grid_size >= 1, degree >= 0, and high > low")
    interior = torch.linspace(low, high, grid_size + 1)
    return torch.cat((interior[:1].repeat(degree), interior, interior[-1:].repeat(degree)))


def bspline_basis(x: Tensor, knots: Tensor, degree: int = 3) -> Tensor:
    if degree < 0 or knots.ndim != 1 or len(knots) < 2 * (degree + 1):
        raise ValueError("Invalid degree or knot vector")
    knots = knots.to(device=x.device, dtype=x.dtype)
    u = x.unsqueeze(-1)
    basis = ((u >= knots[:-1]) & (u < knots[1:])).to(x.dtype)
    for order in range(1, degree + 1):
        width = len(knots) - order - 1
        left_denom = knots[order : order + width] - knots[:width]
        right_denom = knots[order + 1 : order + 1 + width] - knots[1 : 1 + width]
        left = (u - knots[:width]) / torch.where(left_denom > 0, left_denom, 1)
        right = (knots[order + 1 : order + 1 + width] - u) / torch.where(
            right_denom > 0, right_denom, 1
        )
        basis = (
            left * basis[..., :width] * (left_denom > 0)
            + right * basis[..., 1 : width + 1] * (right_denom > 0)
        )
    endpoint = torch.zeros_like(basis)
    endpoint[..., -1] = 1
    return torch.where(u == knots[-1], endpoint, basis)


class BSplineActivation(nn.Module):
    def __init__(self, features: int, grid_size: int = 8, degree: int = 3,
                 low: float = -3.0, high: float = 3.0):
        super().__init__()
        if degree < 1:
            raise ValueError("Node splines require degree >= 1")
        self.degree, self.low, self.high = degree, low, high
        knots = open_uniform_knots(grid_size, degree, low, high)
        self.register_buffer("knots", knots)
        greville = torch.stack([knots[i + 1 : i + 1 + degree].mean()
                                for i in range(grid_size + degree)])
        self.coefficients = nn.Parameter(greville.repeat(features, 1))
        self.tail_slopes = nn.Parameter(torch.ones(features, 2))

    def forward(self, x: Tensor) -> Tensor:
        bounded = x.clamp(self.low, self.high)
        value = (bspline_basis(bounded, self.knots, self.degree) * self.coefficients).sum(-1)
        return (value + (x - self.low).clamp(max=0) * self.tail_slopes[:, 0]
                + (x - self.high).clamp(min=0) * self.tail_slopes[:, 1])


class SplineEdgeLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, grid_size: int = 8,
                 degree: int = 3, low: float = -3.0, high: float = 3.0):
        super().__init__()
        self.degree, self.low, self.high = degree, low, high
        self.register_buffer("knots", open_uniform_knots(grid_size, degree, low, high))
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.coefficients = nn.Parameter(torch.empty(out_features, in_features, grid_size + degree))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.coefficients, std=0.03 / math.sqrt(in_features))

    def edges(self, x: Tensor) -> Tensor:
        basis = bspline_basis(x.clamp(self.low, self.high), self.knots, self.degree)
        return (F.silu(x).unsqueeze(1) * self.base_weight.unsqueeze(0)
                + torch.einsum("nib,oib->noi", basis, self.coefficients))

    def forward(self, x: Tensor) -> Tensor:
        # The factorized form avoids allocating [batch, output, input] in training.
        basis = bspline_basis(x.clamp(self.low, self.high), self.knots, self.degree)
        return F.linear(F.silu(x), self.base_weight, self.bias) + torch.einsum(
            "nib,oib->no", basis, self.coefficients
        )


class GaussianEdgeLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, grid_size: int = 8,
                 low: float = -3.0, high: float = 3.0):
        super().__init__()
        if grid_size < 2 or high <= low:
            raise ValueError("Gaussian basis requires grid_size >= 2 and high > low")
        self.register_buffer("centers", torch.linspace(low, high, grid_size))
        self.width = (high - low) / (grid_size - 1)
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.coefficients = nn.Parameter(torch.empty(out_features, in_features, grid_size))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.coefficients, std=0.03 / math.sqrt(in_features))

    def basis(self, x: Tensor) -> Tensor:
        return torch.exp(-((x.unsqueeze(-1) - self.centers) / self.width).square())

    def edges(self, x: Tensor) -> Tensor:
        return (F.silu(x).unsqueeze(1) * self.base_weight.unsqueeze(0)
                + torch.einsum("nib,oib->noi", self.basis(x), self.coefficients))

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(F.silu(x), self.base_weight, self.bias) + torch.einsum(
            "nib,oib->no", self.basis(x), self.coefficients
        )


class GroupedRationalActivation(nn.Module):
    def __init__(self, features: int, groups: int = 8, numerator: int = 5,
                 denominator: int = 4):
        super().__init__()
        if groups < 1 or numerator < 1 or denominator < 1:
            raise ValueError("Rational groups and degrees must be positive")
        groups = min(groups, features)
        self.register_buffer("group_index", torch.div(
            torch.arange(features) * groups, features, rounding_mode="floor"))
        a = torch.zeros(groups, numerator + 1)
        a[:, 1] = 1
        b = torch.zeros(groups, denominator)
        b[:, 0] = 0.01  # Avoid the zero subgradient of |Q| at initialization.
        self.numerator = nn.Parameter(a)
        self.denominator = nn.Parameter(b)

    def forward(self, x: Tensor) -> Tensor:
        bounded = x.clamp(-10, 10)
        a, b = self.numerator[self.group_index], self.denominator[self.group_index]
        p = a[:, -1].expand_as(bounded)
        for coefficient in a[:, :-1].unbind(-1)[::-1]:
            p = p * bounded + coefficient
        q = b[:, -1].expand_as(bounded)
        for coefficient in b[:, :-1].unbind(-1)[::-1]:
            q = q * bounded + coefficient
        q = q * bounded
        return p / (1 + q.abs()) + (x - bounded)


class WaveletEdgeLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        shape = (out_features, in_features)
        self.base_weight = nn.Parameter(torch.empty(shape))
        self.amplitude = nn.Parameter(torch.empty(shape))
        self.shift = nn.Parameter(torch.zeros(shape))
        self.raw_scale = nn.Parameter(torch.full(shape, math.log(math.expm1(0.95))))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.amplitude, std=0.03 / math.sqrt(in_features))

    def edges(self, x: Tensor) -> Tensor:
        u = (x.unsqueeze(1) - self.shift) / (F.softplus(self.raw_scale) + 0.05)
        # Beyond |u|=30 the wavelet is numerically zero; clipping avoids inf*0.
        u2 = u.clamp(-30, 30).square()
        wavelet = (1 - u2) * torch.exp(-0.5 * u2)
        return self.base_weight * F.silu(x).unsqueeze(1) + self.amplitude * wavelet

    def forward(self, x: Tensor) -> Tensor:
        return self.edges(x).sum(-1) + self.bias


class LogitHead(nn.Module):
    def __init__(self, *layers: nn.Module):
        super().__init__()
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x).squeeze(-1)


def make_head(name: str, in_features: int, hidden: int = 32, **kwargs: Any) -> nn.Module:
    known = {"grid_size", "grid_min", "grid_max", "spline_degree", "rational_groups",
             "rational_numerator", "rational_denominator"}
    unknown = set(kwargs) - known
    if unknown:
        raise TypeError(f"Unknown head options: {sorted(unknown)}")
    if name not in HEAD_NAMES:
        raise ValueError(f"Unknown head {name!r}; expected one of {HEAD_NAMES}")
    if in_features < 1 or (hidden < 1 and name != "linear"):
        raise ValueError("in_features and hidden must be positive (linear has no hidden width)")
    grid = int(kwargs.get("grid_size", 8))
    low, high = float(kwargs.get("grid_min", -3)), float(kwargs.get("grid_max", 3))
    degree = int(kwargs.get("spline_degree", 3))
    if name == "linear":
        return LogitHead(nn.Linear(in_features, 1))
    if name in {"mlp_relu", "mlp_gelu", "mlp_bspline", "kan_rational"}:
        if name == "mlp_relu":
            activation = nn.ReLU()
        elif name == "mlp_gelu":
            activation = nn.GELU()
        elif name == "mlp_bspline":
            activation = BSplineActivation(hidden, grid, degree, low, high)
        else:
            activation = GroupedRationalActivation(
                hidden, int(kwargs.get("rational_groups", 8)),
                int(kwargs.get("rational_numerator", 5)),
                int(kwargs.get("rational_denominator", 4)))
        return LogitHead(nn.Linear(in_features, hidden), activation, nn.Linear(hidden, 1))
    if name == "kan_spline":
        def layer(i: int, o: int) -> nn.Module:
            return SplineEdgeLayer(i, o, grid, degree, low, high)
    elif name == "kan_fast":
        def layer(i: int, o: int) -> nn.Module:
            return GaussianEdgeLayer(i, o, grid, low, high)
    else:
        layer = WaveletEdgeLayer
    return LogitHead(layer(in_features, hidden), layer(hidden, 1))


def count_params(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def match_parameters(name: str, in_features: int, target: int,
                     tolerance: float = 0.05, max_hidden: int = 512,
                     **kwargs: Any) -> tuple[nn.Module, dict[str, Any]]:
    if target <= 0 or tolerance < 0 or max_hidden < 1:
        raise ValueError("Require positive target/max_hidden and nonnegative tolerance")
    with torch.random.fork_rng(devices=[]):
        counts: dict[int, int] = {}

        def parameter_count(width: int) -> int:
            if width not in counts:
                counts[width] = count_params(make_head(name, in_features, width, **kwargs))
            return counts[width]

        minimum, maximum = parameter_count(1), parameter_count(max_hidden)
        low, high = 1, max_hidden
        while low < high:
            middle = (low + high) // 2
            if parameter_count(middle) < target:
                low = middle + 1
            else:
                high = middle
        candidates = {1, max_hidden, low, max(1, low - 1)}
        width = min(candidates, key=lambda w: (abs(parameter_count(w) - target), w))
    model = make_head(name, in_features, width, **kwargs)
    params = count_params(model)
    error = abs(params - target) / target
    return model, {
        "hidden": 0 if name == "linear" else width, "params": params,
        "target_params": int(target), "relative_error": error,
        "feasible": error <= tolerance, "tolerance": tolerance,
        "min_params": minimum, "max_params": maximum,
    }


def benchmark(module: nn.Module, x: Tensor, warmup: int = 20,
              repeats: int = 100) -> dict[str, Any]:
    if warmup < 0 or repeats < 1:
        raise ValueError("Require warmup >= 0 and repeats >= 1")
    was_training = module.training
    module.eval()
    try:
        with torch.inference_mode():
            for _ in range(warmup):
                module(x)
            if x.is_cuda:
                torch.cuda.synchronize(x.device)
                events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                          for _ in range(repeats)]
                with torch.cuda.device(x.device):
                    for start, end in events:
                        start.record()
                        module(x)
                        end.record()
                    torch.cuda.synchronize(x.device)
                times = [start.elapsed_time(end) for start, end in events]
            else:
                times = []
                for _ in range(repeats):
                    start_time = time.perf_counter()
                    module(x)
                    times.append((time.perf_counter() - start_time) * 1000)
    finally:
        module.train(was_training)
    values = torch.tensor(times, dtype=torch.float64)
    return {"p50_ms": values.quantile(0.50).item(), "p95_ms": values.quantile(0.95).item(),
            "mean_ms": values.mean().item(), "batch_size": len(x),
            "repeats": repeats, "device": str(x.device)}