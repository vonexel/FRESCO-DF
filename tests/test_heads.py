import copy

import numpy as np
import pytest
import torch
from torch import nn

from fresco.heads import (
    HEAD_NAMES, BSplineActivation, SplineEdgeLayer, benchmark, bspline_basis,
    count_params, make_head, match_parameters, open_uniform_knots,
)
from fresco.interpret import (
    curve_correlation, fit_pd_polynomial, intervention_decomposition,
    plasticity_score, predict_pd_polynomial, response_curves,
)


@pytest.mark.parametrize("name", HEAD_NAMES)
def test_heads_single_logit_and_finite_parameter_and_input_gradients(name):
    torch.manual_seed(11)
    model = make_head(name, in_features=7, hidden=5)
    x = torch.randn(9, 7, requires_grad=True)
    result = model(x)
    assert result.shape == (9,)
    result.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_spline_partition_of_unity_endpoints_and_identity_linear_tails():
    x = torch.linspace(-3, 3, 71, dtype=torch.float64, requires_grad=True)
    basis = bspline_basis(x, open_uniform_knots(), 3)
    torch.testing.assert_close(basis.sum(-1), torch.ones_like(x))
    assert torch.all(basis >= 0)
    assert basis[0, 0] == 1 and basis[-1, -1] == 1
    activation = BSplineActivation(1).double()
    far = torch.tensor([[-8.0], [-2.1], [0.4], [2.2], [9.0]], dtype=torch.float64,
                       requires_grad=True)
    torch.testing.assert_close(activation(far), far, atol=1e-6, rtol=1e-6)
    activation(far).sum().backward()
    assert torch.isfinite(far.grad).all()
    torch.testing.assert_close(far.grad, torch.ones_like(far), atol=1e-6, rtol=1e-6)


def test_edge_values_sum_to_layer_output():
    model = SplineEdgeLayer(5, 3)
    x = torch.randn(4, 5)
    assert model.edges(x).shape == (4, 3, 5)
    torch.testing.assert_close(model(x), model.edges(x).sum(-1) + model.bias)


def test_parameter_matching_exposes_infeasible_bounds_and_preserves_rng_search():
    torch.manual_seed(13)
    fitted, metadata = match_parameters("mlp_relu", 7, target=91)
    assert metadata["params"] == count_params(fitted) == 91
    assert metadata["hidden"] == 10 and metadata["feasible"]
    torch.manual_seed(13)
    direct = make_head("mlp_relu", 7, hidden=10)
    for a, b in zip(fitted.parameters(), direct.parameters()):
        torch.testing.assert_close(a, b)
    _, low = match_parameters("kan_spline", 7, target=2)
    assert not low["feasible"] and low["hidden"] == 1
    _, high = match_parameters("mlp_relu", 7, target=999999, max_hidden=3)
    assert not high["feasible"] and high["hidden"] == 3
    _, linear = match_parameters("linear", 7, target=91)
    assert linear["hidden"] == 0 and not linear["feasible"]


def test_all_parameter_tensors_counted_and_typo_rejected():
    model = make_head("mlp_bspline", 7, hidden=5)
    next(model.parameters()).requires_grad_(False)
    assert count_params(model) == sum(p.numel() for p in model.parameters())
    with pytest.raises(TypeError):
        make_head("linear", 7, grid_szie=10)


class Interaction(nn.Module):
    def forward(self, x):
        return 2 * x[:, 0] + 3 * x[:, 1] + x[:, 0] * x[:, 1]


def test_intervention_exact_cross_block_interaction():
    clean = torch.tensor([[1.0, 2.0], [2.0, 3.0]])
    changed = clean + torch.tensor([[3.0, 4.0], [-1.0, 2.0]])
    model = Interaction().train()
    result = intervention_decomposition(model, clean, changed, 1)
    np.testing.assert_allclose(result["total"], result["embedding"] + result["covariate"]
                               + result["interaction"])
    np.testing.assert_allclose(result["interaction"], [12, -2])
    assert model.training


def test_partial_dependence_and_polynomial_fidelity_are_prediction_level():
    model = Interaction().train()
    x = torch.tensor([[1.0, 2.0], [2.0, 3.0]])
    grid = torch.linspace(-2, 2, 11)
    curves = response_curves(model, x, [1], grid, batch_size=1)
    expected = 2 * x[:, 0].mean() + (3 + x[:, 0].mean()) * grid
    np.testing.assert_allclose(curves["curves"]["1"]["raw_mean"], expected.numpy(), atol=1e-6)
    assert model.training
    tx, vx = np.linspace(-2, 2, 21), np.linspace(-1.9, 1.9, 20)
    fit = fit_pd_polynomial(tx, tx ** 2 + 2 * tx + 1, vx, vx ** 2 + 2 * vx + 1)
    assert fit["val_rmse"] < 1e-10
    np.testing.assert_allclose(predict_pd_polynomial(fit, vx), vx ** 2 + 2 * vx + 1,
                               atol=1e-10)
    assert curve_correlation(np.ones(10), np.arange(10)) is None


def test_cpu_benchmark_restores_mode_and_reports_latency():
    model = make_head("linear", 3).train()
    result = benchmark(model, torch.zeros(4, 3), warmup=1, repeats=4)
    assert model.training and result["p95_ms"] >= result["p50_ms"] > 0
    assert result["batch_size"] == 4 and result["device"] == "cpu"


def test_plasticity_has_zero_reference_and_rejects_invented_mlp_edges():
    model = make_head("kan_spline", 3, hidden=4)
    initial = copy.deepcopy(model)
    x = torch.randn(5, 3)
    np.testing.assert_array_equal(plasticity_score(model, initial, x), np.zeros(5))
    with torch.no_grad():
        model.network[0].base_weight.add_(1)
    assert np.all(plasticity_score(model, initial, x) > 0)
    with pytest.raises(ValueError, match="explicit first-layer"):
        plasticity_score(make_head("mlp_relu", 3), make_head("mlp_relu", 3), x)
