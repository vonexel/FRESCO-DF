import copy
import numpy as np
import torch
from fresco.extensions.symbolic_model import build_clones
from fresco.heads import make_head


def test_rejected_edges_remain_exact_and_teacher_unchanged():
    torch.manual_seed(1)
    teacher = make_head("kan_spline", 3, 2, grid_size=5, spline_degree=3).eval()
    before = copy.deepcopy(teacher.state_dict())
    records = []
    for layer, module in enumerate(teacher.network):
        for o in range(module.base_weight.shape[0]):
            for i in range(module.base_weight.shape[1]):
                records.append({"layer": layer, "output": o, "input": i, "accepted": False,
                                "family": "zero", "coefficients": []})
    pruned, hybrid, info = build_clones(teacher, records)
    x = torch.from_numpy(np.random.default_rng(7).normal(size=(100, 3)).astype(np.float32))
    with torch.no_grad():
        np.testing.assert_allclose(hybrid(x).numpy(), teacher(x).numpy(), atol=1e-6)
        np.testing.assert_allclose(pruned(x).numpy(), teacher(x).numpy(), atol=1e-6)
    assert info["accepted_symbolic_edges"] == 0 and not info["full_symbolification"]
    assert all(torch.equal(value, before[key]) for key, value in teacher.state_dict().items())


def test_zero_edges_are_pruned_on_copy(monkeypatch):
    teacher = make_head("kan_spline", 3, 2, grid_size=5, spline_degree=3).eval()
    with torch.no_grad():
        for layer in teacher.network:
            layer.base_weight.zero_()
            layer.coefficients.zero_()
    records = [{"layer": layer, "output": o, "input": i, "accepted": True,
                "family": "zero", "coefficients": []}
               for layer, module in enumerate(teacher.network)
               for o in range(module.base_weight.shape[0]) for i in range(module.base_weight.shape[1])]
    pruned, symbolic, info = build_clones(teacher, records)
    x = torch.randn(25, 3)
    with torch.no_grad():
        assert torch.allclose(pruned(x), teacher(x))
        assert torch.allclose(symbolic(x), teacher(x))
    assert info["full_symbolification"] and info["pruned_edges"] == len(records)
    def forbidden(_):
        raise AssertionError("A fully symbolified clone must not evaluate original spline edges")
    for layer in symbolic.network:
        monkeypatch.setattr(layer.original, "edges", forbidden)
    with torch.no_grad():
        assert torch.allclose(symbolic(x), teacher(x))