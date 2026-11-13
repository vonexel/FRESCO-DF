import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from fresco.controlled import TASKS, generate_data, run_controlled, target_function


@pytest.mark.parametrize("task", TASKS)
def test_known_function_derivatives_match_central_finite_difference(task):
    generator = torch.Generator().manual_seed(33)
    x = (.1 + .7*torch.rand(11, 8, generator=generator, dtype=torch.float64)).requires_grad_()
    analytical = torch.autograd.grad(target_function(x, task).sum(), x)[0]
    finite = torch.zeros_like(x)
    for j in range(8):
        perturbation = torch.zeros_like(x)
        perturbation[:, j] = 1e-6
        finite[:, j] = (target_function(x.detach()+perturbation, task)
                        -target_function(x.detach()-perturbation, task))/(2e-6)
    torch.testing.assert_close(analytical, finite, atol=2e-7, rtol=2e-6)


def test_data_are_repeatable_and_noisy_test_retains_clean_truth():
    a = generate_data("noisy_additive", (128, 64, 128))
    b = generate_data("noisy_additive", (128, 64, 128))
    for split in a:
        for first, second in zip(a[split], b[split]):
            torch.testing.assert_close(first, second)
        smooth = generate_data("smooth_additive", (128, 64, 128))
        torch.testing.assert_close(a[split][0], smooth[split][0])
    x, observed, clean = a["test"]
    torch.testing.assert_close(clean, target_function(x, "smooth_additive"))
    assert float((observed-clean).std()) == pytest.approx(.1, abs=.025)
    assert not torch.equal(a["train"][0], a["test"][0])


def test_training_reduces_error_and_run_resumes_verified_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr("fresco.controlled.source_fingerprint", lambda: "fixed-test-source")
    cfg = OmegaConf.create({"paths": {"output_dir": str(tmp_path)},
        "experiment": {"name": "functions", "seeds": [0], "heads": ["mlp_gelu"], "budgets": ["params"],
                       "tasks": ["smooth_additive"], "train_points": 128, "val_points": 64, "test_points": 128},
        "train": {"device": "cpu", "threads": 2, "target_hidden": 16, "lrs": [.01],
                  "epochs": 35, "patience": 20, "batch_size": 64, "weight_decay": .0001, "wall_seconds": .1}})
    directory = run_controlled(cfg)
    records = json.loads((directory/"metrics.json").read_text(encoding="utf-8"))
    assert len(records) == 1
    assert records[0]["train_mse"] < records[0]["initial_train_mse"]*.8
    assert np.isfinite(records[0]["test_gradient_rmse"])
    for name in ["comparison.svg", "comparison.csv", "protocol.json"]:
        assert (directory/name).is_file()
    run = directory/"runs"/"smooth_additive"/"mlp_gelu"/"params"/"seed0"
    timestamp = (run/"model.pt").stat().st_mtime_ns
    assert (run/"DONE.json").is_file()
    assert run_controlled(cfg) == directory
    assert (run/"model.pt").stat().st_mtime_ns == timestamp
