"""Matched augmentation exposure and source-only checkpoint selection."""
import copy

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from fresco import training

CONDITIONS = ["clean", "jpeg70", "blur1", "resize0.5"]


def fixture_data():
    rows = []
    for split in ["train", "val"]:
        for label in [0, 1]:
            for frame in range(2):
                for condition in CONDITIONS:
                    rows.append({"dataset": "ffpp", "split": split, "label": label,
                                 "video_id": f"{split}/{label}", "frame_idx": frame,
                                 "method": "real" if label == 0 else "FaceSwap",
                                 "condition": condition, "scenario": "fixture",
                                 "group_id": f"{split}/{label}"})
    rows.append({"dataset": "celebdfpp", "split": "test", "label": 1,
                 "video_id": "target", "frame_idx": 0, "method": "unseen",
                 "condition": "clean", "scenario": "fixture", "group_id": "target"})
    meta = pd.DataFrame(rows)
    x = np.random.default_rng(81).normal(size=(len(meta), 3)).astype(np.float32)
    return x, meta


def config(conditions=None, **overrides):
    cfg = {"device": "cpu", "threads": 1, "conditions": conditions or ["clean"],
           "weight_decay": 0., "epochs": 1, "patience": 1, "batch_size": 3,
           "wall_seconds": 1., "max_steps": 7, "validation_every_steps": 3}
    cfg.update(overrides)
    return OmegaConf.create(cfg)


class RecordingLinear(torch.nn.Linear):
    def __init__(self):
        super().__init__(3, 1)
        self.training_batch_sizes = []

    def forward(self, x):
        if self.training:
            self.training_batch_sizes.append(len(x))
        return super().forward(x)


def test_clean_and_four_condition_recipes_have_same_steps_examples_and_validation(monkeypatch):
    x, meta = fixture_data()
    validations = []
    actual_predict = training.predict

    def record_validation(model, values, device, batch=4096):
        validations.append(values.copy())
        return actual_predict(model, values, device, batch)

    monkeypatch.setattr(training, "predict", record_validation)
    # A plateau would stop the old patience-based schedule after two checks.
    monkeypatch.setattr(training, "roc_auc_score", lambda *args: .5)
    runs = []
    for conditions in [["clean"], CONDITIONS]:
        model = RecordingLinear()
        _, _, details = training.fit_head(model, x, meta, config(conditions), .001, 0,
                                         "params", progress=False)
        assert model.training_batch_sizes == [3] * 7
        assert details["optimizer_steps"] == 7
        assert details["examples_processed"] == 21
        assert details["validation_steps"] == [3, 6, 7]
        assert details["schedule"] == "fixed_steps"
        assert details["requested_max_steps"] == 7
        assert "disabled" in details["early_stopping"]
        runs.append(details)
    assert runs[0]["training_rows"] == 4
    assert runs[1]["training_rows"] == 16
    expected = x[training.validation_mask(meta)]
    assert len(validations) == 6
    for values in validations:
        np.testing.assert_array_equal(values, expected)


def test_fixed_steps_restores_best_source_checkpoint_and_validates_final_remainder(monkeypatch):
    x, meta = fixture_data()
    aucs = iter([.7, .9, .8])
    monkeypatch.setattr(training, "roc_auc_score", lambda *args: next(aucs))
    states = []
    original_step = torch.optim.AdamW.step

    def record_step(optimizer, *args, **kwargs):
        value = original_step(optimizer, *args, **kwargs)
        states.append(copy.deepcopy(model.state_dict()))
        return value

    monkeypatch.setattr(torch.optim.AdamW, "step", record_step)
    model = torch.nn.Linear(3, 1)
    fitted, initial, details = training.fit_head(model, x, meta, config(), .001, 0,
                                                "params", progress=False)
    assert len(states) == 7
    for name, tensor in fitted.state_dict().items():
        torch.testing.assert_close(tensor, states[5][name], rtol=0, atol=0)
    assert details["validation_auc"] == .9
    assert details["validation_steps"] == [3, 6, 7]
    assert any(not torch.equal(initial[name], value) for name, value in fitted.state_dict().items())


@pytest.mark.parametrize("overrides,reason", [
    ({"max_steps": None}, "specified together"),
    ({"validation_every_steps": None}, "specified together"),
    ({"max_steps": 0}, "positive integer"),
    ({"max_steps": True}, "positive integer"),
    ({"max_steps": 2.5}, "positive integer"),
    ({"validation_every_steps": -1}, "positive integer"),
    ({"validation_every_steps": "3"}, "positive integer"),
    ({"batch_size": 0}, "batch_size"),
])
def test_partial_or_invalid_schedule_fails_before_training(overrides, reason):
    x, meta = fixture_data()
    with pytest.raises(ValueError, match=reason):
        training.fit_head(torch.nn.Linear(3, 1), x, meta, config(**overrides), .001, 0,
                          "params", progress=False)


def test_legacy_epoch_schedule_retains_partial_batches_and_patience(monkeypatch):
    x, meta = fixture_data()
    cfg = config(max_steps=None, validation_every_steps=None, epochs=10)
    model = RecordingLinear()
    monkeypatch.setattr(training, "roc_auc_score", lambda *args: .5)
    _, _, details = training.fit_head(model, x, meta, cfg, .001, 0, "params", progress=False)
    assert details["epochs"] == 2  # first best, then one stale epoch
    assert model.training_batch_sizes == [3, 1, 3, 1]
    assert details["optimizer_steps"] == 4
    assert details["examples_processed"] == 8
    assert details["validation_steps"] == [2, 4]
    assert details["schedule"] == "epochs"
    assert details["requested_max_steps"] is None


def test_wall_clock_series_does_not_apply_parameter_step_limit(monkeypatch):
    x, meta = fixture_data()
    ticks = iter(np.arange(1000) * .01)
    monkeypatch.setattr(training.time, "perf_counter", lambda: float(next(ticks)))
    _, _, details = training.fit_head(torch.nn.Linear(3, 1), x, meta,
                                     config(max_steps=1, validation_every_steps=1,
                                            wall_seconds=.045),
                                     .001, 0, "time", progress=False)
    assert details["optimizer_steps"] > 1
    assert details["schedule"] == "wall_clock"
    assert details["requested_max_steps"] is None
    assert details["requested_wall_s"] == .045


def test_response_presets_declare_identical_parameter_step_schedules():
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        original = compose(config_name="config", overrides=["experiment=main"])
        clean = compose(config_name="config", overrides=["experiment=response_clean"])
        augmented = compose(config_name="config", overrides=["experiment=response_augmented"])
    assert original.train.max_steps is None
    assert original.train.validation_every_steps is None
    assert clean.train.max_steps == augmented.train.max_steps == 3000
    assert clean.train.validation_every_steps == augmented.train.validation_every_steps == 100
