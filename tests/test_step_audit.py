"""Saved audit reports must distinguish fixed steps from epochs and verify exposure."""
import json

import numpy as np
import pandas as pd
import pytest

from fresco.factorial_audit import HEADS, RECIPES, audit_factorial
from fresco.research_audit import AuditInputError, _convergence
from fresco.utils import atomic_json, mark_complete


def saved_history(tmp_path, *, fixed=True):
    history = pd.DataFrame({"epoch": [0, 1, 1] if fixed else [0, 1, 2],
                            "loss": [.5, .4, .3], "val_video_auc": [.6, .8, .7],
                            "wall_s": [1., 2., 3.]})
    train = {"epochs": 30 if fixed else 3}
    if fixed:
        history["step"] = [3, 6, 7]
        history["examples_processed"] = [9, 18, 21]
        history["epoch_fraction"] = [.45, .9, 1.05]
        train.update(max_steps=7, validation_every_steps=3, batch_size=3)
    history.to_csv(tmp_path / "train_log.csv", index=False)
    return {"train": train}


def test_fixed_step_history_is_not_reported_as_three_epochs(tmp_path):
    cfg = saved_history(tmp_path)
    result = _convergence(tmp_path, "kan_spline", 0, "params", cfg, {}, None)
    assert result["schedule"] == "fixed_steps"
    assert result["validation_checks"] == 3
    assert result["optimizer_steps_observed"] == 7
    assert result["best_step"] == 6
    assert result["last_epoch_fraction"] == 1.05
    assert result["examples_processed"] == 21
    assert result["configured_max_steps"] == 7
    assert result["hit_step_limit"] is True
    for key in ["epochs_observed", "best_epoch", "window_epochs", "configured_max_epochs", "hit_epoch_limit"]:
        assert result[key] is None


def test_old_epoch_history_remains_supported(tmp_path):
    cfg = saved_history(tmp_path, fixed=False)
    result = _convergence(tmp_path, "kan_spline", 0, "params", cfg, {}, None)
    assert result["epochs_observed"] == 3
    assert result["best_epoch"] == 1
    assert result["hit_epoch_limit"] is True
    assert result["validation_checks"] == 3
    assert result["optimizer_steps_observed"] is None
    assert result["hit_step_limit"] is None


@pytest.mark.parametrize("corruption", ["missing", "decreasing", "fractional"])
def test_new_step_history_requires_valid_step_counts(tmp_path, corruption):
    cfg = saved_history(tmp_path)
    history = pd.read_csv(tmp_path / "train_log.csv")
    if corruption == "missing":
        history = history.drop(columns="step")
    elif corruption == "decreasing":
        history["step"] = [3, 2, 7]
    else:
        history["examples_processed"] = [9, 18.5, 21]
    history.to_csv(tmp_path / "train_log.csv", index=False)
    with pytest.raises(AuditInputError, match="[Ff]ixed-step"):
        _convergence(tmp_path, "kan_spline", 0, "params", cfg, {}, None)


def save_factorial(tmp_path, *, fixed=True):
    paths, metrics_files = [], []
    labels, groups = np.tile([0, 1], 6), np.repeat(np.arange(6), 2)
    for recipe in RECIPES:
        experiment = tmp_path / recipe
        experiment.mkdir()
        paths.append(experiment)
        conditions = ["clean"] if recipe == "clean" else ["clean", "jpeg70"]
        train = {"conditions": conditions, "target_params": 1585, "batch_size": 3}
        if fixed:
            train.update(max_steps=7, validation_every_steps=3)
        config = {"experiment": {"seeds": [0, 1], "source_conditions": ["clean", "jpeg70"]},
                  "features": {"source_conditions": ["clean", "jpeg70"]},
                  "train": train, "model": {"grid_size": 5}}
        atomic_json(experiment / "manifest.json", {"config": config, "source": "same-source",
                    "cache": {name: "a" * 64 for name in ["features.npy", "covariates.npy", "metadata.csv"]}})
        for cov in (0, 1):
            common = experiment / f"k16_cov{cov}" / "heldout_none"
            common.mkdir(parents=True)
            np.savez(common / "coordinates.npz", zmean=np.zeros(16), use_covariates=bool(cov))
            atomic_json(common / "fit_provenance.json", {"dataset": "ffpp", "split": "train",
                        "coordinate_fit_conditions": ["clean"], "held_out_method": None,
                        "head_training_conditions": conditions, "video_ids": ["real", "fake"]})
            for head in HEADS:
                base = common / head / "params"
                base.mkdir(parents=True)
                atomic_json(base / "hpo.json", {"selection": "FF++ seen-method validation video AUC only",
                                                "selection_seed": 0})
                mark_complete(base, "hpo", ["hpo.json"])
                for seed in (0, 1):
                    run = base / f"seed{seed}"
                    run.mkdir()
                    training = {"training_conditions": conditions,
                                "validation_selection": "FF++ clean seen-method video AUROC only"}
                    if fixed:
                        training.update(schedule="fixed_steps", optimizer_steps=7, requested_max_steps=7,
                                        examples_processed=21, batch_size=3,
                                        validation_every_steps=3, validation_steps=[3, 6, 7])
                    atomic_json(run / "metrics.json", {"run": {"head": head, "budget": "params", "k": 16,
                                "covariates": bool(cov), "held_out": None, "seed": seed},
                                "matching": {"feasible": True, "target_params": 1585}, "training": training})
                    pd.DataFrame({"dataset": "celebdfpp", "split": "test", "condition": "clean",
                                  "video_id": [f"v{i}" for i in range(len(labels))], "label": labels,
                                  "method": np.where(labels, "unseen", "real"), "group_id": groups,
                                  "probability": .1 + .8*labels, "frames": 1}).to_csv(run / "scores.csv", index=False)
                    mark_complete(run, "run", ["metrics.json", "scores.csv"])
                    metrics_files.append(run / "metrics.json")
    return paths, metrics_files


def test_factorial_accepts_and_labels_verified_equal_step_exposure(tmp_path):
    paths, _ = save_factorial(tmp_path)
    result = audit_factorial(*paths, tmp_path / "audit", resamples=20)
    assert result["status"] == "COMPUTED"
    control = result["training_exposure_control"]
    assert control["status"] == "VERIFIED"
    assert control["optimizer_exposure_matched"] is True
    assert control["examples_processed"] == 21
    assert control["validation_steps"] == [3, 6, 7]


@pytest.mark.parametrize("field,value", [
    ("optimizer_steps", 6), ("examples_processed", 20), ("validation_steps", [3, 7]),
    ("validation_steps", [3, 6, 7.0]), ("schedule", "epochs"), ("batch_size", 4),
    ("requested_max_steps", None), ("validation_every_steps", 2),
])
def test_factorial_rejects_mismatched_recorded_exposure_even_with_valid_checksums(tmp_path, field, value):
    paths, files = save_factorial(tmp_path)
    path = files[-1]
    metrics = json.loads(path.read_text())
    metrics["training"][field] = value
    atomic_json(path, metrics)
    mark_complete(path.parent, "run", ["metrics.json", "scores.csv"])
    result = audit_factorial(*paths, tmp_path / "audit", resamples=20)
    assert result["status"] == "INCONCLUSIVE"
    assert f"exposure mismatch for {field}" in result["reason"]
    assert result["training_exposure_control"]["status"] == "DECLARED_NOT_VERIFIED"


def test_factorial_can_read_legacy_but_does_not_call_its_exposure_matched(tmp_path):
    paths, _ = save_factorial(tmp_path, fixed=False)
    result = audit_factorial(*paths, tmp_path / "audit", resamples=20)
    assert result["status"] == "COMPUTED"
    control = result["training_exposure_control"]
    assert control["status"] == "NOT_MATCHED"
    assert control["optimizer_exposure_matched"] is False
    assert "optimizer exposure" in control["limitation"]
