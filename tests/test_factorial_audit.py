import json

import numpy as np
import pandas as pd
import pytest

from fresco.factorial_audit import HEADS, RECIPES, audit_factorial, factorial_auc_uncertainty
from fresco.utils import atomic_json, digest_file, mark_complete


def factorial_scores():
    y = np.tile([0, 1], 6)
    scores = np.broadcast_to(y, (2, 2, 2, 2, len(y))).astype(float).copy()
    scores[0, 0, 1] = 1-y  # Covariates harm only clean-trained KAN.
    return y, scores, np.repeat(np.arange(6), 2)


def test_factorial_interactions_have_correct_direction_and_shared_draws():
    y, scores, groups = factorial_scores()
    result = factorial_auc_uncertainty(y, scores, groups, resamples=100)
    effects = {row["contrast"]: row for row in result["contrasts"]}
    clean = effects["clean: architecture x covariates interaction"]
    assert clean["difference"] == pytest.approx(-1)
    assert clean["crossed_seeds_clusters"]["ci95"] == pytest.approx([-1, -1])
    assert effects["augmented: architecture x covariates interaction"]["difference"] == pytest.approx(0)
    change = effects["augmentation change in architecture x covariates interaction"]
    assert change["difference"] == pytest.approx(1)
    assert change["conditional_clusters"]["ci95"] == pytest.approx([1, 1])


def test_factorial_identical_cells_have_zero_interactions_under_seed_variation():
    y, _, groups = factorial_scores()
    scores = np.random.default_rng(5).random((3, len(y)))
    cells = np.broadcast_to(scores, (2, 2, 2, *scores.shape)).copy()
    result = factorial_auc_uncertainty(y, cells, groups, resamples=100)
    assert all(row["difference"] == pytest.approx(0) for row in result["contrasts"])
    assert all(row["crossed_seeds_clusters"]["ci95"] == pytest.approx([0, 0])
               for row in result["contrasts"])
    with pytest.raises(ValueError, match="binary"):
        factorial_auc_uncertainty(y, cells[..., :-1], groups)


def save_factorial(tmp_path):
    y, values, groups = factorial_scores()
    paths = []
    for r, recipe in enumerate(RECIPES):
        experiment = tmp_path / recipe
        experiment.mkdir()
        paths.append(experiment)
        conditions = ["clean"] if r == 0 else ["clean", "jpeg70"]
        config = {"experiment": {"seeds": [0, 1], "source_conditions": ["clean", "jpeg70"]},
                  "features": {"source_conditions": ["clean", "jpeg70"]},
                  "train": {"conditions": conditions, "target_params": 1585}, "model": {"grid_size": 5}}
        atomic_json(experiment / "manifest.json", {"config": config, "source": "same-source",
            "cache": {name: "a"*64 for name in ["features.npy", "covariates.npy", "metadata.csv"]}})
        for h, head in enumerate(HEADS):
            for cov in (0, 1):
                common = experiment / f"k16_cov{cov}" / "heldout_none"
                base = common / head / "params"
                base.mkdir(parents=True)
                np.savez(common / "coordinates.npz", zmean=np.zeros(16), use_covariates=bool(cov))
                atomic_json(common / "fit_provenance.json", {"dataset": "ffpp", "split": "train",
                    "coordinate_fit_conditions": ["clean"], "held_out_method": None,
                    "head_training_conditions": conditions, "video_ids": ["source-a", "source-b"]})
                atomic_json(base / "hpo.json", {"selection": "FF++ seen-method validation video AUC only",
                                                "selection_seed": 0})
                mark_complete(base, "hpo", ["hpo.json"])
                for seed in (0, 1):
                    run = base / f"seed{seed}"
                    run.mkdir()
                    atomic_json(run / "metrics.json", {"run": {"head": head, "budget": "params", "k": 16,
                        "covariates": bool(cov), "held_out": None, "seed": seed},
                        "matching": {"feasible": True, "target_params": 1585},
                        "training": {"training_conditions": conditions,
                                     "validation_selection": "FF++ clean seen-method video AUROC only"}})
                    pd.DataFrame({"dataset": "celebdfpp", "split": "test", "condition": "clean",
                        "video_id": [f"v{i}" for i in range(len(y))], "label": y,
                        "method": np.where(y, "unknown", "real"), "group_id": groups,
                        "probability": values[r, h, cov, seed], "frames": 1}).to_csv(run / "scores.csv", index=False)
                    mark_complete(run, "run", ["metrics.json", "scores.csv"])
    return paths


def test_factorial_saved_runs_are_checked_and_originals_preserved(tmp_path):
    paths = save_factorial(tmp_path)
    before = {str(p): digest_file(p) for d in paths for p in d.rglob("*") if p.is_file()}
    result = audit_factorial(*paths, tmp_path / "audit", resamples=30)
    assert result["status"] == "COMPUTED"
    assert len(result["cells"]) == 8 and len(result["contrasts"]) == 11
    assert before == {str(p): digest_file(p) for d in paths for p in d.rglob("*") if p.is_file()}
    assert (tmp_path / "audit" / "contrasts.csv").is_file()
    np.savez(paths[1] / "k16_cov0" / "heldout_none" / "coordinates.npz",
             zmean=np.ones(16), use_covariates=False)
    changed = audit_factorial(*paths, tmp_path / "bad", resamples=30)
    assert changed["status"] == "INCONCLUSIVE"
    assert "coordinates differ" in changed["reason"]
    assert json.loads((tmp_path / "bad" / "factorial_audit.json").read_text())["status"] == "INCONCLUSIVE"
