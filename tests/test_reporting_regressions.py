"""Regression coverage for the confirmatory curve-stability requirement."""
import json

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from fresco.evaluation import COVARIATE_NAMES
from fresco.reporting import _curves, agreement_test, hypothesis_tests


def primary_fixture(tmp_path):
    runs = []
    for head in ["kan_spline", "mlp_gelu"]:
        for seed in range(10):
            relative = f"{head}/seed{seed}"
            directory = tmp_path / relative
            directory.mkdir(parents=True)
            runs.append({"head": head, "budget": "params", "k": 16, "covariates": True,
                         "held_out": None, "seed": seed, "run_dir": relative,
                         "budget_feasible": True})
            pd.DataFrame([
                {"feature": feature, "grid_type": "shared_standardized", "grid": grid,
                 "response": (index + 1) * grid * (1 + seed / 100)}
                for index, feature in enumerate(COVARIATE_NAMES)
                for grid in [-2., 0., 2.]
            ]).to_csv(directory / "pd_curves.csv", index=False)
            pd.DataFrame({
                "dataset": ["celebdfpp"] * 10, "video_id": [f"v{i}" for i in range(10)],
                "group_id": [f"g{i}" for i in range(10)],
                "error_sse": [.01 if head == "kan_spline" else .25] * 10,
                "total_sse": [1.] * 10, "frame_count": [1] * 10,
            }).to_csv(directory / "intervention.csv", index=False)
            pd.DataFrame({
                "dataset": ["celebdfpp"] * 10, "condition": ["clean"] * 10,
                "video_id": [f"v{i}" for i in range(10)], "group_id": [f"g{i}" for i in range(10)],
                "split": ["test"] * 10, "label": [i % 2 for i in range(10)],
                "probability": [.1 if i % 2 == 0 else .9 for i in range(10)],
            }).to_csv(directory / "scores.csv", index=False)
    cfg = OmegaConf.create({"experiment": {"synthetic": False, "seeds": list(range(10))},
                            "eval": {"bootstrap": 100, "margin": .01,
                                     "min_seeds_confirmatory": 10, "grid_points": 3}})
    return runs, cfg


def test_complete_stable_curves_can_support_the_primary_contrast(tmp_path):
    runs, cfg = primary_fixture(tmp_path)
    result = agreement_test(tmp_path, runs, cfg)
    coverage = result["curve_stability_coverage"]
    assert result["status"] == "SUPPORTED_CONDITIONAL_ON_SEEDS"
    assert result["median_curve_stability"] == pytest.approx(1)
    assert coverage["complete"]
    assert coverage["complete_covariates"] == 4
    assert coverage["defined_pairs"] == coverage["expected_pairs"] == 180


def test_secondary_equivalence_requires_every_declared_seed(tmp_path, monkeypatch):
    runs = [{"head": head, "budget": "params", "k": 8, "covariates": True,
             "held_out": None, "seed": seed, "val_auc": .8}
            for head in ["mlp_gelu", "kan_spline"] for seed in range(10)]
    cfg = OmegaConf.create({"experiment": {"synthetic": False, "seeds": list(range(20))},
                            "eval": {"min_seeds_confirmatory": 10}})
    monkeypatch.setattr("fresco.reporting._aligned_scores",
                        lambda *args: pytest.fail("Do not infer equivalence from an incomplete declared series"))
    result = hypothesis_tests(tmp_path, runs, cfg)
    assert len(result["H2a"]) == 1
    assert result["H2a"][0]["status"] == "INCONCLUSIVE"
    assert result["H2a"][0]["reason"] == "Incomplete or duplicate declared seed sets"


def test_one_defined_pair_cannot_hide_constant_seeds_and_covariates(tmp_path):
    runs, cfg = primary_fixture(tmp_path)
    for run in runs:
        if run["head"] != "kan_spline":
            continue
        path = tmp_path / run["run_dir"] / "pd_curves.csv"
        frame = pd.read_csv(path)
        frame.loc[(frame.feature != COVARIATE_NAMES[0]) | (run["seed"] >= 2), "response"] = 0
        frame.to_csv(path, index=False)
    _, stability = _curves(tmp_path, runs, cfg)
    kan = stability[stability["head"] == "kan_spline"]
    assert kan.pairs.sum() == 1
    assert kan.expected_pairs.sum() == 180
    assert kan.median_seed_correlation.notna().sum() == 1
    assert not kan.coverage_complete.any()
    # Even a previously saved optimistic aggregate cannot override actual curves.
    pd.DataFrame({"median_seed_correlation": [1]}).to_csv(tmp_path / "stability.csv", index=False)
    result = agreement_test(tmp_path, runs, cfg)
    assert result["status"] == "INCONCLUSIVE"
    assert result["median_curve_stability"] == pytest.approx(1)
    assert result["curve_stability_coverage"]["defined_pairs"] == 1
    assert result["curve_stability_coverage"]["complete_covariates"] == 0
    assert "constant curve" in json.dumps(result["curve_stability_coverage"])


@pytest.mark.parametrize("defect,reason", [
    ("missing_covariate", "missing or nonfinite curve"),
    ("missing_point", "incomplete or different shared grid"),
    ("different_grid", "incomplete or different shared grid"),
    ("nonfinite_response", "missing or nonfinite curve"),
    ("nonfinite_grid", "missing or nonfinite curve"),
    ("duplicate_point", "duplicate grid points"),
    ("missing_file", "unavailable curves"),
    ("missing_run", "expected one run, found 0"),
])
def test_incomplete_curve_coverage_is_inconclusive(tmp_path, defect, reason):
    runs, cfg = primary_fixture(tmp_path)
    run = runs[0]
    path = tmp_path / run["run_dir"] / "pd_curves.csv"
    frame = pd.read_csv(path)
    first_feature = frame.feature == COVARIATE_NAMES[0]
    if defect == "missing_covariate":
        frame = frame.loc[~first_feature]
    elif defect == "missing_point":
        frame = frame.drop(frame.index[first_feature][0])
    elif defect == "different_grid":
        frame.loc[first_feature, "grid"] += .1
    elif defect == "nonfinite_response":
        frame.loc[frame.index[first_feature][0], "response"] = np.nan
    elif defect == "nonfinite_grid":
        frame.loc[frame.index[first_feature][0], "grid"] = np.inf
    elif defect == "duplicate_point":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif defect == "missing_file":
        path.unlink()
    elif defect == "missing_run":
        # Identical incomplete model seed sets used to pass the pairing check.
        runs = [r for r in runs if r["seed"] != 0]
    if defect not in {"missing_file", "missing_run"}:
        frame.to_csv(path, index=False)
    result = agreement_test(tmp_path, runs, cfg)
    assert result["status"] == "INCONCLUSIVE"
    assert not result["curve_stability_coverage"]["complete"]
    assert reason in json.dumps(result["curve_stability_coverage"])
