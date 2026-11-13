import copy

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from scipy.special import expit

from fresco.evaluation import evaluate_run, fit_video_temperature
from fresco.heads import make_head
from fresco.synthetic import make_cache
from fresco.training import Coordinates, video_predictions


def fixture_data(tmp_path, covariates=True):
    cache = make_cache(tmp_path / "cache")
    meta = pd.read_csv(cache / "metadata.csv")
    features = np.load(cache / "features.npy")
    cov = np.load(cache / "covariates.npy")
    coords = Coordinates().fit(features, cov, meta, 4, covariates)
    X = coords.transform(features, cov)
    cfg = OmegaConf.create({"train": {"device": "cpu"}, "experiment": {"synthetic": True},
        "eval": {"interpretation_samples": 8, "grid_points": 5, "benchmark_repeats": 3,
                 "benchmark_warmup": 1, "symbolic_samples": 256}})
    return meta, X, coords, cfg


def test_temperature_objective_uses_mean_frame_probabilities():
    meta = pd.DataFrame({"dataset": ["ffpp"]*8, "video_id": np.repeat(["a", "b", "c", "d"], 2),
                         "label": np.repeat([0, 0, 1, 1], 2)})
    logits = np.array([-10, 8, -2, -1, 1, 2, -8, 10.0])
    temperature, info = fit_video_temperature(meta, logits)
    p = expit(logits / temperature).reshape(-1, 2).mean(1)
    y = np.array([0, 0, 1, 1])
    expected = -np.mean(y*np.log(p) + (1-y)*np.log1p(-p))
    assert info["validation_video_nll_after"] == pytest.approx(expected)
    wrong = expit(logits.reshape(-1, 2).mean(1)/temperature)
    assert not np.allclose(p, wrong)


def test_full_evaluation_artifacts_real_controls_and_exact_response_gap(tmp_path):
    meta, X, coords, cfg = fixture_data(tmp_path)
    model = make_head("kan_spline", X.shape[1], hidden=3)
    initial = copy.deepcopy(model.state_dict())
    result = evaluate_run(model, initial, X, meta, coords, cfg, tmp_path / "eval")
    out = tmp_path / "eval"
    for filename in ["scores.csv", "frame_scores.csv", "calibration.csv", "metrics.csv", "temperature.json",
                     "pd_curves.csv", "intervention.csv", "openset_scores.csv", "symbolic_surrogate.json"]:
        assert (out / filename).is_file()
    frames = pd.read_csv(out / "frame_scores.csv")
    videos = pd.read_csv(out / "scores.csv")
    assert set(frames.split) == {"test"}
    assert {"train", "val", "test"}.issubset(set(videos.split))
    expected = video_predictions(meta, model_predict(model, X), result["temperature"]["temperature"])
    np.testing.assert_allclose(videos.calibrated, expected.calibrated, rtol=1e-6)
    for row in result["eval"]:
        assert row["n_real"] > 0 and row["n_fake"] > 0
        assert row["auc"] is not None
        if row["level"] == "frame":
            assert "auc_ci_delong_video" not in row
    interventions = pd.read_csv(out / "intervention.csv")
    np.testing.assert_allclose(interventions.total, interventions.covariate + interventions.embedding
                               + interventions.interaction, atol=1e-6)
    pooled = np.sqrt(interventions.error_sse.sum()/interventions.total_sse.sum())
    assert result["interp"]["intervention"]["pooled_paired_response_agreement_gap"] == pytest.approx(pooled)
    assert interventions.frame_count.eq(4).all()
    assert result["openset"]["plasticity"]["full_canevaro_pipeline_reproduced"] is False
    assert all("confounded" in row["confounding"] for row in result["openset"]["metrics"])
    assert result["interp"]["symbolic"]["hypothesis_pass_inferred"] is False
    assert result["compute"]["latency_batch256"]["batch_size"] == 256


def model_predict(model, X):
    import torch
    with torch.inference_mode():
        return model(torch.from_numpy(X)).numpy()


def test_target_changes_cannot_change_calibration_or_validation_threshold(tmp_path):
    meta, X, coords, cfg = fixture_data(tmp_path)
    model = make_head("linear", X.shape[1])
    initial = copy.deepcopy(model.state_dict())
    a = evaluate_run(model, initial, X, meta, coords, cfg, tmp_path / "a")
    changed = X.copy()
    changed[(meta.split == "test").to_numpy()] *= -17
    b = evaluate_run(model, initial, changed, meta, coords, cfg, tmp_path / "b")
    assert a["temperature"] == b["temperature"]
    assert a["interp"]["pd"] == b["interp"]["pd"]
    assert a["interp"]["symbolic"]["degree"] == b["interp"]["symbolic"]["degree"]
    assert a["interp"]["symbolic"]["val_rmse"] == b["interp"]["symbolic"]["val_rmse"]


def test_covariate_ablation_and_lomo_novelty_population(tmp_path):
    meta, X, coords, cfg = fixture_data(tmp_path, covariates=False)
    model = make_head("mlp_relu", X.shape[1], hidden=5)
    result = evaluate_run(model, model.state_dict(), X, meta, coords, cfg, tmp_path / "ablation", held_out="Deepfakes")
    assert result["interp"]["pd"]["status"] == "not_applicable"
    assert result["interp"]["intervention"]["status"] == "not_applicable"
    assert result["openset"]["plasticity"]["status"] == "unsupported"
    lomo = [r for r in result["openset"]["metrics"] if r["comparison"] == "FF++ seen fake vs held-out fake"]
    assert lomo and all(r["n_unknown"] == 3 and r["n_id"] == 9 for r in lomo)
    assert (tmp_path / "ablation" / "intervention.csv").is_file()
