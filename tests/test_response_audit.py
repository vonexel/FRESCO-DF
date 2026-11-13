from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from fresco.heads import make_head
from fresco.response_audit import (audit_responses, curve_delta, source_curve_grids,
                                   summarize_curve_response)


def fixture():
    records = []
    for split in ["train", "val"]:
        for label in [0, 1]:
            records.append(dict(dataset="ffpp", split=split, condition="clean", frame_idx=0,
                label=label, method="real" if label == 0 else "Deepfakes",
                video_id=f"{split}{label}", group_id=f"{split}{label}"))
    for dataset in ["ffpp", "celebdfpp"]:
        for condition in ["clean", "jpeg70"]:
            records.append(dict(dataset=dataset, split="test", condition=condition, frame_idx=0,
                label=1, method="Deepfakes", video_id=dataset+"test", group_id=dataset+"group"))
    meta = pd.DataFrame(records)
    X = np.zeros((len(meta), 5), dtype=np.float32)
    X[0, 1:], X[1, 1:] = -1, 1
    X[2, 0], X[3, 0] = -1, 1
    X[meta.condition == "jpeg70", 1:] = .5
    coords = SimpleNamespace(zmean=np.zeros(1), use_covariates=True,
        clow=np.full(4, -.1), chigh=np.full(4, .1), cmean=np.zeros(4), cstd=np.ones(4))
    cfg = OmegaConf.create({"train": {"device": "cpu"}, "eval": {
        "interpretation_samples": 8, "grid_points": 11, "audit_batch_size": 3,
        "spline_support_samples": 2}})
    return meta, X, coords, cfg


class Additive(torch.nn.Module):
    def forward(self, X):
        return X[:, 0]+X[:, 1:].sum(1)


class Interacting(torch.nn.Module):
    def forward(self, X):
        return X[:, 0]*X[:, 1]


def test_curve_prediction_recovers_additive_response_and_preserves_mode(tmp_path):
    meta, X, coords, cfg = fixture()
    model = Additive().train()
    result = audit_responses(model, X, meta, coords, cfg, tmp_path)
    assert model.training
    assert result["curve_response"]["status"] == "computed"
    assert result["curve_response"]["gap"] < 1e-6
    assert result["curve_response"]["coverage"] == 1
    assert result["curve_response"]["sign_agreement"] == 1
    for name in ["curve_predictor.csv", "curve_intervention.csv", "response_support.csv", "spline_support.csv"]:
        assert (tmp_path/name).is_file()
    assert not (tmp_path/"pd_curves.csv").exists()


def test_correct_pd_curves_do_not_explain_an_interacting_target(tmp_path):
    meta, X, coords, cfg = fixture()
    X[meta.split == "test", 0] = 2
    result = audit_responses(Interacting(), X, meta, coords, cfg, tmp_path)
    # Mean source validation z is zero: the source PD of c is exactly flat.
    assert result["curve_response"]["gap"] == pytest.approx(1.)
    assert result["curve_response"]["sign_agreement"] == 0


def test_target_extremes_never_extend_grid_or_source_curves(tmp_path):
    meta, X, coords, cfg = fixture()
    original = source_curve_grids(X, meta, coords, 9)
    changed = X.copy()
    changed[meta.dataset == "celebdfpp", 1:] = 1e6
    # Endpoints must actually differ. Equal outside-grid coordinates cancel
    # exactly and are not an extrapolation request for a delta predictor.
    changed[(meta.dataset == "celebdfpp") & (meta.condition == "jpeg70"), 1:] += 1
    actual = source_curve_grids(changed, meta, coords, 9)
    np.testing.assert_array_equal(original, actual)
    a = audit_responses(Additive(), X, meta, coords, cfg, tmp_path/"a")
    b = audit_responses(Additive(), changed, meta, coords, cfg, tmp_path/"b")
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path/"a"/"curve_predictor.csv"),
                                 pd.read_csv(tmp_path/"b"/"curve_predictor.csv"))
    assert a["curve_response"]["coverage"] == 1
    assert b["curve_response"]["coverage"] == .5
    assert b["curve_response"]["status"] == "partial_coverage"
    target = next(d for d in b["curve_response"]["by_dataset"] if d["dataset"] == "celebdfpp")
    assert target["status"] == "unavailable" and target["gap"] is None


def test_grid_can_extend_using_seen_source_degradations_but_not_heldout():
    meta, X, coords, _ = fixture()
    extra = meta.iloc[[1]].copy()
    extra["condition"] = "blur2"
    meta = pd.concat([meta, extra], ignore_index=True)
    X = np.vstack([X, np.array([0, 9, 9, 9, 9])])
    assert source_curve_grids(X, meta, coords)[0][-1] == 9
    # Keep a real training row when Deepfakes is held out.
    assert source_curve_grids(X, meta, coords, held_out="Deepfakes")[0][-1] < 0


def test_unavailable_interpolation_does_not_clip_to_endpoint():
    prediction, covered = curve_delta(np.array([[0.], [0.]]), np.array([[.5], [9.]]),
                                     [np.array([-1., 0., 1.])], [np.array([-2., 0., 2.])])
    assert prediction[0] == 1
    assert np.isnan(prediction[1])
    assert covered.tolist() == [True, False]


def test_unchanged_outside_grid_face_size_does_not_reject_a_pair(tmp_path):
    meta, X, coords, cfg = fixture()
    # Face size is unchanged by the physical image probes. Its marginal value
    # may lie outside source quantiles without requiring r(face)-r(face).
    X[meta.split == "test", 3] = 9.
    result = audit_responses(Additive(), X, meta, coords, cfg, tmp_path)
    assert result["curve_response"]["coverage"] == 1.
    assert result["curve_response"]["gap"] < 1e-6
    assert result["curve_predictor"]["coverage_rule_version"] == 2
    assert result["curve_response"]["zero_change_reference_gap"] == 1.
    support = pd.read_csv(tmp_path/"response_support.csv")
    face = support.loc[(support.split == "test") & (support.feature == "face_size")]
    assert face.outside_fraction.eq(1.).all()


def test_exact_cancellation_does_not_treat_small_changes_as_unchanged():
    clean = np.array([[9.], [9.]], dtype=np.float64)
    changed = np.array([[9.], [9.+1e-12]], dtype=np.float64)
    prediction, covered = curve_delta(clean, changed, [np.array([-1., 0., 1.])],
                                     [np.array([-2., 0., 2.])])
    assert covered.tolist() == [True, False]
    assert prediction[0] == 0.
    assert np.isnan(prediction[1])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_equal_endpoints_never_count_as_covered(value):
    prediction, covered = curve_delta(np.array([[value]]), np.array([[value]]),
                                     [np.array([-1., 1.])], [np.array([-1., 1.])])
    assert not covered[0]
    assert np.isnan(prediction[0])


def test_constant_outside_grid_covariates_with_embedding_change_have_unit_gap(tmp_path):
    meta, X, coords, cfg = fixture()
    X[meta.split == "test", 1:] = 9.
    X[(meta.split == "test") & (meta.condition == "jpeg70"), 0] = 2.
    result = audit_responses(Additive(), X, meta, coords, cfg, tmp_path)
    measured = result["curve_response"]
    assert measured["coverage"] == 1.
    assert measured["gap"] == pytest.approx(1.)
    assert measured["zero_change_reference_gap"] == 1.
    assert measured["sign_agreement"] == 0.
    assert "not directly comparable" in measured["population_warning"]


def test_actual_spline_grid_support_is_separate_from_covariate_quantiles(tmp_path):
    meta, X, coords, cfg = fixture()
    model = make_head("kan_spline", 5, hidden=2)
    first = model.network[0]
    with torch.no_grad():
        first.base_weight.zero_()
        first.coefficients.zero_()
        first.bias.fill_(7.)
    audit_responses(model, X, meta, coords, cfg, tmp_path)
    support = pd.read_csv(tmp_path/"response_support.csv")
    grid = pd.read_csv(tmp_path/"spline_support.csv")
    assert support.loc[(support.condition == "jpeg70") & (support.feature == "all_covariates"),
                       "outside_fraction"].eq(1).all()
    assert grid.loc[grid.layer == "network.0", "outside_fraction"].eq(0).all()
    assert grid.loc[grid.layer == "network.1", "outside_fraction"].eq(1).all()
    assert grid.sampled_rows.le(2).all()
    assert len(first._forward_pre_hooks) == 0


def test_near_zero_response_remains_unavailable(tmp_path):
    meta, X, coords, cfg = fixture()
    X[meta.condition == "jpeg70"] = X[meta.condition == "clean"][-1]
    # All actual test endpoints are identical, even though source grids vary.
    X[meta.split == "test"] = 0
    result = audit_responses(Additive(), X, meta, coords, cfg, tmp_path)
    assert result["curve_response"]["status"] == "unavailable"
    assert result["curve_response"]["gap"] is None
    assert result["curve_response"]["reason"].startswith("near-zero")
    assert result["curve_response"]["zero_change_reference_gap"] is None
    assert summarize_curve_response(pd.DataFrame())["status"] == "unavailable"


def test_missing_pairs_and_covariates_are_explicit(tmp_path):
    meta, X, coords, cfg = fixture()
    keep = meta.condition == "clean"
    result = audit_responses(Additive(), X[keep], meta[keep], coords, cfg, tmp_path/"clean")
    assert result["curve_response"]["status"] == "unavailable"
    coords.use_covariates = False
    result = audit_responses(torch.nn.Linear(1, 1), X[:, :1], meta, coords, cfg, tmp_path/"no_cov")
    assert result["curve_response"]["status"] == "not_applicable"


def test_equal_video_gap_is_not_pooled_frame_gap():
    table = pd.DataFrame({"dataset": ["x", "x"], "video_id": ["a", "b"],
        "paired_frames": [100, 1], "covered_frames": [100, 1], "error_sse": [0, 4],
        "total_sse": [100, 1], "total_sse_all": [100, 1], "absolute_error_sum": [0, 2],
        "sign_trials": [100, 1], "sign_agree_count": [100, 0], "near_zero_total_frames": [0, 0]})
    assert summarize_curve_response(table)["gap"] == pytest.approx(np.sqrt(2))


def test_supplied_test_logits_reuse_preserves_observed_response(tmp_path, monkeypatch):
    meta, X, coords, cfg = fixture()
    model = Additive()
    with torch.inference_mode():
        logits = model(torch.from_numpy(X)).numpy()
    # Train/validation logits are deliberately unused for response ground truth.
    logits[meta.split != "test"] = np.nan
    def unnecessary_prediction(*args, **kwargs):
        raise AssertionError("Observed test logits should be reused")
    monkeypatch.setattr("fresco.response_audit._predict", unnecessary_prediction)
    result = audit_responses(model, X, meta, coords, cfg, tmp_path, logits=logits)
    assert result["curve_response"]["gap"] < 1e-6
    logits[meta.split == "test"] = np.nan
    with pytest.raises(ValueError, match="logits"):
        audit_responses(model, X, meta, coords, cfg, tmp_path/"bad", logits=logits)
