import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from fresco.explanation_audit import (audit_saved, classification_by_group, fit_source_curves,
    ladder_batch, pair_statistics, summarize_ladder)
from fresco.heads import make_head
from fresco.replay_audit import restore_coordinates
from fresco.synthetic import make_cache
from fresco.training import Coordinates, predict
from fresco.utils import atomic_json, digest_file, mark_complete


class Interaction(torch.nn.Module):
    def forward(self, x):
        return x[:, 0]*x[:, 1]+x[:, 1]*x[:, 2]+2*x[:, 0]


def test_ladder_separates_context_covariate_interaction_and_representation():
    model = Interaction().train()
    a = np.array([[2., 1., 1.]], np.float32)
    b = np.array([[3., 2., 3.]], np.float32)
    grids = [np.array([-10., 10.])]*2
    curves = [np.array([0., 0.])]*2
    result = ladder_batch(model, a, b, 1, grids, curves)
    assert model.training
    expected = {"global_pd": 0, "local_ice": 5, "joint_covariate": 7, "actual": 11,
                "context": 5, "covariate_interaction": 2, "representation_remainder": 4}
    for name, value in expected.items():
        assert result[name][0] == pytest.approx(value)
    assert result["covered"].tolist() == [True]
    assert sum(result[name][0] for name in ["context", "covariate_interaction", "representation_remainder"]) == 11


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is optional in local CPU verification")
def test_ladder_cuda_toy_has_the_same_signed_decomposition():
    result = ladder_batch(Interaction().to("cuda"), np.array([[2., 1., 1.]], np.float32),
        np.array([[3., 2., 3.]], np.float32), 1, [np.array([-10., 10.])]*2,
        [np.zeros(2)]*2, device="cuda")
    assert result["local_ice"].item() == 5
    assert result["joint_covariate"].item() == 7
    assert result["actual"].item() == 11


def paired_fixture():
    rows = []
    for video in ["v1", "v2"]:
        for condition in ["clean", "jpeg70"]:
            rows.append(dict(dataset="celebdfpp", condition=condition, video_id=video, label=1,
                method="generator", scenario="swap", group_id=video, split="test", frame_idx=0))
    meta = pd.DataFrame(rows)
    # v2's changing covariate is outside the global PD grid, while B/C are defined.
    X = np.array([[2, 1, 1], [3, 2, 3], [2, 1, 1], [3, 20, 3]], dtype=np.float32)
    return meta, X


def test_all_ladder_metrics_use_the_identical_covered_population():
    meta, X = paired_fixture()
    table = pair_statistics(Interaction(), X, meta, 1, [np.array([-10., 10.])]*2,
                            [np.zeros(2)]*2, batch=1)
    summary = summarize_ladder(table)
    assert summary["coverage"] == .5
    assert summary["covered_videos"] == 1
    assert summary["predictors"]["global_pd"]["gap"] == 1
    assert summary["predictors"]["local_ice"]["gap"] == pytest.approx(6/11)
    assert summary["predictors"]["joint_covariate"]["gap"] == pytest.approx(4/11)
    assert summary["predictors"]["zero_change"]["gap"] == 1
    assert summary["remainder_sse_identity_absolute_error"] == 0
    assert table.local_ice_error_sse.sum() == 36  # uncovered v2 cannot enter B or C
    assert table.joint_covariate_error_sse.sum() == 16


def test_unchanged_outside_grid_coordinate_cancels_but_context_remains():
    a = np.array([[2, 1, 99]], dtype=np.float32)
    b = np.array([[2, 2, 99]], dtype=np.float32)
    result = ladder_batch(Interaction(), a, b, 1, [np.array([-3., 3.])]*2, [np.zeros(2)]*2)
    assert result["covered"].item()
    assert result["local_ice"].item() == 101  # outside, unchanged c2 affects c1's local effect


def test_unchanged_covariates_are_exactly_zero_even_with_replayed_fp32_rounding():
    a = np.array([[2, 1, 99]], dtype=np.float32)
    b = np.array([[3, 1, 99]], dtype=np.float32)
    clean = Interaction()(torch.from_numpy(a)).detach().numpy().astype(float)+1e-6
    changed = Interaction()(torch.from_numpy(b)).detach().numpy().astype(float)
    result = ladder_batch(Interaction(), a, b, 1, [np.array([-3., 3.])]*2, [np.zeros(2)]*2,
                          clean_logits=clean, changed_logits=changed)
    assert result["local_ice"].item() == 0
    assert result["joint_covariate"].item() == 0
    assert result["actual"].item() == pytest.approx(float(changed[0]-clean[0]))


def test_pair_provenance_mismatch_fails_instead_of_averaging_bad_pairs():
    meta, X = paired_fixture()
    meta.loc[1, "label"] = 0
    with pytest.raises(ValueError, match="provenance mismatch"):
        pair_statistics(Interaction(), X, meta, 1, [np.array([-10., 10.])]*2, [np.zeros(2)]*2)
    meta, X = paired_fixture()
    meta = meta.drop(index=0)
    with pytest.raises(ValueError, match="clean counterpart"):
        pair_statistics(Interaction(), X, meta, 1, [np.array([-10., 10.])]*2, [np.zeros(2)]*2)


def test_zero_observed_response_does_not_produce_artificial_perfect_gap():
    meta, X = paired_fixture()
    table = pair_statistics(Interaction(), X, meta, 1, [np.array([-100., 100.])]*2,
                            [np.zeros(2)]*2, logits=np.ones(len(meta)))
    result = summarize_ladder(table)
    assert result["predictors"]["global_pd"]["gap"] is None
    assert result["predictors"]["global_pd"]["sign_agreement"] is None


def _saved(tmp_path, *, native=False, covariates=True, conditions=None):
    cache = make_cache(tmp_path/"cache")
    meta = pd.read_csv(cache/"metadata.csv", keep_default_na=False)
    features, quality = np.load(cache/"features.npy"), np.load(cache/"covariates.npy")
    k = 32 if native else 4
    coords = Coordinates().fit(features, quality, meta, k, covariates)
    X = coords.transform(features, quality)
    experiment = tmp_path/"original"
    common = experiment/f"k{k}_cov{int(covariates)}"/"heldout_none"
    run = common/"linear"/"params"/"seed0"
    run.mkdir(parents=True)
    coords.save(common/"coordinates.npz")
    model = make_head("linear", X.shape[1])
    torch.save({"head": "linear", "in_features": X.shape[1], "state": model.state_dict(),
                "matching": {"hidden": 1, "head_kwargs": {}}}, run/"checkpoint.pt")
    identity = {"head": "linear", "budget": "params", "k": k, "covariates": covariates,
                "held_out": None, "seed": 0}
    atomic_json(run/"metrics.json", {"run": identity})
    mask = (meta.split == "test").to_numpy()
    if conditions:
        mask &= meta.condition.isin(conditions).to_numpy()
    scores = meta.loc[mask].copy()
    scores["logit"] = predict(model, X[mask], "cpu")
    scores.to_csv(run/"frame_scores.csv", index=False)
    mark_complete(run, "fixture", ["checkpoint.pt", "metrics.json", "frame_scores.csv"])
    cfg = {"experiment": {"seeds": [0], "conditions": conditions},
           "train": {"device": "cpu", "threads": 1}, "eval": {"interpretation_samples": 4, "grid_points": 3}}
    cache_manifest = json.loads((cache/"manifest.json").read_text())
    atomic_json(experiment/"manifest.json", {"config": cfg, "cache": {
        name: cache_manifest[name.split(".")[0]+"_sha256"] for name in ["features.npy", "covariates.npy", "metadata.csv"]}})
    return experiment, cache, k, run


def test_saved_head_ladder_replays_legacy_logits_and_keeps_original_immutable(tmp_path):
    experiment, cache, k, _ = _saved(tmp_path)
    before = {str(p): digest_file(p) for p in experiment.rglob("*") if p.is_file()}
    output = audit_saved(experiment, cache, tmp_path/"ladder", heads=["linear"], k=k, batch=13)
    assert before == {str(p): digest_file(p) for p in experiment.rglob("*") if p.is_file()}
    provenance = json.loads((output/"provenance.json").read_text())
    assert provenance["status"] == "COMPUTED"
    assert provenance["runs"][0]["original_logit_reproduction"]["frames"] > 0
    assert not (output/"_transformed_coordinates.npy").exists()
    measured = json.loads((output/"linear"/"seed0"/"ladder.json").read_text())
    assert measured["summary"]["covered_frames"] > 0
    assert set(measured["strata"]) == {"dataset", "condition", "generator", "class", "scenario"}


def test_native_clean_only_reference_evaluates_new_conditions_with_streaming_batches(tmp_path, monkeypatch):
    experiment, cache, k, _ = _saved(tmp_path, native=True, covariates=False, conditions=["clean"])
    original = Coordinates.transform
    observed = []

    def bounded(self, features, covariates):
        observed.append(len(features))
        assert len(features) <= 17
        return original(self, features, covariates)

    monkeypatch.setattr(Coordinates, "transform", bounded)
    output = audit_saved(experiment, cache, tmp_path/"native", heads=["linear"], k=k,
        covariates=False, allow_additional_conditions=True, batch=17)
    assert observed
    report = json.loads((output/"linear"/"seed0"/"ladder.json").read_text())
    assert report["summary"]["status"] == "not_applicable"
    assert {row["condition"] for row in report["classification"]} == {"clean", "jpeg90", "jpeg30"}
    provenance = json.loads((output/"provenance.json").read_text())
    assert provenance["runs"][0]["original_logit_reproduction"]["frames"] == 48*4


def test_native_and_source_range_coordinates_roundtrip_without_pickle(tmp_path):
    cache = make_cache(tmp_path/"cache")
    meta = pd.read_csv(cache/"metadata.csv", keep_default_na=False)
    f, c = np.load(cache/"features.npy"), np.load(cache/"covariates.npy")
    source = (meta.dataset == "ffpp") & (meta.split == "train")
    degraded = meta.loc[source].copy()
    degraded["condition"] = "jpeg70"
    all_meta = pd.concat([meta, degraded], ignore_index=True)
    features, covariates = np.concatenate([f, f[source]]), np.concatenate([c, c[source]+100])
    coords = Coordinates().fit(features, covariates, all_meta, 32, True, covariate_indices=[0, 3],
        covariate_scaling="source_range", scaling_conditions=["clean", "jpeg70"])
    path = tmp_path/"coordinates.npz"
    coords.save(path)
    restored = restore_coordinates(path)
    assert restored.covariate_scaling == "source_range"
    assert restored.scaling_conditions.tolist() == ["clean", "jpeg70"]
    assert restored.covariate_indices.tolist() == [0, 3]
    np.testing.assert_array_equal(restored.transform(features, covariates), coords.transform(features, covariates))
    normalized = f/np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-8)
    np.testing.assert_allclose(restored.center, normalized[source].mean(0), atol=1e-7)
    expected = (normalized-restored.center-restored.zmean)/restored.zstd
    np.testing.assert_allclose(restored.transform(f, c)[:, :32], expected, atol=1e-6)


def test_failed_legacy_reproduction_is_recorded_and_temporary_map_removed(tmp_path):
    experiment, cache, k, run = _saved(tmp_path)
    scores = pd.read_csv(run/"frame_scores.csv", keep_default_na=False)
    scores["logit"] += 1
    scores.to_csv(run/"frame_scores.csv", index=False)
    mark_complete(run, "fixture", ["checkpoint.pt", "metrics.json", "frame_scores.csv"])
    output = tmp_path/"rejected"
    with pytest.raises(ValueError, match="original test logits"):
        audit_saved(experiment, cache, output, heads=["linear"], k=k)
    assert json.loads((output/"provenance.json").read_text())["status"] == "FAILED"
    assert not (output/"_transformed_coordinates.npy").exists()


def test_mismatched_cache_hash_fails_before_creating_output(tmp_path):
    experiment, cache, k, _ = _saved(tmp_path)
    manifest = json.loads((cache/"manifest.json").read_text())
    manifest["features_sha256"] = "0"*64
    atomic_json(cache/"manifest.json", manifest)
    with pytest.raises(ValueError, match="does not match"):
        audit_saved(experiment, cache, tmp_path/"rejected", heads=["linear"], k=k)
    assert not (tmp_path/"rejected").exists()


def test_curve_fit_is_unchanged_by_target_extremes_and_supports_selected_covariates():
    records = []
    for split in ["train", "val"]:
        for label in [0, 1]:
            records.append(dict(dataset="ffpp", split=split, condition="clean", frame_idx=0,
                label=label, method="real" if label == 0 else "Deepfakes", video_id=f"{split}{label}",
                group_id=f"{split}{label}"))
    records.append(dict(dataset="celebdfpp", split="test", condition="clean", frame_idx=0,
        label=1, method="novel", video_id="target", group_id="target"))
    meta = pd.DataFrame(records)
    X = np.array([[-1, -1, -1], [1, 1, 1], [-1, -.5, -.5], [1, .5, .5], [2, 2, 2]], np.float32)
    coords = SimpleNamespace(zmean=np.zeros(1), covariate_indices=np.array([0, 3]))
    first = fit_source_curves(Interaction(), X, meta, coords, points=3)
    X[-1] = 1e6
    second = fit_source_curves(Interaction(), X, meta, coords, points=3)
    pd.testing.assert_frame_equal(first[2], second[2])
    assert set(first[2].feature) == {"jpeg_blockiness_proxy", "high_frequency_ratio"}
    assert first[3] == [2, 3]


def test_generator_metrics_share_real_controls_and_report_ap_prevalence():
    table = pd.DataFrame({"dataset": ["target"]*4, "condition": ["clean"]*4,
        "label": [0, 0, 1, 1], "method": ["real", "real", "a", "b"],
        "probability": [.1, .2, .8, .9]})
    rows = classification_by_group(table)
    assert all(row["auc"] == 1 for row in rows)
    assert all(row["n_real"] == 2 for row in rows)
    assert rows[0]["fake_prevalence_ap_reference"] == .5
