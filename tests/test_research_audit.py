import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from fresco.research_audit import (
    _aligned, AuditInputError, audit_experiment, paired_auc_uncertainty, paired_gap_uncertainty,
)
from fresco.statistics import paired_auc_bootstrap


def test_crossed_intervals_capture_training_seed_variability():
    labels = np.tile([0, 1], 6)
    groups = np.repeat(np.arange(6), 2)
    perfect = labels.astype(float)
    a = np.stack([perfect]*5 + [1-perfect]*5)
    b = np.full_like(a, .5)
    result = paired_auc_uncertainty(labels, a, b, groups, resamples=500, seed=2)
    assert result["difference"] == pytest.approx(0)
    assert result["conditional_clusters"]["ci95"] == pytest.approx([0, 0])
    lo, hi = result["crossed_seeds_clusters"]["ci95"]
    assert lo < -.1 and hi > .1
    assert result["seed_difference_sd"] > .5


def test_identical_heads_remain_paired_when_seeds_are_resampled():
    labels = np.tile([0, 1], 6)
    rng = np.random.default_rng(1)
    scores = rng.random((10, len(labels)))
    result = paired_auc_uncertainty(labels, scores, scores, np.repeat(np.arange(6), 2), resamples=100)
    assert result["conditional_clusters"]["ci95"] == [0, 0]
    assert result["crossed_seeds_clusters"]["ci95"] == [0, 0]


def test_data_only_interval_reproduces_existing_conditional_bootstrap():
    labels = np.tile([0, 1], 10)
    groups = np.repeat(np.arange(10), 2)
    rng = np.random.default_rng(19)
    a, b = rng.random((3, 20)), rng.random((3, 20))
    old = paired_auc_bootstrap(labels, a, b, groups, resamples=60, seed=27)
    new = paired_auc_uncertainty(labels, a, b, groups, resamples=60, seed=27)
    assert new["difference"] == pytest.approx(old["difference"])
    assert new["conditional_clusters"]["ci95"] == pytest.approx(old["ci95"])


def test_gap_uses_ratio_of_video_mean_squares_and_then_seed_mean():
    # Averaging video ratios gives a different result for the unequal totals.
    total = np.array([[1, 4, 1, 4, 1, 4], [4, 1, 4, 1, 4, 1]], float)
    error = np.ones_like(total)
    result = paired_gap_uncertainty(error, total, error/4, total, np.arange(6), resamples=100)
    assert result["head_a_mean"] == pytest.approx(np.sqrt(1/2.5))
    assert result["head_b_mean"] == pytest.approx(np.sqrt(.25/2.5))
    assert result["difference"] == pytest.approx(np.sqrt(1/2.5)/2)


def test_gap_crossed_interval_captures_seed_variation():
    total = np.ones((10, 6))
    error_a = np.repeat(np.linspace(.01, 1, 10)[:, None], 6, axis=1)
    error_b = np.full_like(error_a, .25)
    result = paired_gap_uncertainty(error_a, total, error_b, total, np.arange(6), resamples=300)
    data = result["conditional_clusters"]["ci95"]
    crossed = result["crossed_seeds_clusters"]["ci95"]
    assert data[1]-data[0] < 1e-12
    assert crossed[1]-crossed[0] > .1


def test_zero_response_has_no_fabricated_zero_gap():
    result = paired_gap_uncertainty(np.zeros((2, 6)), np.zeros((2, 6)),
                                   np.zeros((2, 6)), np.ones((2, 6)), np.arange(6), resamples=10)
    assert result["status"] == "INCONCLUSIVE"
    assert "undefined" in result["reason"]
    assert np.isnan(result["head_a_mean"])


def test_single_seed_does_not_claim_population_seed_interval():
    labels = np.tile([0, 1], 6)
    result = paired_auc_uncertainty(labels, labels, 1-labels, np.repeat(np.arange(6), 2), resamples=30)
    assert result["conditional_clusters"]["status"] == "COMPUTED"
    assert result["crossed_seeds_clusters"]["status"] == "INCONCLUSIVE"


def test_invalid_labels_groups_and_negative_sse_fail_closed():
    with pytest.raises(ValueError, match="binary"):
        paired_auc_uncertainty([0, 2], [[0, 1]], [[0, 1]], ["a", "b"], resamples=3)
    with pytest.raises(ValueError, match="group"):
        paired_auc_uncertainty([0, 1], [[0, 1]], [[0, 1]], ["a"], resamples=3)
    with pytest.raises(ValueError, match="nonnegative"):
        paired_gap_uncertainty([[-1, 0]], [[1, 1]], [[0, 0]], [[1, 1]], ["a", "b"], resamples=3)
    result = paired_auc_uncertainty([0, 1], [[0, 1]], [[0, 1]], ["a", "a"], resamples=3)
    assert result["status"] == "INCONCLUSIVE"


def test_alignment_does_not_intersect_missing_videos_or_clusters():
    first = pd.DataFrame({"video_id": ["a", "b"], "label": [0, 1], "group_id": ["x", "y"], "p": [.2, .8]})
    with pytest.raises(AuditInputError, match="pairing differs"):
        _aligned([("a", first), ("b", first.iloc[:1])], ["p"], ["label", "group_id"])
    second = first.copy()
    second.loc[0, "group_id"] = "other"
    with pytest.raises(AuditInputError, match="pairing differs"):
        _aligned([("a", first), ("b", second)], ["p"], ["label", "group_id"])


def _fixture(tmp_path):
    experiment = tmp_path/"experiment"
    experiment.mkdir()
    manifest = {"synthetic": True, "source": "training-source", "config": {
        "experiment": {"seeds": [0, 1]}, "train": {"epochs": 3},
        "features": {"conditions": ["clean", "jpeg70", "blur1"]}}}
    (experiment/"manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for head in ["kan_spline", "mlp_gelu"]:
        for seed in [0, 1]:
            directory = experiment/"k16_cov1"/"heldout_none"/head/"params"/f"seed{seed}"
            directory.mkdir(parents=True)
            scores, intervention = [], []
            for video in range(12):
                label = video % 2
                base = {"dataset": "celebdfpp", "video_id": f"v{video}", "label": label,
                        "method": "fake" if label else "real", "group_id": f"g{video//2}"}
                scores.append({**base, "condition": "clean", "split": "test", "probability": .1+.8*label})
                for condition in ["jpeg70", "blur1"]:
                    intervention.append({**base, "condition": condition, "error_sse": .3,
                                         "total_sse": 3., "frame_count": 3, "total": 1., "covariate": .8,
                                         "covariate_train_support_fraction": .5})
            pd.DataFrame(scores).to_csv(directory/"scores.csv", index=False)
            pd.DataFrame(intervention).to_csv(directory/"intervention.csv", index=False)
            pd.DataFrame({"epoch": [0, 1, 2], "loss": [.5, .4, .3],
                          "val_video_auc": [.6, .7, .65], "wall_s": [1, 2, 3]}).to_csv(directory/"train_log.csv", index=False)
            files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}
            (directory/"DONE.json").write_text(json.dumps({"files": files}), encoding="utf-8")
    return experiment


def test_offline_audit_preserves_experiment_and_writes_distinct_results(tmp_path):
    experiment = _fixture(tmp_path)
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in experiment.rglob("*") if p.is_file()}
    result = audit_experiment(experiment, tmp_path/"audit", resamples=50)
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in experiment.rglob("*") if p.is_file()}
    assert before == after
    assert result["status"] == "COMPUTED"
    assert result["analysis_status"] == "EXPLORATORY_RETROSPECTIVE"
    assert result["G_block"]["head_a_mean"] == pytest.approx(np.sqrt(.1))
    assert result["auroc"]["difference"] == 0
    assert (tmp_path/"audit"/"provenance.json").is_file()
    assert len(pd.read_csv(tmp_path/"audit"/"stratified_response.csv")) > 0
    assert len(pd.read_csv(tmp_path/"audit"/"convergence.csv")) == 4
    with pytest.raises(ValueError, match="outside"):
        audit_experiment(experiment, experiment/"audit", resamples=5)
    with pytest.raises(FileExistsError):
        audit_experiment(experiment, tmp_path/"audit", resamples=5)


def test_missing_seed_produces_inconclusive_without_selection(tmp_path):
    experiment = _fixture(tmp_path)
    manifest = json.loads((experiment/"manifest.json").read_text())
    manifest["config"]["experiment"]["seeds"] = [0, 1, 2]
    (experiment/"manifest.json").write_text(json.dumps(manifest))
    result = audit_experiment(experiment, tmp_path/"audit", resamples=20)
    assert result["status"] == "INCONCLUSIVE"
    assert "seed sets differ" in result["reason"]


def test_incomplete_conditions_in_all_runs_are_not_silently_accepted(tmp_path):
    experiment = _fixture(tmp_path)
    manifest = json.loads((experiment/"manifest.json").read_text())
    manifest["config"]["features"]["conditions"].append("resize0.5")
    (experiment/"manifest.json").write_text(json.dumps(manifest))
    result = audit_experiment(experiment, tmp_path/"audit", resamples=20)
    assert result["status"] == "INCONCLUSIVE"
    assert "incomplete predeclared" in result["reason"]


def test_artifact_changed_after_completion_is_detected(tmp_path):
    experiment = _fixture(tmp_path)
    path = next(experiment.rglob("scores.csv"))
    path.write_text(path.read_text()+"\n")
    result = audit_experiment(experiment, tmp_path/"audit", resamples=20)
    assert result["status"] == "INCONCLUSIVE"
    assert "DONE checksum" in result["reason"]


def test_missing_completion_checksum_is_not_treated_as_verified(tmp_path):
    experiment = _fixture(tmp_path)
    path = next(experiment.rglob("seed0/DONE.json"))
    marker = json.loads(path.read_text())
    del marker["files"]["scores.csv"]
    path.write_text(json.dumps(marker))
    result = audit_experiment(experiment, tmp_path/"audit", resamples=20)
    assert result["status"] == "INCONCLUSIVE"
    assert "DONE checksum" in result["reason"]
