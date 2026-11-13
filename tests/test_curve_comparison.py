import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from fresco.curve_comparison import compare_curves


def fixture_audit(tmp_path, error_kan=0., error_mlp=1.):
    directory = tmp_path/"curves"
    directory.mkdir()
    runs = []
    for head, error in [("kan_spline", error_kan), ("mlp_gelu", error_mlp)]:
        for seed in [0, 1]:
            destination = directory/head/f"seed{seed}"
            destination.mkdir(parents=True)
            runs.append({"run": {"head": head, "seed": seed}})
            rows = []
            for video in range(6):
                for condition in ["jpeg70", "blur1"]:
                    covered = 0 if video == 5 else 2
                    rows.append({"dataset": "celebdfpp", "video_id": f"v{video}",
                        "condition": condition, "group_id": f"g{video}", "paired_frames": 4,
                        "covered_frames": covered, "error_sse": covered*error, "total_sse": covered})
            pd.DataFrame(rows).to_csv(destination/"curve_intervention.csv", index=False)
    (directory/"audit.json").write_text(json.dumps({"runs": runs}))
    (directory/"provenance.json").write_text(json.dumps({"declared_seeds": [0, 1],
        "selected_seeds": [0, 1], "complete_declared_seed_coverage": True}))
    return directory


def test_additive_curve_predictor_beats_zero_change_and_uses_covered_population(tmp_path):
    directory = fixture_audit(tmp_path)
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.rglob("*") if p.is_file()}
    result = compare_curves(directory, tmp_path/"comparison", resamples=50)
    assert result["status"] == "COMPUTED"
    population = result["datasets"][0]
    assert population["coverage"] == pytest.approx(20/48)
    assert population["covered_videos"] == 5 and population["videos"] == 6
    assert population["clusters"] == 5
    contrast = population["comparisons"]["kan_spline_minus_zero_change"]
    assert contrast["head_a_mean"] == 0 and contrast["head_b_mean"] == 1
    assert contrast["difference"] == -1
    assert contrast["crossed_seeds_clusters"]["ci95"] == [-1, -1]
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.rglob("*") if p.is_file()}
    assert (tmp_path/"comparison"/"REPORT.md").is_file()


def test_zero_change_reference_and_identical_heads_have_exact_zero_differences(tmp_path):
    directory = fixture_audit(tmp_path, error_kan=1., error_mlp=1.)
    result = compare_curves(directory, tmp_path/"comparison", resamples=50)
    for value in result["datasets"][0]["comparisons"].values():
        assert value["difference"] == 0
        assert value["head_a_mean"] == value["head_b_mean"] == 1
        assert value["crossed_seeds_clusters"]["ci95"] == [0, 0]


def test_mismatched_coverage_is_rejected_without_selecting_intersection(tmp_path):
    directory = fixture_audit(tmp_path)
    path = directory/"mlp_gelu"/"seed1"/"curve_intervention.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "covered_frames"] = 1
    frame.to_csv(path, index=False)
    result = compare_curves(directory, tmp_path/"comparison", resamples=10)
    assert result["status"] == "INCONCLUSIVE"
    assert "coverage differ" in result["reason"]
    with pytest.raises(ValueError, match="outside"):
        compare_curves(directory, directory/"comparison", resamples=10)


def test_zero_total_response_does_not_receive_a_fabricated_perfect_gap(tmp_path):
    directory = fixture_audit(tmp_path, error_kan=0., error_mlp=0.)
    for path in directory.rglob("curve_intervention.csv"):
        frame = pd.read_csv(path)
        frame["total_sse"] = 0
        frame.to_csv(path, index=False)
    result = compare_curves(directory, tmp_path/"comparison", resamples=20)
    assert result["status"] == "INCONCLUSIVE"
    values = result["datasets"][0]["comparisons"]["kan_minus_mlp"]
    assert values["status"] == "INCONCLUSIVE"
    assert np.isnan(values["head_a_mean"])
