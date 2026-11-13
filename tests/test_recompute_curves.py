import json

import numpy as np
import pandas as pd
import pytest

from fresco.recompute_curves import saved_logits, population_summaries
from fresco.utils import digest_file, mark_complete


def saved_frames(tmp_path):
    meta = pd.DataFrame({"row": [0, 1, 2, 3], "split": ["train", "test", "val", "test"]})
    pd.DataFrame({"row": [1, 3], "logit": [.25, -.5]}).to_csv(tmp_path / "frame_scores.csv", index=False)
    mark_complete(tmp_path, "fixture", ["frame_scores.csv"])
    return meta, digest_file(tmp_path / "frame_scores.csv")


def test_verified_original_logits_are_read_without_model_inference(tmp_path):
    meta, expected = saved_frames(tmp_path)
    result = saved_logits(tmp_path, meta, expected_hash=expected)
    np.testing.assert_array_equal(result[[1, 3]], [.25, -.5])
    assert np.isnan(result[[0, 2]]).all()
    changed = meta.copy()
    changed.loc[[1, 3], "row"] = [3, 1]
    with pytest.raises(ValueError, match="ordering"):
        saved_logits(tmp_path, changed, expected_hash=expected)


def test_original_frame_checksum_is_required(tmp_path):
    meta, expected = saved_frames(tmp_path)
    marker = json.loads((tmp_path / "DONE.json").read_text())
    marker["files"].clear()
    (tmp_path / "DONE.json").write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="provenance"):
        saved_logits(tmp_path, meta, expected_hash=expected)


def test_population_summary_preserves_coverage_v2_and_zero_reference():
    table = pd.DataFrame({"dataset": ["target", "target"], "condition": ["jpeg", "blur"],
        "video_id": ["a", "b"], "method": ["fake", "real"], "label": [1, 0],
        "paired_frames": [3, 2], "covered_frames": [3, 0], "error_sse": [0, 0],
        "total_sse": [3, 0], "total_sse_all": [3, 2], "absolute_error_sum": [0, 0],
        "sign_trials": [3, 0], "sign_agree_count": [3, 0], "near_zero_total_frames": [0, 0]})
    result = population_summaries(table, 0, 1e-8)
    assert result["coverage"] == .6
    assert result["gap"] == 0
    assert result["coverage_rule_version"] == 2
    assert result["zero_change_reference_gap"] == 1
    assert len(result["by_dataset_method"]) == 2
