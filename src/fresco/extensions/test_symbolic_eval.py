import json
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from fresco.extensions.inventory import sha256
from fresco.extensions.symbolic_eval import (_per_condition, _symbolic_intervals, _symbolic_transfer, _test_coordinates)
from fresco.extensions.symbolic_visuals_v2 import symbolic_figures


def _videos():
    return pd.DataFrame({"dataset": ["ffpp"]*6+["celebdfpp"]*6, "condition": ["clean"]*12,
                         "video_id": [f"v{i}" for i in range(12)], "group_id": [f"g{i}" for i in range(12)],
                         "label": [0, 1]*6, "frames": [2]*12,
                         "teacher_probability": [.1, .9]*6, "pruned_probability": [.2, .8]*6,
                         "mixed_probability": [.3, .7]*6, "pruned_error_sse": [2.]*12,
                         "mixed_error_sse": [8.]*12, "pruned_absolute_error": [2.]*12,
                         "mixed_absolute_error": [4.]*12, "teacher_logit_sse": [8.]*12,
                         "teacher_logit_sum": [-4., 4.]*6,
                         "pruned_logit_sum": [-2., 2.]*6, "mixed_logit_sum": [0.]*12,
                         "pruned_logit_sse": [2.]*12, "mixed_logit_sse": [0.]*12,
                         "pruned_cross_logit": [4.]*12, "mixed_cross_logit": [0.]*12,
                         "source_supported": [1]*12})


def test_symbolic_intervals_equal_video_and_pairing():
    table = _videos()
    result = _symbolic_intervals({i: table for i in range(10)}, 30, 2026)
    assert result["ffpp/pruned_logit_rmse"]["difference"] == 1.
    assert result["ffpp/mixed_logit_rmse"]["difference"] == 2.
    assert result["ffpp/pruned_auc_change"]["difference"] == 0.
    assert result["ffpp/pruned_normalized_logit_error"]["difference"] == .5
    transfer = _symbolic_transfer({i: table for i in range(10)}, 30, 2026)
    assert transfer["status"] == "COMPUTED" and transfer["difference"] == 0.


def test_symbolic_figures_from_saved_csv(tmp_path: Path):
    complexity = {"total_edges": 168, "pruned_edges": 2, "retained_spline_edges": 17,
                  "accepted_symbolic_edges": 151, "full_symbolification": False}
    timing = {name: {"p50_ms": 1, "model_bytes": 100} for name in ("pruned", "mixed")}
    metrics = _per_condition(_videos(), 0, .5, complexity, timing, {"pruned": .1, "mixed": .2})
    assert metrics.loc[metrics.clone == "pruned", "logit_pearson_equal_video"].eq(1.).all()
    assert metrics.loc[metrics.clone == "mixed", "logit_pearson_equal_video"].isna().all()
    metrics.to_csv(tmp_path/"symbolic_tradeoff.csv", index=False)
    pd.DataFrame([{"layer": layer, "output": 0, "input": 0, "x": x, "actual": x,
                   "approximation": x*.9, "source_bin_count": 5, "accepted": True,
                   "family": "linear", "edge_validation_rmse": .01}
                  for layer in (0, 1) for x in (-1., 0., 1.)]).to_csv(
                      tmp_path/"symbolic_edges.csv", index=False)
    symbolic_figures(tmp_path)
    for name in ("symbolic_tradeoff.svg", "symbolic_edges.svg"):
        target = tmp_path/name
        ET.parse(target)
        before = target.read_bytes()
        symbolic_figures(tmp_path)
        assert target.read_bytes() == before


def test_symbolic_reuses_only_checksum_bound_test_projection(tmp_path):
    old, new = tmp_path/"old", tmp_path/"new"
    old.mkdir()
    new.mkdir()
    np.save(old/"test_coordinates.npy", np.ones((2, 20), np.float32))
    identity = {"original_manifest_sha256": "main", "coordinates_sha256": "pca",
                "cache_hashes": {"features.npy": "features"}}
    (old/"manifest.json").write_text(json.dumps({"signature": "old-signature", "identity": identity}))
    (old/"test_coordinates.done.json").write_text(json.dumps(
        {"signature": "old-signature", "sha256": sha256(old/"test_coordinates.npy")}))
    inputs = SimpleNamespace(meta=pd.DataFrame({"split": ["train", "test", "test"], "row": [0, 1, 2]}),
                             preflight={"main_manifest_sha256": "main", "coordinates_sha256": "pca",
                                        "array_checks": {"features.npy": {"expected_sha256": "features"}}})
    meta, values = _test_coordinates(inputs, new, "new-signature", old)
    assert len(meta) == 2 and values.shape == (2, 20)
    # Windows cannot replace a file while its read-only mapping is open.
    values._mmap.close()
    np.save(old/"test_coordinates.npy", np.zeros((2, 20), np.float32))
    with pytest.raises(ValueError, match="checksum"):
        _test_coordinates(inputs, new, "new-signature", old)