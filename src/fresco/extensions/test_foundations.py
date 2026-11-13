import copy
import numpy as np
import pandas as pd
import pytest
import torch
from fresco.extensions.effects import additive_delta, effect_delta, fit_ale, fit_pd
from fresco.extensions.inventory import create, verify
from fresco.extensions.paired import (assert_source_splits, common_mask, paired_rows,
                                      response_metrics, sample_hash)
from fresco.extensions.preflight import scan_metadata
from fresco.extensions.reuse import inspect_candidate
from fresco.extensions.surrogates import fit_surrogates
from fresco.extensions.symbolic import bounded_attempt, extract_edges
from fresco.heads import make_head


def test_pairs_and_source_leakage():
    row = {"dataset": "ffpp", "video_id": "v1", "frame_idx": 7, "label": 0,
           "method": "real", "group_id": "g1", "split": "train"}
    meta = pd.DataFrame([{**row, "condition": "clean"}, {**row, "condition": "jpeg70"}])
    pairs = paired_rows(meta, split="train")
    assert len(pairs) == 1 and pairs.iloc[0].clean_row == 0
    assert len(sample_hash(pairs)) == 64
    with pytest.raises(ValueError, match="anchor"):
        paired_rows(meta.iloc[1:], split="train")
    train = meta.iloc[:1]
    val = train.assign(split="val", video_id="v2")
    with pytest.raises(ValueError, match="leakage"):
        assert_source_splits(train, val)
    assert_source_splits(train, val.assign(group_id="g2"))
    with pytest.raises(ValueError, match="validation"):
        assert_source_splits(train, val.assign(dataset="celebdfpp", group_id="g2"))


def test_equal_video_gap_zero_and_support():
    actual = np.array([1., 1., 2.])
    masks = common_mask(np.array([True, True, True]), np.array([True, True, False]))
    assert masks.tolist() == [True, True, False]
    score = response_metrics(actual, np.zeros(3), ["v1", "v1", "v2"])
    assert score["G"] == pytest.approx(1.)
    assert score["denominator_mean_square"] == pytest.approx(2.5)
    assert response_metrics(np.zeros(2), np.zeros(2), ["a", "b"])["G"] is None
    with pytest.raises(ValueError, match="finite"):
        response_metrics(actual, [0, np.nan, 0], ["v1", "v1", "v2"])


def test_pd_and_centered_ale_recover_additive_response():
    rng = np.random.default_rng(3)
    x = rng.uniform(-1, 1, (400, 2))
    teacher = lambda a: 2*a[:, 0] - 3*a[:, 1] + 5
    split = pd.DataFrame({"dataset": ["ffpp"] * len(x), "split": ["train"] * len(x)})
    pd_effects = [fit_pd(teacher, x, split, j, bins=8) for j in range(2)]
    ale_effects = [fit_ale(teacher, x, split, j, bins=8, min_bin_count=3) for j in range(2)]
    a, b = x[:40], x[40:80]
    wanted = teacher(b)-teacher(a)
    for effects in (pd_effects, ale_effects):
        prediction, covered = additive_delta(effects, a, b)
        assert covered.all()
        np.testing.assert_allclose(prediction, wanted, atol=1e-10)
    assert np.mean(np.interp(x[:, 0], ale_effects[0].grid, ale_effects[0].values)) == pytest.approx(0, abs=1e-12)
    delta, covered = effect_delta(ale_effects[0], [100, 100], [100, 101])
    assert covered.tolist() == [True, False] and delta[0] == 0 and np.isnan(delta[1])
    with pytest.raises(ValueError, match="training"):
        fit_pd(teacher, x, split.assign(dataset="celebdfpp"), 0)


def test_ale_rejects_underoccupied_bins():
    x = np.arange(8, dtype=float).reshape(-1, 1)
    meta = pd.DataFrame({"dataset": ["ffpp"]*8, "split": ["train"]*8})
    effect = fit_ale(lambda a: a[:, 0], x, meta, 0, bins=4, min_bin_count=3)
    assert not effect.occupied.any()
    delta, covered = effect_delta(effect, [0., 2.], [7., 2.])
    assert covered.tolist() == [False, True] and np.isnan(delta[0])


def test_additive_surrogates_use_source_only():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(100, 8))
    y = 2*x[:, 0]-x[:, 1]
    tr = pd.DataFrame({"dataset": "ffpp", "split": "train", "video_id": [f"t{i}" for i in range(80)],
                       "group_id": [f"t{i}" for i in range(80)]})
    va = pd.DataFrame({"dataset": "ffpp", "split": "val", "video_id": [f"v{i}" for i in range(20)],
                       "group_id": [f"v{i}" for i in range(20)]})
    models, metadata = fit_surrogates(x[:80], y[:80], tr, x[80:], y[80:], va, seed=0, include_ebm=False)
    assert set(models) == {"zero", "ridge", "gam"}
    assert np.sqrt(np.mean((models["ridge"].predict(x[80:])-y[80:])**2)) < 0.02
    assert metadata["gam"]["interactions"] == 0
    with pytest.raises(ValueError, match="validation"):
        fit_surrogates(x[:80], y[:80], tr, x[80:], y[80:], va.assign(split="test"), seed=0, include_ebm=False)


def test_real_edges_clone_and_bounded_pruning():
    torch.manual_seed(5)
    model = make_head("kan_spline", 3, 2, grid_size=5, spline_degree=3)
    original = copy.deepcopy(model.state_dict())
    x = np.random.default_rng(3).normal(size=(30, 3)).astype(np.float32)
    edges = extract_edges(model, x)
    first = edges[0]
    with torch.no_grad():
        expected = model.network[0](torch.tensor(x)).numpy()
    np.testing.assert_allclose(first["edges"].sum(-1)+first["bias"], expected, atol=1e-6)
    tr = pd.DataFrame({"dataset": "ffpp", "split": "train", "video_id": [f"t{i}" for i in range(20)],
                       "group_id": [f"t{i}" for i in range(20)]})
    va = pd.DataFrame({"dataset": "ffpp", "split": "val", "video_id": [f"v{i}" for i in range(10)],
                       "group_id": [f"v{i}" for i in range(10)]})
    result = bounded_attempt(model, x[:20], x[20:], train_meta=tr, val_meta=va, tolerance=0.05, max_rows=20)
    assert result["total_edges"] == 8 and result["status"] == "FULL_SYMBOLIFICATION_NOT_EVALUATED"
    assert np.isfinite(result["pruned_clone_validation_rmse"])
    assert all(torch.equal(v, original[k]) for k, v in model.state_dict().items())


def test_legacy_inventory_detects_tampering(tmp_path):
    source = tmp_path / "legacy"
    source.mkdir()
    (source / "old.txt").write_text("original", encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    assert create(source, baseline, project=False) == 1
    assert not verify(source, baseline)
    (source / "old.txt").write_text("tampered", encoding="utf-8")
    assert verify(source, baseline) == ["changed: old.txt"]


def test_preflight_checks_labels_pairs_and_split_leakage(tmp_path):
    records = [
        {"row": 0, "dataset": "ffpp", "split": "train", "video_id": "train", "frame_idx": 0,
         "condition": "clean", "label": 0, "method": "real", "group_id": "g1"},
        {"row": 1, "dataset": "ffpp", "split": "test", "video_id": "test", "frame_idx": 0,
         "condition": "clean", "label": 1, "method": "fake", "group_id": "g2"},
        {"row": 2, "dataset": "ffpp", "split": "test", "video_id": "test", "frame_idx": 0,
         "condition": "jpeg70", "label": 1, "method": "fake", "group_id": "g2"},
    ]
    path = tmp_path / "metadata.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    assert scan_metadata(path, ["clean", "jpeg70"])["frames"] == 2
    records[2]["group_id"] = "g1"
    pd.DataFrame(records).to_csv(path, index=False)
    with pytest.raises(ValueError, match="provenance|spans"):
        scan_metadata(path, ["clean", "jpeg70"])
    assert inspect_candidate(tmp_path / "missing", None, tmp_path, "clean")["status"] == "BLOCKED_INPUT"