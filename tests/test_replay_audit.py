import json

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from fresco.heads import make_head
from fresco.replay_audit import replay, restore_coordinates, verify_replayed_logits
from fresco.synthetic import make_cache
from fresco.training import Coordinates, predict
from fresco.utils import atomic_json, digest_file, mark_complete


def saved_experiment(tmp_path, bind_coordinates=True):
    cache = make_cache(tmp_path / "cache")
    meta = pd.read_csv(cache / "metadata.csv", keep_default_na=False)
    feat, cov = np.load(cache / "features.npy"), np.load(cache / "covariates.npy")
    coords = Coordinates().fit(feat, cov, meta, 4, True)
    X = coords.transform(feat, cov)
    experiment = tmp_path / "original"
    common = experiment / "k4_cov1" / "heldout_none"
    run = common / "linear" / "params" / "seed0"
    run.mkdir(parents=True)
    coords.save(common / "coordinates.npz")
    model = make_head("linear", X.shape[1])
    matching = {"hidden": 1, "head_kwargs": {}}
    checkpoint = {"head": "linear", "in_features": X.shape[1], "state": model.state_dict(),
                  "matching": matching}
    if bind_coordinates:
        checkpoint["coordinates_sha256"] = digest_file(common / "coordinates.npz")
    torch.save(checkpoint, run / "checkpoint.pt")
    identity = {"head": "linear", "budget": "params", "k": 4, "covariates": True,
                "held_out": None, "seed": 0}
    atomic_json(run / "metrics.json", {"run": identity})
    logits = predict(model, X, "cpu")
    frames = meta.loc[meta.split == "test"].copy()
    frames["logit"] = logits[(meta.split == "test").to_numpy()]
    frames.to_csv(run / "frame_scores.csv", index=False)
    mark_complete(run, "fixture", ["metrics.json", "checkpoint.pt", "frame_scores.csv"])
    cfg = {"experiment": {"synthetic": True, "seeds": [0]},
           "train": {"device": "cpu", "threads": 1},
           "eval": {"interpretation_samples": 8, "grid_points": 5, "spline_support_samples": 4}}
    original = json.loads((cache / "manifest.json").read_text())
    atomic_json(experiment / "manifest.json", {"config": cfg,
        "cache": {name: original[name.split(".")[0]+"_sha256"]
                  for name in ["features.npy", "covariates.npy", "metadata.csv"]}})
    return experiment, cache, coords, feat, cov, meta, logits, run


def test_saved_head_replay_preserves_original_and_uses_saved_coordinates(tmp_path):
    experiment, cache, coords, feat, cov, _, _, _ = saved_experiment(tmp_path)
    before = {str(p): digest_file(p) for p in experiment.rglob("*") if p.is_file()}
    restored = restore_coordinates(experiment / "k4_cov1" / "heldout_none" / "coordinates.npz")
    # Exact saved parameters are required. Computed FP32 projections may differ
    # by rounding when npz packs the original strided PCA view for BLAS.
    for name, value in vars(coords).items():
        if name != "train_videos":
            np.testing.assert_array_equal(np.asarray(getattr(restored, name)), np.asarray(value), strict=True)
    actual = restored.transform(feat, cov)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(coords.transform(feat, cov), actual, rtol=2e-6, atol=1e-6, equal_nan=False)
    output = replay(experiment, cache, tmp_path / "audit", heads=["linear"], k=4)
    audit = json.loads((output / "audit.json").read_text())
    assert len(audit["runs"]) == 1
    assert (output / "linear" / "seed0" / "curve_intervention.csv").is_file()
    assert before == {str(p): digest_file(p) for p in experiment.rglob("*") if p.is_file()}
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["complete_declared_seed_coverage"]
    assert "coordinate checksum" in provenance["runs"][0]["replay_validation"]["evidence"]


def test_legacy_replay_rejects_changed_logits_and_frame_identity(tmp_path):
    _, _, _, _, _, meta, logits, run = saved_experiment(tmp_path, bind_coordinates=False)
    evidence = verify_replayed_logits(run, meta, logits)
    assert evidence["frames"] == int((meta.split == "test").sum())
    changed = logits.copy()
    changed[(meta.split == "test").to_numpy()] += .1
    with pytest.raises(ValueError, match="original test logits"):
        verify_replayed_logits(run, meta, changed)
    altered_meta = meta.copy()
    altered_meta.loc[altered_meta.split == "test", "frame_idx"] += 1
    with pytest.raises(ValueError, match="frame keys"):
        verify_replayed_logits(run, altered_meta, logits)


def test_replay_rejects_original_output_or_unmatched_cache(tmp_path):
    experiment, cache, *_ = saved_experiment(tmp_path)
    with pytest.raises(ValueError, match="new output"):
        replay(experiment, cache, experiment / "audit", heads=["linear"], k=4)
    manifest = json.loads((cache / "manifest.json").read_text())
    manifest["features_sha256"] = "0" * 64
    atomic_json(cache / "manifest.json", manifest)
    with pytest.raises(ValueError, match="does not match"):
        replay(experiment, cache, tmp_path / "audit", heads=["linear"], k=4)
    assert not (tmp_path / "audit").exists()


def test_followup_role_and_curve_coverage_are_explicit(tmp_path):
    from fresco.reporting import _analysis_role, response_audit_table
    cfg = OmegaConf.create({"experiment": {}, "train": {"conditions": ["clean", "jpeg70"]}})
    assert _analysis_role(cfg)
    cfg.train.conditions = ["clean"]
    assert _analysis_role(cfg) is None
    rows = response_audit_table(tmp_path, [{"head": "kan_spline", "details": {"interp": {
        "response_audit": {"curve_response": {"by_dataset": [{"dataset": "celebdfpp",
            "status": "partial_coverage", "coverage": .2, "gap": .1}]}}}}}])
    assert rows.iloc[0].coverage == .2
    assert rows.iloc[0].status == "partial_coverage"
