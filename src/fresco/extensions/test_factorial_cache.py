from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from fresco.extensions import __main__ as cli
from fresco.extensions import factorial_cache, figure_manifest, representation_run, representation_visuals
from fresco.extensions.factorial_cache import _replay_factorial_samples, audit_factorial_cache
from fresco.extensions.inventory import sha256
from fresco.extensions.source_pairs import load_source_pairs
from fresco.heads import make_head
from fresco.replay_audit import restore_coordinates
from fresco.training import predict


_CHECKPOINT = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
_NAMES = ("features.npy", "covariates.npy", "metadata.csv")
_REUSE = {key: {"status": "POSSIBLE_MATCH_REVIEW_REQUIRED"} for key in ("clean", "augmented")}


def _update_hashes(root):
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for name in _NAMES:
        manifest[name.split(".")[0] + "_sha256"] = sha256(root / name)
    path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    sampled = [{"recipe": r, "head": h, "covariates": bool(c), "seed": 0, "test_frames": 8}
               for r in ("clean", "augmented") for h in ("kan_spline", "mlp_gelu") for c in (0, 1)]
    monkeypatch.setattr(factorial_cache, "_replay_factorial_samples", lambda *args: sampled)
    original, historical = tmp_path / "original", tmp_path / "historical"
    original.mkdir()
    historical.mkdir()
    frames = [
        {"dataset": "ffpp", "video_id": "train", "frame_idx": 0, "split": "train", "label": 0,
         "method": "real", "scenario": "source", "group_id": "group-train"},
        {"dataset": "ffpp", "video_id": "train-fake", "frame_idx": 0, "split": "train", "label": 1,
         "method": "fake", "scenario": "source", "group_id": "group-train-fake"},
        {"dataset": "ffpp", "video_id": "val", "frame_idx": 0, "split": "val", "label": 1,
         "method": "fake", "scenario": "source", "group_id": "group-val"},
        {"dataset": "ffpp", "video_id": "val-real", "frame_idx": 0, "split": "val", "label": 0,
         "method": "real", "scenario": "source", "group_id": "group-val-real"},
        {"dataset": "ffpp", "video_id": "test", "frame_idx": 0, "split": "test", "label": 0,
         "method": "real", "scenario": "source", "group_id": "group-test"},
        {"dataset": "celebdfpp", "video_id": "target", "frame_idx": 0, "split": "test", "label": 1,
         "method": "fake", "scenario": "target", "group_id": "group-target"},
    ]
    for root, combined in ((original, False), (historical, True)):
        rows, feats, quals = [], [], []
        for index, frame in enumerate(frames):
            conditions = (["clean", "jpeg70", "blur1", "resize0.5"] if combined and frame["split"] != "test"
                          else ["clean", "jpeg70", "blur1", "resize0.5"] if frame["split"] == "test" else ["clean"])
            for condition in conditions:
                rows.append({"row": len(rows), **frame, "condition": condition})
                feats.append(np.full(1024, index + (0.1 if combined else 0), dtype=np.float32))
                quals.append(np.full(4, index / 10, dtype=np.float32))
        pd.DataFrame(rows).to_csv(root / "metadata.csv", index=False)
        np.save(root / "features.npy", np.asarray(feats))
        np.save(root / "covariates.npy", np.asarray(quals))
        manifest = {"faces_sha256": "same-face-manifest", "crop_content_sha256": "same-crops",
                    "encoder": "ViT-L-14", "pretrained": "openai", "feature": "fixture-cls",
                    "covariates": ["a", "b", "c", "d"], "conditions": ["clean", "jpeg70", "blur1", "resize0.5"],
                    "dtype": "float32", "activation_policy": "fixture", "torch_version": "2.6.0",
                    "open_clip_torch_version": "2.32.0",
                    "encoder_provenance": {"checkpoint_sha256": _CHECKPOINT,
                        "projection_consistency": {"passed": True}, "visual_trainable_parameter_count": 0,
                        "whole_clip_trainable_parameter_count": 0,
                        "device_name": "GPU-B" if combined else "GPU-A"}}
        if combined:
            manifest["source_conditions"] = ["clean", "jpeg70", "blur1", "resize0.5"]
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        _update_hashes(root)
    checks = {name: {"status": "VERIFIED", "expected_sha256": sha256(original / name)} for name in _NAMES}
    inputs = SimpleNamespace(cache=original, preflight={"array_checks": checks},
                             meta=pd.read_csv(original / "metadata.csv", keep_default_na=False),
                             features=np.load(original / "features.npy", mmap_mode="r"),
                             quality=np.load(original / "covariates.npy", mmap_mode="r"))
    return inputs, historical


def test_separate_factorial_cache_records_drift_without_weakening_original_check(fixture):
    inputs, historical = fixture
    before = {p: sha256(p) for root in (inputs.cache, historical) for p in root.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="clean source rows differ"):
        load_source_pairs(inputs, historical)
    audit = audit_factorial_cache(inputs, historical, _REUSE)
    assert audit["status"] == "VERIFIED_WITHIN_HISTORICAL_FACTORIAL_CACHE"
    assert audit["source_pairs"] == 12
    assert audit["held_out_frame_conditions_identical"] == 8
    assert audit["original_cache_features"]["identity"] == "DIFFERENT"
    assert audit["original_cache_features"]["rows_outside_tolerance"] == 4
    assert audit["original_cache_features"]["max_abs_difference"] == pytest.approx(.1)
    assert "baseline uses its own verified cache" in audit["limitation"]
    assert before == {p: sha256(p) for p in before}


def test_factorial_audit_accepts_frozen_inputs_categorical_metadata(fixture):
    inputs, historical = fixture
    for field in ("dataset", "video_id", "split", "method", "scenario", "group_id", "condition"):
        inputs.meta[field] = inputs.meta[field].astype("category")
    assert audit_factorial_cache(inputs, historical, _REUSE)["held_out_frame_conditions_identical"] == 8


@pytest.mark.parametrize("mutation, expected", [
    ("crop", "provenance mismatch"),
    ("quality", "quality covariates differ"),
    ("source_label", "source provenance differs"),
    ("test_identity", "held-out metadata differ"),
    ("nonfinite", "nonfinite"),
    ("missing_transform", "incomplete source transformations"),
])
def test_factorial_audit_refuses_unverified_inputs(fixture, mutation, expected):
    inputs, historical = fixture
    if mutation == "crop":
        path = historical / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["crop_content_sha256"] = "different"
        path.write_text(json.dumps(manifest))
    elif mutation in {"source_label", "test_identity", "missing_transform"}:
        path = historical / "metadata.csv"
        meta = pd.read_csv(path, keep_default_na=False)
        if mutation == "source_label":
            meta.loc[0, "label"] = 1
        elif mutation == "test_identity":
            meta.loc[meta.split == "test", "video_id"] = "different"
        else:
            meta = meta.iloc[1:].copy()
            meta["row"] = np.arange(len(meta))
            for name in ("features.npy", "covariates.npy"):
                path_array = historical / name
                np.save(path_array, np.load(path_array)[1:])
        meta.to_csv(path, index=False)
        _update_hashes(historical)
    else:
        name = "covariates.npy" if mutation == "quality" else "features.npy"
        path = historical / name
        array = np.load(path)
        array[0, 0] = 1 if mutation == "quality" else np.nan
        np.save(path, array)
        _update_hashes(historical)
    with pytest.raises(ValueError, match=expected):
        audit_factorial_cache(inputs, historical, _REUSE)


def test_factorial_audit_requires_both_reuse_reviews(fixture):
    inputs, historical = fixture
    with pytest.raises(ValueError, match="reuse audits"):
        audit_factorial_cache(inputs, historical, {"clean": {"status": "POSSIBLE_MATCH_REVIEW_REQUIRED"}})


@pytest.mark.parametrize("mutation", [None, "different_frames", "bad_hash", "duplicate"])
def test_all_saved_factorial_scores_are_verified_before_training(tmp_path, mutation):
    main, clean, augmented = tmp_path / "main", tmp_path / "clean", tmp_path / "augmented"
    frame = pd.DataFrame([
        {"dataset": "ffpp", "condition": "clean", "video_id": "source-test", "label": 0,
         "split": "test", "method": "real", "scenario": "source", "group_id": "g1", "frames": 32},
        {"dataset": "celebdfpp", "condition": "blur1", "video_id": "target-test", "label": 1,
         "split": "test", "method": "fake", "scenario": "target", "group_id": "g2", "frames": 31},
    ])

    def write_run(run, altered=None):
        run.mkdir(parents=True)
        for name in ("checkpoint.pt", "metrics.json", "temperature.json"):
            (run / name).write_text(name, encoding="utf-8")
        scores = frame.copy()
        if altered == "different_frames":
            scores.loc[0, "frames"] = 30
        elif altered == "duplicate":
            scores = pd.concat([scores, scores.iloc[[0]]], ignore_index=True)
        scores.to_csv(run / "scores.csv", index=False)
        marker = {"files": {name: sha256(run / name)
                            for name in ("checkpoint.pt", "metrics.json", "scores.csv", "temperature.json")}}
        (run / "DONE.json").write_text(json.dumps(marker), encoding="utf-8")
        if altered == "bad_hash":
            (run / "scores.csv").write_text("changed", encoding="utf-8")

    write_run(main / "k16_cov1/heldout_none/kan_spline/params/seed0")
    for root in (clean, augmented):
        for cov in (0, 1):
            for head in ("kan_spline", "mlp_gelu"):
                for seed in range(10):
                    selected = mutation if root == augmented and cov == 1 and head == "mlp_gelu" and seed == 9 else None
                    write_run(root / f"k16_cov{cov}/heldout_none/{head}/params/seed{seed}", selected)
    if mutation is None:
        assert representation_run.validate_saved_factorial_artifacts(clean, augmented, main) == 80
    else:
        with pytest.raises(ValueError, match="population|verification"):
            representation_run.validate_saved_factorial_artifacts(clean, augmented, main)


def test_sampled_factorial_replays_detect_changed_logits(fixture, tmp_path):
    _, historical = fixture
    meta = pd.read_csv(historical / "metadata.csv", keep_default_na=False)
    features = np.load(historical / "features.npy", mmap_mode="r")
    quality = np.load(historical / "covariates.npy", mmap_mode="r")
    held_out = meta.loc[meta.split == "test"].reset_index(drop=True)
    indices = held_out.row.to_numpy(dtype=np.int64)
    reuse = {}
    for recipe in ("clean", "augmented"):
        experiment = tmp_path / recipe
        reuse[recipe] = {"experiment": str(experiment)}
        for cov in (0, 1):
            common = experiment / f"k16_cov{cov}" / "heldout_none"
            common.mkdir(parents=True)
            components = np.zeros((16, 1024), dtype=np.float32)
            components[:, :16] = np.eye(16, dtype=np.float32)
            coordinate_file = common / "coordinates.npz"
            np.savez(coordinate_file, center=np.zeros(1024, np.float32), components=components,
                     zmean=np.zeros(16, np.float32), zstd=np.ones(16, np.float32),
                     cmean=np.zeros(4, np.float32), cstd=np.ones(4, np.float32),
                     clow=np.zeros(4, np.float32), chigh=np.ones(4, np.float32),
                     covariate_indices=np.arange(4), native=False, use_covariates=bool(cov))
            x = restore_coordinates(coordinate_file).transform(features[indices], quality[indices])
            for head in ("kan_spline", "mlp_gelu"):
                run = common / head / "params" / "seed0"
                run.mkdir(parents=True)
                torch.manual_seed(2026)
                model = make_head(head, 16 + 4*cov, 2).float().eval()
                checkpoint = {"head": head, "in_features": 16 + 4*cov, "coordinates_sha256": sha256(coordinate_file),
                              "matching": {"hidden": 2, "head_kwargs": {}}, "state": model.state_dict()}
                torch.save(checkpoint, run / "checkpoint.pt")
                recorded = held_out.copy()
                recorded["logit"] = predict(model, x, "cpu")
                recorded.to_csv(run / "frame_scores.csv", index=False)
                marker = {"files": {name: sha256(run / name) for name in ("checkpoint.pt", "frame_scores.csv")}}
                (run / "DONE.json").write_text(json.dumps(marker), encoding="utf-8")
    reports = _replay_factorial_samples(meta, features, quality, reuse, "cpu")
    assert len(reports) == 8 and all(item["test_frames"] == 8 for item in reports)

    run = tmp_path / "clean/k16_cov0/heldout_none/kan_spline/params/seed0"
    scores = pd.read_csv(run / "frame_scores.csv", keep_default_na=False)
    scores.loc[0, "logit"] += 1
    scores.to_csv(run / "frame_scores.csv", index=False)
    marker = json.loads((run / "DONE.json").read_text())
    marker["files"]["frame_scores.csv"] = sha256(run / "frame_scores.csv")
    (run / "DONE.json").write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="saved logits do not replay"):
        _replay_factorial_samples(meta, features, quality, reuse, "cpu")


def test_factorial_cli_cannot_be_used_for_other_analyses():
    with pytest.raises(SystemExit) as exc:
        cli.main(["response", "--historical-factorial-cache", "--dry-run"])
    assert exc.value.code == 2


def test_factorial_cli_full_preflight_preserves_limitation(monkeypatch, capsys, fixture):
    inputs, historical = fixture
    checks = {"issues": [], "warnings": ["Source-transformed cache presence does not establish pairing or exact signature"],
              "source_transformed_cache": {"status": "PRESENT_UNVERIFIED"}, **inputs.preflight}
    monkeypatch.setattr(cli, "inspect", lambda *args, **kwargs: checks)
    monkeypatch.setattr(representation_run, "validate_reuse_candidates", lambda *args: _REUSE)
    monkeypatch.setattr(representation_run, "validate_saved_factorial_artifacts", lambda *args: 80)
    args = ["representation", "--main", "main", "--cache", str(inputs.cache), "--source-cache", str(historical),
            "--historical-factorial-cache", "--validate-only", "--full-check"]
    assert cli.main(args) == 0
    report = json.loads(capsys.readouterr().out)["input_preflight"]
    assert report["status"] == "VERIFIED_WITH_LIMITATIONS"
    assert report["source_transformed_cache"]["status"] == "VERIFIED_FACTORIAL_ONLY"
    assert "original baseline" in report["limitations"][0]


def test_factorial_run_uses_one_historical_cache_and_records_distinct_baseline(monkeypatch, tmp_path):
    cache = tmp_path / "historical-cache"
    clean, augmented = tmp_path / "old-clean", tmp_path / "old-augmented"
    for root in (cache, clean, augmented):
        root.mkdir()
        (root / "manifest.json").write_text("{}", encoding="utf-8")
    run = tmp_path / "new-run"
    run.mkdir()
    checks, training = [], []
    provenance = {"status": "VERIFIED_WITHIN_HISTORICAL_FACTORIAL_CACHE",
                  "manifest_sha256": sha256(cache / "manifest.json"),
                  "original_cache_features": {"identity": "DIFFERENT"}}
    monkeypatch.setattr(representation_run, "validate_reuse_candidates", lambda *args: _REUSE)
    def validate_saved(*args):
        checks.append(args)
        return 80

    monkeypatch.setattr(representation_run, "validate_saved_factorial_artifacts", validate_saved)
    monkeypatch.setattr(representation_run, "FrozenInputs", lambda *args: object())
    monkeypatch.setattr(factorial_cache, "audit_factorial_cache", lambda *args, **kwargs: provenance)
    monkeypatch.setattr(representation_run, "discover_from_main", lambda _: (None, {"status": "NOT_AVAILABLE"}))

    def create_run(experiment, run_id, cfg, inputs, extra):
        assert extra["historical_factorial_cache_audit"] == provenance
        return run, "fixture"

    monkeypatch.setattr(representation_run, "create_run", create_run)
    monkeypatch.setattr(representation_run, "_config_for_k128", lambda recipe, *args: recipe)

    def train(config, cache_arg):
        training.append((config, Path(cache_arg)))
        return tmp_path / f"new-{config}"

    monkeypatch.setattr(representation_run, "train_experiment", train)
    monkeypatch.setattr(representation_run, "_saved_seed_valid", lambda _: True)
    monkeypatch.setattr(representation_run, "_assert_matching_coordinates", lambda *args: None)
    monkeypatch.setattr(representation_run, "_analyze", lambda *args: (pd.DataFrame(), {}))
    monkeypatch.setattr(representation_visuals, "representation_tables", lambda *args: None)
    monkeypatch.setattr(representation_visuals, "representation_figures", lambda *args: None)
    monkeypatch.setattr(figure_manifest, "write_figure_manifest", lambda *args: None)
    result = representation_run.run_representation(None, tmp_path / "base", tmp_path / "main", None,
        cache, clean, augmented, tmp_path / "code", "test", device="cpu", historical_factorial_cache=True)
    assert result == run and checks == [(clean, augmented, tmp_path / "main")]
    assert training == [("clean", cache), ("augmented", cache)]
    statuses = json.loads((run / "seed_status.json").read_text())
    assert sum(item["status"] == "REUSED" for item in statuses.values()) == 80
    assert sum(item["status"] == "COMPLETED" for item in statuses.values()) == 80
    assert json.loads((run / "factorial_cache_audit.json").read_text()) == provenance
    assert "original baseline is a separate reference" in (run / "REPORT.md").read_text()