"""Source-only augmentation must preserve pairing, weights, and frozen coordinates."""
import json

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from PIL import Image

from fresco import features
from fresco.checkpoints import EncoderCheckpoint
from fresco.data import sha256_file, write_csv
from fresco.training import (Coordinates, build_model, fit_head, train_mask, training_row_weights,
                             validate_training_rows)


CONDITIONS = ["clean", "jpeg70", "blur1", "resize0.5"]


def test_factorial_presets_share_cache_and_keep_source_only_selection():
    from pathlib import Path
    from hydra import compose, initialize_config_dir

    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        original = compose(config_name="config", overrides=["experiment=main"])
        clean = compose(config_name="config", overrides=["experiment=response_clean"])
        augmented = compose(config_name="config", overrides=["experiment=response_augmented"])
    assert original.train.conditions is None
    assert original.train.target_params is None
    assert original.features.source_conditions is None
    assert original.paths.feature_cache == "data/cache/clip_vitl14"
    assert clean.paths.feature_cache == augmented.paths.feature_cache == "data/cache/clip_vitl14_response"
    assert list(clean.train.conditions) == ["clean"]
    assert list(augmented.train.conditions) == CONDITIONS
    assert list(clean.features.source_conditions) == list(augmented.features.source_conditions) == CONDITIONS
    assert list(clean.experiment.covariates) == [True, False]
    assert list(clean.experiment.seeds) == list(range(10))
    assert clean.train.target_params == augmented.train.target_params == 1585


@pytest.mark.parametrize("dim", [16, 20])
@pytest.mark.parametrize("head", ["kan_spline", "mlp_gelu"])
def test_common_parameter_budget_covers_both_covariate_recipes(dim, head):
    from fresco.heads import count_params

    model, matching = build_model(head, dim, "params", 72, target_params=1585)
    assert matching["feasible"]
    assert matching["target_params"] == 1585
    assert matching["params"] == count_params(model)
    assert abs(count_params(model) / 1585 - 1) <= .05


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "1585"])
def test_parameter_target_is_not_silently_coerced(value):
    with pytest.raises(ValueError, match="positive integer"):
        build_model("mlp_gelu", 20, "params", 72, target_params=value)


def test_common_parameter_target_does_not_change_time_budget_architecture():
    from fresco.heads import count_params

    reference, _ = build_model("mlp_gelu", 20, "time", 72)
    explicit, _ = build_model("mlp_gelu", 20, "time", 72, target_params=100)
    assert count_params(reference) == count_params(explicit)


def source_metadata():
    rows = []
    for split, condition_list in [("train", CONDITIONS), ("val", CONDITIONS), ("test", ["clean"])]:
        for video, label in enumerate([0, 1, 1]):
            for frame in range(video + 1):
                for condition in condition_list:
                    rows.append({"dataset": "ffpp", "video_id": f"{split}/{video}", "label": label,
                                 "split": split, "frame_idx": frame, "condition": condition,
                                 "method": "real" if label == 0 else ("FaceSwap" if video == 1 else "Deepfakes"),
                                 "group_id": f"{split}/group{video}", "scenario": "fixture"})
    rows.append({"dataset": "celebdfpp", "video_id": "target/0", "label": 1,
                 "split": "test", "frame_idx": 0, "condition": "clean", "method": "Novel",
                 "group_id": "target/0", "scenario": "fixture"})
    return pd.DataFrame(rows)


def test_complete_source_recipe_and_clean_default():
    meta = source_metadata()
    default = train_mask(meta)
    selected = validate_training_rows(meta, conditions=CONDITIONS)
    assert meta.loc[default, "condition"].eq("clean").all()
    assert selected.sum() == default.sum() * len(CONDITIONS)
    assert set(meta.loc[selected, "dataset"]) == {"ffpp"}
    assert set(meta.loc[selected, "split"]) == {"train"}
    heldout = validate_training_rows(meta, held_out="Deepfakes", conditions=CONDITIONS)
    assert "Deepfakes" not in set(meta.loc[heldout, "method"])


@pytest.mark.parametrize("corruption,reason", [
    ("missing", "Missing paired"), ("duplicate", "Duplicate"),
    ("label", "Inconsistent label"), ("target_train", "TEST ONLY"),
    ("group_leak", "group leakage"),
])
def test_augmented_recipe_rejects_incomplete_or_leaking_metadata(corruption, reason):
    meta = source_metadata()
    first = meta.index[(meta.split == "train") & (meta.condition == "jpeg70")][0]
    if corruption == "missing":
        meta = meta.drop(first)
    elif corruption == "duplicate":
        meta = pd.concat([meta, meta.loc[[first]]], ignore_index=True)
    elif corruption == "label":
        meta.loc[first, "label"] = 1
    elif corruption == "target_train":
        meta.loc[meta.dataset == "celebdfpp", "split"] = "train"
    else:
        meta.loc[meta.split == "val", "group_id"] = "train/group0"
    with pytest.raises(ValueError, match=reason):
        validate_training_rows(meta, conditions=CONDITIONS)


def test_repeated_conditions_do_not_increase_a_video_or_class_weight():
    meta = source_metadata()
    selected = meta.loc[validate_training_rows(meta, conditions=CONDITIONS)].copy()
    selected["weight"] = training_row_weights(selected)
    totals = selected.groupby(["label", "video_id"]).weight.sum()
    fake = totals.loc[1]
    assert fake.iloc[0] == pytest.approx(fake.iloc[1])
    assert totals.loc[0].sum() == pytest.approx(totals.loc[1].sum())
    recipe_totals = selected.groupby(["label", "condition"]).weight.sum()
    for label in [0, 1]:
        assert recipe_totals.loc[label].max() == pytest.approx(recipe_totals.loc[label].min())


def test_degraded_source_and_target_cannot_change_clean_fitted_coordinates():
    meta = source_metadata()
    rng = np.random.default_rng(31)
    x = rng.normal(size=(len(meta), 6)).astype(np.float32)
    c = rng.normal(size=(len(meta), 4)).astype(np.float32)
    original = Coordinates().fit(x, c, meta, 2, True)
    outside_clean_train = ~train_mask(meta)
    altered_x, altered_c = x.copy(), c.copy()
    altered_x[outside_clean_train] += 100
    altered_c[outside_clean_train] *= 100
    altered = Coordinates().fit(altered_x, altered_c, meta, 2, True)
    np.testing.assert_array_equal(original.components, altered.components)
    np.testing.assert_array_equal(original.cmean, altered.cmean)
    np.testing.assert_array_equal(original.clow, altered.clow)


def test_fit_uses_augmented_train_but_clean_source_validation():
    import torch

    meta = source_metadata()
    rng = np.random.default_rng(8)
    x = rng.normal(size=(len(meta), 3)).astype(np.float32)
    cfg = OmegaConf.create({"device": "cpu", "conditions": CONDITIONS, "threads": 1,
                            "weight_decay": 0.0, "epochs": 1, "patience": 1,
                            "batch_size": 16, "wall_seconds": 1})
    model = torch.nn.Linear(3, 1)
    _, _, log = fit_head(model, x, meta, cfg, .001, 0, "params", progress=False)
    assert log["training_rows"] == 24
    assert log["training_videos"] == 3
    assert log["training_conditions"] == CONDITIONS
    assert "clean" in log["validation_selection"]


@pytest.fixture
def extraction_fixture(tmp_path, monkeypatch):
    identity = {"kind": "official_openai", "encoder": "ViT-L-14", "sha256": "a" * 64}
    calls = []

    class FakeEncoder:
        def __init__(self, *args, **kwargs):
            self.provenance = {"loader": "offline fixture", "checkpoint_sha256": "a" * 64,
                               "visual_trainable_parameter_count": 0,
                               "whole_clip_trainable_parameter_count": 0,
                               "projection_consistency": {"passed": True}}

        def __call__(self, images):
            calls.append(len(images))
            values = np.array([np.asarray(image).mean() for image in images], np.float32)
            return np.repeat(values[:, None], 1024, axis=1)

        def close(self):
            pass

    monkeypatch.setattr(features, "CLIPCLSExtractor", FakeEncoder)
    monkeypatch.setattr(features, "version", lambda _: "offline fixture")
    monkeypatch.setattr(features, "resolve_encoder_checkpoint",
                        lambda *args: EncoderCheckpoint(identity, None, {}))
    rows = []
    for index, (dataset, split, label) in enumerate([
            ("ffpp", "train", 0), ("ffpp", "train", 1),
            ("ffpp", "val", 0), ("ffpp", "test", 1), ("celebdfpp", "test", 1)]):
        crop = tmp_path / f"crop{index}.png"
        Image.new("RGB", (32, 32), (30 + index * 20, 40, 100)).save(crop)
        rows.append({"video_id": f"{dataset}/{split}/{index}", "frame_idx": 0,
                     "crop_path": crop.name, "crop_sha256": sha256_file(crop), "face_size": .3,
                     "dataset": dataset, "split": split, "label": label,
                     "method": "real" if label == 0 else "Fake", "scenario": "fixture",
                     "group_id": f"{dataset}/{split}/{index}"})
    listing = tmp_path / "faces.csv"
    write_csv(listing, rows, list(rows[0]))
    return listing, tmp_path / "cache", calls, rows


def test_source_conditions_are_paired_for_every_class_and_do_not_expand_target(extraction_fixture):
    listing, cache, calls, _ = extraction_fixture
    test_conditions = ["clean", "jpeg90", "blur3"]
    features.extract_features(listing, cache, device="cpu", conditions=test_conditions,
                              source_conditions=CONDITIONS)
    meta = pd.read_csv(cache / "metadata.csv")
    for _, video in meta.groupby("video_id"):
        expected = test_conditions if video.split.iloc[0] == "test" else CONDITIONS
        assert video.condition.tolist() == expected
    manifest = json.loads((cache / "manifest.json").read_text())
    assert manifest["source_conditions"] == CONDITIONS
    assert len(meta) == 3 * len(CONDITIONS) + 2 * len(test_conditions)
    calls.clear()
    features.extract_features(listing, cache, device="cpu", conditions=test_conditions,
                              source_conditions=CONDITIONS)
    assert calls == []


def test_augmented_policy_cannot_reuse_existing_clean_cache(extraction_fixture):
    listing, cache, _, _ = extraction_fixture
    features.extract_features(listing, cache, device="cpu", conditions=["clean"])
    original = json.loads((cache / "specification.json").read_text())
    assert "source_conditions" not in original
    with pytest.raises(ValueError, match="configuration/weights changed"):
        features.extract_features(listing, cache, device="cpu", conditions=["clean"],
                                  source_conditions=CONDITIONS)
    assert json.loads((cache / "specification.json").read_text()) == original


def test_extraction_refuses_target_validation_rows(extraction_fixture):
    listing, cache, calls, rows = extraction_fixture
    rows[-1]["split"] = "val"
    write_csv(listing, rows, list(rows[0]))
    with pytest.raises(ValueError, match="TEST ONLY"):
        features.extract_features(listing, cache, device="cpu", conditions=["clean"],
                                  source_conditions=CONDITIONS)
    assert calls == []
