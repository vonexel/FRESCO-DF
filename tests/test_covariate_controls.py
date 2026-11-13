"""Physical covariate selection and train-only range scaling must remain auditable."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from fresco.replay_audit import restore_coordinates
from fresco.response_audit import audit_responses
from fresco.synthetic import make_cache
from fresco.training import Coordinates, selected_covariate_names, validate_covariate_indices
from fresco.run import load_cache


@pytest.mark.parametrize("indices", [[], [0, 0], [4], [-1], [True], [1.5], True, "013"])
def test_covariate_selection_fails_closed(indices):
    with pytest.raises(ValueError, match="covariate_indices"):
        validate_covariate_indices(indices)


def test_subset_keeps_physical_names_and_saved_coordinate_identity(tmp_path):
    cache = make_cache(tmp_path / "cache")
    f, c, m = load_cache(cache)
    full = Coordinates().fit(f, c, m, 8, True)
    subset = Coordinates().fit(f, c, m, 8, True, covariate_indices=[0, 1, 3])
    x = subset.transform(f, c)
    np.testing.assert_allclose(x, full.transform(f, c)[:, list(range(8)) + [8, 9, 11]])
    assert selected_covariate_names(subset) == (
        "jpeg_blockiness_proxy", "log_laplacian_variance", "high_frequency_ratio")
    subset.save(tmp_path / "coordinates.npz")
    restored = restore_coordinates(tmp_path / "coordinates.npz")
    # Serialization preserves every value/dtype/shape exactly, but packs a
    # strided PCA view. BLAS may then accumulate FP32 dot products differently.
    for name, value in vars(subset).items():
        if name != "train_videos":
            np.testing.assert_array_equal(np.asarray(getattr(restored, name)), np.asarray(value), strict=True)
    actual = restored.transform(f, c)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, x, rtol=2e-6, atol=1e-6, equal_nan=False)
    cfg = OmegaConf.create({"train": {"device": "cpu"}, "eval": {
        "interpretation_samples": 8, "grid_points": 5, "spline_support_samples": 2}})
    audit_responses(torch.nn.Linear(11, 1), x, m, subset, cfg, tmp_path / "audit")
    curves = pd.read_csv(tmp_path / "audit" / "curve_predictor.csv")
    assert set(curves.feature) == set(selected_covariate_names(subset))
    assert "face_size" not in set(pd.read_csv(tmp_path / "audit" / "response_support.csv").feature)


def test_source_range_scaling_is_fixed_without_validation_or_target(tmp_path):
    f, c, m = load_cache(make_cache(tmp_path / "cache"))
    f, c = np.asarray(f), np.asarray(c)
    source = (m.dataset == "ffpp") & (m.split == "train")
    degraded = m.loc[source].copy()
    degraded["condition"] = "jpeg70"
    meta = pd.concat([m, degraded], ignore_index=True)
    features = np.concatenate([f, f[source]])
    quality = np.concatenate([c, c[source] + 100])
    options = {"covariate_scaling": "source_range", "scaling_conditions": ["clean", "jpeg70"]}
    original = Coordinates().fit(features, quality, meta, 8, True, **options)
    changed_f, changed_c = features.copy(), quality.copy()
    outside = ~((meta.dataset == "ffpp") & (meta.split == "train"))
    changed_f[outside], changed_c[outside] = 1e5, -1e5
    altered = Coordinates().fit(changed_f, changed_c, meta, 8, True, **options)
    np.testing.assert_array_equal(original.components, altered.components)
    np.testing.assert_array_equal(original.cmean, altered.cmean)
    np.testing.assert_array_equal(original.cstd, altered.cstd)
    standard = Coordinates().fit(features, quality, meta, 8, True)
    np.testing.assert_array_equal(original.components, standard.components)
    assert np.all(original.cstd > standard.cstd)
    source_rows = (meta.dataset == "ffpp") & (meta.split == "train")
    x = original.transform(features, quality)[source_rows, 8:]
    assert np.quantile(np.abs(x), .99) < 2.6
    assert original.scaling_conditions.tolist() == ["clean", "jpeg70"]


def test_legacy_coordinates_and_covariates_off_keep_original_transform(tmp_path):
    f, c, m = load_cache(make_cache(tmp_path / "cache"))
    coords = Coordinates().fit(f, c, m, 8, True)
    original = coords.transform(f, c)
    del coords.covariate_indices
    np.testing.assert_array_equal(coords.transform(f, c), original)
    coords.use_covariates = False
    np.testing.assert_array_equal(coords.transform(f, c), original[:, :8])


def test_control_presets_share_source_recipe_steps_and_budget():
    root = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=root):
        names = ["response_clean", "support_quality3", "support_geometry1", "support_range", "basis_control"]
        configs = [compose(config_name="config", overrides=["experiment=" + name]) for name in names]
    for cfg in configs:
        assert cfg.train.max_steps == 3000 and cfg.train.validation_every_steps == 100
        assert cfg.train.target_params == 1585
        assert list(cfg.train.conditions) == ["clean"]
        assert cfg.paths.feature_cache == "data/cache/clip_vitl14_response"
        assert cfg.experiment.analysis_role.startswith("exploratory")


@pytest.mark.parametrize("preset", ["support_quality3", "support_range"])
def test_covariate_controls_train_evaluate_and_report_without_wrong_names(tmp_path, preset):
    import json
    from fresco.reporting import build_report
    from fresco.run import train_experiment
    from fresco.utils import atomic_json, digest_file

    cache = make_cache(tmp_path / "cache")
    features, quality, meta = load_cache(cache)
    # Release read-only mmap handles before replacing this synthetic fixture's
    # files: Windows rejects truncating an actively mapped file.
    features, quality = np.array(features, copy=True), np.array(quality, copy=True)
    source = (meta.dataset == "ffpp") & meta.split.isin(["train", "val"])
    tables, embeddings, covariates = [meta], [features], [quality]
    for i, condition in enumerate(["jpeg70", "blur1", "resize0.5"], start=1):
        rows = meta.loc[source].copy()
        rows["condition"] = condition
        tables.append(rows)
        embeddings.append(features[source] + i*.01)
        covariates.append(quality[source] + i*.02)
    meta = pd.concat(tables, ignore_index=True)
    meta["row"] = np.arange(len(meta))
    meta.to_csv(cache / "metadata.csv", index=False)
    np.save(cache / "features.npy", np.concatenate(embeddings))
    np.save(cache / "covariates.npy", np.concatenate(covariates))
    atomic_json(cache / "manifest.json", {"synthetic": True, **{
        key + "_sha256": digest_file(cache / filename) for key, filename in
        [("features", "features.npy"), ("covariates", "covariates.npy"), ("metadata", "metadata.csv")]}})
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1]/"configs")):
        cfg = compose(config_name="config", overrides=["experiment=" + preset,
            "experiment.synthetic=true", "experiment.seeds=[0]", "train.device=cpu", "train.threads=1",
            "train.max_steps=2", "train.validation_every_steps=1", "train.lrs=[0.001]",
            "eval.bootstrap=10", "eval.interpretation_samples=8", "eval.grid_points=5",
            "eval.spline_support_samples=2", "+eval.benchmark_repeats=1", "+eval.benchmark_warmup=1"])
    cfg.paths.output_dir = str(tmp_path / "outputs")
    out = train_experiment(cfg, cache)
    build_report(out, cfg)
    stats = json.loads((out / "stats.json").read_text())
    assert stats["primary_paired_response"]["status"] == "EXPLORATORY"
    expected = {"jpeg_blockiness_proxy", "log_laplacian_variance", "high_frequency_ratio"}
    if preset == "support_range":
        expected.add("face_size")
    for path in out.rglob("curve_predictor.csv"):
        assert set(pd.read_csv(path).feature) == expected
    for path in out.rglob("metrics.json"):
        assert json.loads(path.read_text())["training"]["optimizer_steps"] == 2
