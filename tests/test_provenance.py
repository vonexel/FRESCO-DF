"""Scientific labels follow recorded data, never a newly composed preset."""
import json

import numpy as np
import pytest
from omegaconf import OmegaConf

from fresco.provenance import cache_is_synthetic, read_cache_provenance, report_configuration, validate_run_provenance
from fresco.run import load_cache, train_experiment
from fresco.synthetic import make_cache


def write_manifest(directory, value):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps(value), encoding="utf-8")


def configuration(synthetic=True):
    return OmegaConf.create({
        "experiment": {"name": "smoke" if synthetic else "main", "synthetic": synthetic, "seeds": [0, 1]},
        "eval": {"margin": .01, "min_seeds_confirmatory": 10, "bootstrap": 20}, "force": False,
    })


def experiment_manifest(config, cache):
    return {"synthetic": config.experiment.synthetic, "config": OmegaConf.to_container(config),
            "cache_provenance": cache}


def test_real_training_rejects_synthetic_cache_before_loading_arrays(tmp_path, monkeypatch):
    cache = make_cache(tmp_path / "cache")
    def unexpected_load(*args):
        pytest.fail("Provenance must be checked before any training/cache load")
    monkeypatch.setattr("fresco.run.load_cache", unexpected_load)
    with pytest.raises(ValueError, match="Synthetic/real cache provenance"):
        train_experiment(configuration(False), cache)
    assert not (tmp_path / "outputs").exists()


def test_unknown_and_malformed_cache_origins_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="manifest"):
        read_cache_provenance(tmp_path, synthetic=False)
    for record in ({}, {"synthetic": "false"}, {"synthetic": 0}, {"synthetic": None}):
        write_manifest(tmp_path, record)
        with pytest.raises(ValueError, match="provenance"):
            read_cache_provenance(tmp_path, synthetic=False)
    write_manifest(tmp_path, {"synthetic": False})
    with pytest.raises(ValueError, match="Synthetic/real"):
        read_cache_provenance(tmp_path, synthetic=True)


def test_audited_legacy_real_cache_remains_usable_without_rewriting():
    record = {"version": 5, "signature": "a" * 64, "encoder": "ViT-L-14",
              "encoder_provenance": {"visual_trainable_parameter_count": 0, "whole_clip_trainable_parameter_count": 0},
              **{name + "_sha256": "b" * 64 for name in ("features", "covariates", "metadata")}}
    assert cache_is_synthetic(record) is False
    assert "synthetic" not in record
    record["encoder_provenance"]["visual_trainable_parameter_count"] = 1
    with pytest.raises(ValueError, match="origin"):
        cache_is_synthetic(record)


def test_report_uses_saved_scientific_settings(tmp_path):
    saved = configuration()
    write_manifest(tmp_path, experiment_manifest(saved, {"synthetic": True}))
    requested = configuration()
    requested.eval.margin = .5
    requested.eval.min_seeds_confirmatory = 1
    requested.experiment.seeds = [0]
    requested.force = True
    result, _ = report_configuration(tmp_path, requested)
    assert result.eval.margin == .01
    assert result.eval.min_seeds_confirmatory == 10
    assert list(result.experiment.seeds) == [0, 1]
    assert result.force is True


@pytest.mark.parametrize("change", ["cache", "config", "requested", "name"])
def test_report_rejects_conflicting_provenance_without_replacing_report(tmp_path, change):
    from fresco.reporting import build_report
    config = configuration()
    record = experiment_manifest(config, {"synthetic": True})
    if change == "cache":
        record["cache_provenance"]["synthetic"] = False
    elif change == "config":
        record["config"]["experiment"]["synthetic"] = False
    elif change == "requested":
        config.experiment.synthetic = False
    else:
        config.experiment.name = "other"
    write_manifest(tmp_path, record)
    report = tmp_path / "REPORT.md"
    report.write_text("previous report", encoding="utf-8")
    with pytest.raises(ValueError, match="provenance|disagree"):
        build_report(tmp_path, config)
    assert report.read_text(encoding="utf-8") == "previous report"


def test_report_rejects_a_synthetic_cache_previously_mislabeled_real(tmp_path):
    config = configuration(False)
    write_manifest(tmp_path, experiment_manifest(config, {"synthetic": True}))
    with pytest.raises(ValueError, match="cache synthetic/real provenance"):
        report_configuration(tmp_path, config)


def test_per_run_origin_must_match_the_experiment():
    with pytest.raises(ValueError, match="Run synthetic/real"):
        validate_run_provenance([{"synthetic": True, "run_dir": "seed0"}], {"synthetic": False})


def test_synthetic_cache_hashes_are_checked(tmp_path):
    cache = make_cache(tmp_path / "cache")
    array = np.load(cache / "features.npy", mmap_mode="r+")
    array[0, 0] += 1
    array.flush()
    del array
    with pytest.raises(ValueError, match="Corrupted features cache"):
        load_cache(cache)
