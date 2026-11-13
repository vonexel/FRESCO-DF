from __future__ import annotations

import re
import json
from pathlib import Path
from omegaconf import OmegaConf


def _read_manifest(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Missing or unreadable provenance manifest: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Provenance manifest must be a JSON object: {path}")
    return value


def cache_is_synthetic(manifest: dict) -> bool:
    # recognize explicit labels and the original audited real feature schema
    if "synthetic" in manifest:
        if type(manifest["synthetic"]) is not bool:
            raise ValueError("Cache provenance synthetic flag must be a boolean")
        return manifest["synthetic"]
    encoder = manifest.get("encoder_provenance")
    hashes = [manifest.get(name + "_sha256", "") for name in ("features", "covariates", "metadata")]
    legacy_real = (
            isinstance(manifest.get("version"), int) and manifest["version"] >= 5
            and isinstance(manifest.get("signature"), str)
            and re.fullmatch(r"[0-9a-f]{64}", manifest["signature"]) is not None
            and bool(manifest.get("encoder")) and isinstance(encoder, dict)
            and encoder.get("visual_trainable_parameter_count") == 0
            and encoder.get("whole_clip_trainable_parameter_count") == 0
            and all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes)
    )
    if legacy_real:
        return False
    raise ValueError("Cache provenance has no verified synthetic/real data origin; regenerate its manifest")


def read_cache_provenance(cache: Path, *, synthetic: bool) -> dict:
    manifest = _read_manifest(Path(cache) / "manifest.json")
    if cache_is_synthetic(manifest) != synthetic:
        raise ValueError("Synthetic/real cache provenance disagrees with experiment.synthetic; select the correct cache")
    return manifest


def report_configuration(out: Path, requested):
    manifest = _read_manifest(Path(out) / "manifest.json")
    synthetic = manifest.get("synthetic")
    saved = manifest.get("config")
    if type(synthetic) is not bool or not isinstance(saved, dict):
        raise ValueError("Experiment manifest lacks its saved config or boolean synthetic provenance")
    experiment = saved.get("experiment", {})
    if type(experiment.get("synthetic")) is not bool or experiment["synthetic"] != synthetic:
        raise ValueError("Saved experiment config and manifest synthetic provenance disagree")
    cache = manifest.get("cache_provenance")
    if not isinstance(cache, dict) or cache_is_synthetic(cache) != synthetic:
        raise ValueError("Experiment and cache synthetic/real provenance disagree; do not use this report")
    if requested.experiment.name != experiment.get("name"):
        raise ValueError("Requested report preset disagrees with the saved experiment name")
    if bool(requested.experiment.synthetic) != synthetic:
        raise ValueError("Requested report synthetic flag disagrees with saved data provenance")
    config = OmegaConf.create(saved)
    config.force = bool(requested.get("force", False))
    return config, manifest


def validate_run_provenance(runs: list[dict], manifest: dict) -> None:
    for run in runs:
        if type(run.get("synthetic")) is not bool or run["synthetic"] != manifest["synthetic"]:
            raise ValueError(f"Run synthetic/real provenance disagrees with its experiment: {run.get('run_dir')}")