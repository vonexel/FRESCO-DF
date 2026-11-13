import json
import numpy as np
import pandas as pd
import pytest
from fresco.extensions import __main__ as cli
from fresco.extensions import representation_run, source_pairs, surrogate_run
from fresco.extensions.inventory import sha256


@pytest.mark.parametrize("arguments", [
    ["representation"], ["response"], ["explanations", "--analysis", "pd_ale"],
    ["explanations", "--analysis", "surrogates"], ["explanations", "--analysis", "symbolic"],
])
def test_dry_run_declares_all_ten_seeds(arguments, capsys):
    assert cli.main([*arguments, "--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["seeds"] == list(range(10))
    assert result["execution_status"] == "IMPLEMENTED"


def test_full_validate_surrogates_checks_source_pairs(monkeypatch, capsys, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    np.save(cache / "features.npy", np.zeros((1, 1024), dtype=np.float32))
    np.save(cache / "covariates.npy", np.zeros((1, 4), dtype=np.float32))
    monkeypatch.setattr(cli, "inspect", lambda *a, **k: {
        "issues": [], "source_transformed_cache": {"status": "PRESENT_UNVERIFIED"}})
    monkeypatch.setattr(surrogate_run, "ebm_environment", lambda _: {"interpret-core": "fixture"})
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: pd.DataFrame({"row": [0]}))
    monkeypatch.setattr(source_pairs, "load_source_pairs", lambda *a, **k: {
        "metadata": [1, 2], "manifest_sha256": "fixture"})
    args = ["explanations", "--analysis", "surrogates", "--main", "main", "--cache", str(cache),
            "--source-cache", "source",
            "--validate-only", "--full-check"]
    assert cli.main(args) == 0
    result = json.loads(capsys.readouterr().out)["input_preflight"]
    assert result["source_transformed_cache"]["status"] == "VERIFIED"
    assert result["source_transformed_cache"]["source_pairs"] == 2
    monkeypatch.setattr(source_pairs, "load_source_pairs", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad pair")))
    assert cli.main(args) == 2
    assert "bad pair" in capsys.readouterr().out


@pytest.mark.parametrize("changed_clean_anchor", [None, "features", "quality"])
def test_full_validate_representation_checks_historical_clean_anchors(
        monkeypatch, capsys, tmp_path, changed_clean_anchor):
    """The historical combined-cache path checks both clean embeddings and covariates."""
    original = tmp_path / "original"
    historical = tmp_path / "historical"
    original.mkdir()
    historical.mkdir()
    metadata = pd.DataFrame([
        {"row": 0, "dataset": "ffpp", "video_id": "train", "frame_idx": 0, "label": 0,
         "split": "train", "method": "real", "group_id": "group-train", "condition": "clean"},
        {"row": 1, "dataset": "ffpp", "video_id": "val", "frame_idx": 0, "label": 1,
         "split": "val", "method": "fake", "group_id": "group-val", "condition": "clean"},
    ])
    features = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 1000
    quality = np.arange(8, dtype=np.float32).reshape(2, 4) / 10
    metadata.to_csv(original / "metadata.csv", index=False)
    np.save(original / "features.npy", features)
    np.save(original / "covariates.npy", quality)

    old_rows, old_features, old_quality = [], [], []
    for row in metadata.to_dict("records"):
        for condition in ("clean", "jpeg70", "blur1", "resize0.5"):
            old_rows.append({**row, "row": len(old_rows), "condition": condition})
            old_features.append(features[row["row"]].copy())
            old_quality.append(quality[row["row"]].copy())
    if changed_clean_anchor == "features":
        old_features[0][0] += 1
    if changed_clean_anchor == "quality":
        old_quality[0][0] += 1
    pd.DataFrame(old_rows).to_csv(historical / "metadata.csv", index=False)
    np.save(historical / "features.npy", np.asarray(old_features))
    np.save(historical / "covariates.npy", np.asarray(old_quality))
    encoder = {"checkpoint_sha256": "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836",
               "projection_consistency": {"passed": True}}
    manifest = {"encoder": "ViT-L-14", "pretrained": "openai", "encoder_provenance": encoder,
                "source_conditions": ["clean", "jpeg70", "blur1", "resize0.5"]}
    for name in ("features.npy", "covariates.npy", "metadata.csv"):
        manifest[name.split(".")[0] + "_sha256"] = sha256(historical / name)
    (historical / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def inspect_fixture(*args, **kwargs):
        return {"issues": [], "warnings": ["Source-transformed cache presence does not establish pairing or exact signature"],
                "source_transformed_cache": {"status": "PRESENT_UNVERIFIED"},
                "array_checks": {name: {"expected_sha256": sha256(original / name)}
                                 for name in ("features.npy", "covariates.npy", "metadata.csv")}}

    monkeypatch.setattr(cli, "inspect", inspect_fixture)
    monkeypatch.setattr(representation_run, "validate_reuse_candidates", lambda *args: {
        "clean": {"status": "POSSIBLE_MATCH_REVIEW_REQUIRED"},
        "augmented": {"status": "POSSIBLE_MATCH_REVIEW_REQUIRED"}})
    args = ["representation", "--main", "main", "--cache", str(original), "--source-cache", str(historical),
            "--validate-only", "--full-check"]
    rc = cli.main(args)
    result = json.loads(capsys.readouterr().out)["input_preflight"]
    if changed_clean_anchor:
        assert rc == 2
        assert any("clean source rows differ" in issue for issue in result["issues"])
    else:
        assert rc == 0
        assert result["status"] == "VERIFIED"
        assert result["source_transformed_cache"]["source_pairs"] == 6