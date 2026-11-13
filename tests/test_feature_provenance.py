"""Offline regression checks for immutable weights and crash-safe provenance."""
import json

import numpy as np
import pytest
from PIL import Image

from fresco import checkpoints, features
from fresco.checkpoints import EncoderCheckpoint
from fresco.data import sha256_file, write_csv, write_json


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    calls = []
    cache = tmp_path / "cache"
    identity = {"kind": "official_openai", "encoder": "ViT-L-14", "sha256": "a" * 64,
                "url": "https://openaipublic.azureedge.net/clip/models/" + "a" * 64 + "/ViT-L-14.pt"}

    class FakeEncoder:
        def __init__(self, *args, checkpoint=None, **kwargs):
            calls.append("load")
            self.weight = float(checkpoint.path.read_text()) if checkpoint.path else 1.0
            self.provenance = {"loader": "offline fixture", "checkpoint_sha256": checkpoint.identity["sha256"],
                "visual_trainable_parameter_count": 0, "whole_clip_trainable_parameter_count": 0,
                "projection_consistency": {"passed": False}}

        def __call__(self, images):
            calls.append("encode")
            # The loaded record must exist before the first call can produce a chunk.
            record = json.loads((cache / "encoder_provenance.json").read_text())
            assert record["state"] in {"loaded", "encoded"}
            self.provenance["projection_consistency"] = {"passed": True}
            return np.full((len(images), 1024), self.weight, np.float32)

        def close(self):
            pass

    monkeypatch.setattr(features, "CLIPCLSExtractor", FakeEncoder)
    monkeypatch.setattr(features, "version", lambda _: "offline fixture")
    monkeypatch.setattr(features, "resolve_encoder_checkpoint",
                        lambda *args: EncoderCheckpoint(identity, None, {}))
    crop = tmp_path / "crop.png"
    Image.new("RGB", (224, 224), "gray").save(crop)
    row = {"video_id": "ffpp/original/000", "frame_idx": 0, "crop_path": crop.name,
           "crop_sha256": sha256_file(crop), "face_size": .3, "dataset": "ffpp", "label": 0,
           "split": "train", "method": "real", "scenario": "in_domain", "group_id": "ffpp:000"}
    listing = tmp_path / "faces.csv"
    write_csv(listing, [row], list(row))

    def run(**options):
        return features.extract_features(listing, cache, device="cpu", conditions=["clean"], **options)

    return run, cache, calls, identity


def test_durable_provenance_survives_all_chunks_without_final_manifest(pipeline):
    run, cache, calls, _ = pipeline
    run()
    provenance = json.loads((cache / "encoder_provenance.json").read_text())
    digest = provenance.pop("record_sha256")
    assert digest == features._signature(provenance)
    assert provenance["state"] == "encoded"
    assert provenance["encoder_provenance"]["projection_consistency"]["passed"] is True
    (cache / "manifest.json").unlink()
    calls.clear()
    run()
    assert calls == []
    manifest = json.loads((cache / "manifest.json").read_text())
    assert manifest["encoder_provenance"]["projection_consistency"]["passed"] is True
    assert manifest["synthetic"] is False
    assert manifest["performance"]["newly_encoded_rows"] == 0


def test_missing_evidence_requires_explicit_recovery(pipeline):
    run, cache, calls, _ = pipeline
    run()
    (cache / "encoder_provenance.json").unlink()
    (cache / "manifest.json").unlink()
    calls.clear()
    with pytest.raises(ValueError, match="recover_provenance=true"):
        run()
    assert calls == []
    with pytest.warns(RuntimeWarning, match="Explicitly recovering"):
        run(recover_provenance=True)
    assert calls == ["load", "encode"]
    record = json.loads((cache / "encoder_provenance.json").read_text())
    assert record["recovery"]["source"] == "explicit fresh encoder verification"


def test_explicit_recovery_refuses_different_features(pipeline):
    run, cache, calls, identity = pipeline
    run()
    (cache / "encoder_provenance.json").unlink()
    (cache / "manifest.json").unlink()
    chunk = cache / "chunks/000000000.features.npy"
    np.save(chunk, np.full((1, 1024), 2, np.float32))
    marker = cache / "chunks/000000000.complete.json"
    record = json.loads(marker.read_text())
    record["feature_sha256"] = sha256_file(chunk)
    write_json(marker, record)
    with pytest.warns(RuntimeWarning, match="Explicitly recovering"):
        with pytest.raises(ValueError, match="differs from cached data"):
            run(recover_provenance=True)
    assert json.loads((cache / "encoder_provenance.json").read_text())["state"] == "loaded"


def test_corrupt_durable_record_is_not_trusted(pipeline):
    run, cache, calls, _ = pipeline
    run()
    path = cache / "encoder_provenance.json"
    record = json.loads(path.read_text())
    record["encoder_provenance"]["checkpoint_sha256"] = "b" * 64
    write_json(path, record)
    calls.clear()
    with pytest.raises(ValueError, match="provenance record is corrupt"):
        run()
    assert calls == []


def test_legacy_v5_official_manifest_resumes_without_reencoding(pipeline):
    run, cache, calls, _ = pipeline
    run()
    spec = json.loads((cache / "specification.json").read_text())
    spec.pop("signature")
    spec.pop("weights_identity")
    spec["version"] = 5
    legacy_signature = features._signature(spec)
    write_json(cache / "specification.json", {"signature": legacy_signature, **spec})
    manifest = json.loads((cache / "manifest.json").read_text())
    manifest["signature"] = legacy_signature
    manifest["version"] = 5
    manifest.pop("weights_identity")
    write_json(cache / "manifest.json", manifest)
    marker = cache / "chunks/000000000.complete.json"
    completion = json.loads(marker.read_text())
    completion["signature"] = legacy_signature
    write_json(marker, completion)
    (cache / "encoder_provenance.json").unlink()
    calls.clear()
    run()
    assert calls == []
    migrated = json.loads((cache / "specification.json").read_text())
    assert migrated["version"] == 6
    assert migrated["legacy_chunk_signature"] == legacy_signature
    run()
    assert calls == []


def test_same_path_custom_safetensors_replacement_invalidates_cache(pipeline, tmp_path, monkeypatch):
    import open_clip
    run, cache, calls, _ = pipeline
    monkeypatch.setattr(open_clip, "get_pretrained_cfg", lambda *args: {})
    monkeypatch.setattr(features, "resolve_encoder_checkpoint", checkpoints.resolve_encoder_checkpoint)
    weights = tmp_path / "weights.safetensors"
    weights.write_text("1")
    run(pretrained=str(weights))
    assert np.load(cache / "features.npy")[0, 0] == 1
    calls.clear()
    weights.write_text("2")
    with pytest.raises(ValueError, match="configuration/weights changed"):
        run(pretrained=str(weights))
    assert calls == []
    assert np.load(cache / "features.npy")[0, 0] == 1


@pytest.mark.parametrize("suffix", [".pt", ".bin", ".pth", ".SAFETENSORS"])
def test_custom_unsafe_formats_blocked_before_factory(tmp_path, monkeypatch, suffix):
    import open_clip
    path = tmp_path / ("untrusted" + suffix)
    path.write_bytes(b"not deserialized")
    monkeypatch.setattr(open_clip, "get_pretrained_cfg", lambda *args: {})
    monkeypatch.setattr(open_clip, "create_model_and_transforms",
                        lambda *args, **kwargs: pytest.fail("Unsafe checkpoint reached factory"))
    monkeypatch.setattr(open_clip, "load_openai_model",
                        lambda *args, **kwargs: pytest.fail("Custom checkpoint reached official loader"))
    with pytest.raises(ValueError, match="must use .safetensors"):
        features.CLIPCLSExtractor(pretrained=str(path), device="cpu")


def test_remote_tag_cannot_bypass_serialization_guard(tmp_path, monkeypatch):
    import open_clip
    import open_clip.pretrained
    path = tmp_path / "tag.bin"
    path.write_bytes(b"not deserialized")
    monkeypatch.setattr(open_clip, "get_pretrained_cfg", lambda *args: {"hf_hub": "example/model/"})
    monkeypatch.setattr(open_clip.pretrained, "download_pretrained", lambda *args, **kwargs: str(path))
    monkeypatch.setattr(open_clip, "create_model_and_transforms",
                        lambda *args, **kwargs: pytest.fail("Unsafe remote tag reached factory"))
    with pytest.raises(ValueError, match="must use .safetensors"):
        features.CLIPCLSExtractor(pretrained="remote-tag", device="cpu")


def test_remote_safe_tag_identity_changes_with_resolved_file(tmp_path, monkeypatch):
    import open_clip
    import open_clip.pretrained
    path = tmp_path / "tag.safetensors"
    path.write_bytes(b"first weights")
    monkeypatch.setattr(open_clip, "get_pretrained_cfg", lambda *args: {"hf_hub": "example/model/"})
    monkeypatch.setattr(open_clip.pretrained, "download_pretrained", lambda *args, **kwargs: str(path))
    first = checkpoints.resolve_encoder_checkpoint("ViT-L-14", "remote-tag", tmp_path)
    path.write_bytes(b"replacement weights")
    second = checkpoints.resolve_encoder_checkpoint("ViT-L-14", "remote-tag", tmp_path)
    assert first.path == second.path == path
    assert first.identity["sha256"] != second.identity["sha256"]
    with pytest.raises(ValueError, match="changed after identity resolution"):
        checkpoints.verify_checkpoint_file(first)


def test_safe_tag_factory_uses_resolved_file_and_original_preprocessing(tmp_path, monkeypatch):
    import open_clip
    path = tmp_path / "resolved.safetensors"
    path.write_bytes(b"fixture; factory is intercepted before deserialization")
    metadata = {"quick_gelu": True, "mean": [.1, .2, .3], "std": [.4, .5, .6],
                "interpolation": "bilinear", "resize_mode": "squash"}
    checkpoint = EncoderCheckpoint({"kind": "resolved_tag_safetensors", "sha256": sha256_file(path)},
                                   path, metadata)
    received = {}

    class FactoryReached(Exception):
        pass

    def factory(*args, **kwargs):
        received.update(kwargs)
        raise FactoryReached

    monkeypatch.setattr(open_clip, "create_model_and_transforms", factory)
    with pytest.raises(FactoryReached):
        features.CLIPCLSExtractor(pretrained="mutable-remote-tag", device="cpu", checkpoint=checkpoint)
    assert received["pretrained"] == str(path)
    assert received["force_quick_gelu"] is True
    assert received["image_mean"] == metadata["mean"]
    assert received["image_std"] == metadata["std"]
    assert received["image_interpolation"] == "bilinear"
    assert received["image_resize_mode"] == "squash"
    assert received["pretrained_hf"] is False
    assert received["pretrained_image"] is False


def test_official_identity_lookup_does_not_download(tmp_path, monkeypatch):
    import open_clip
    import open_clip.pretrained
    url = "https://openaipublic.azureedge.net/clip/models/" + "c" * 64 + "/ViT-L-14.pt"
    monkeypatch.setattr(open_clip, "get_pretrained_cfg", lambda *args: {"url": url})
    monkeypatch.setattr(open_clip.pretrained, "download_pretrained_from_url",
                        lambda *args, **kwargs: pytest.fail("Identity lookup must remain offline"))
    checkpoint = checkpoints.resolve_encoder_checkpoint("ViT-L-14", "openai", tmp_path)
    assert checkpoint.identity["sha256"] == "c" * 64
    assert checkpoint.path is None
