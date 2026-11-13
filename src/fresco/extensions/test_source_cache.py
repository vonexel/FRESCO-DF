from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from fresco.extensions.inventory import sha256
from fresco.extensions.source_cache import _source_faces, build_source_cache
from fresco.extensions.source_pairs import load_source_pairs


def test_source_faces_match_clean_rows_and_reject_leakage(tmp_path):
    cache = tmp_path / "original"
    cache.mkdir()
    base = [
        {"row": 0, "dataset": "ffpp", "video_id": "a", "frame_idx": 0, "label": 0,
         "split": "train", "method": "real", "scenario": "source", "group_id": "g1", "condition": "clean"},
        {"row": 1, "dataset": "ffpp", "video_id": "b", "frame_idx": 0, "label": 1,
         "split": "val", "method": "fake", "scenario": "source", "group_id": "g2", "condition": "clean"},
        {"row": 2, "dataset": "celebdfpp", "video_id": "c", "frame_idx": 0, "label": 0,
         "split": "test", "method": "real", "scenario": "target", "group_id": "g3", "condition": "clean"},
    ]
    pd.DataFrame(base).to_csv(cache / "metadata.csv", index=False)
    faces = [{**{k: v for k, v in row.items() if k not in ("row", "condition")},
              "crop_path": f"{row['video_id']}.png", "crop_sha256": "hash", "face_size": .25}
             for row in base]
    face_path = tmp_path / "faces.csv"
    pd.DataFrame(faces).to_csv(face_path, index=False)
    chosen = _source_faces(cache, face_path)
    assert chosen.video_id.tolist() == ["a", "b"]
    assert chosen.base_row.tolist() == [0, 1]
    base[1]["group_id"] = "g1"
    faces[1]["group_id"] = "g1"
    pd.DataFrame(base).to_csv(cache / "metadata.csv", index=False)
    pd.DataFrame(faces).to_csv(face_path, index=False)
    with pytest.raises(ValueError, match="overlaps"):
        _source_faces(cache, face_path)


def test_source_cache_refuses_output_outside_extension_root(tmp_path):
    with pytest.raises(ValueError, match="inside outputs"):
        build_source_cache(Path("missing"), Path("missing"), tmp_path / "elsewhere", device="cpu")


def test_source_cache_generates_only_missing_pairs_and_resumes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / "base"
    cache.mkdir()
    face_dir = tmp_path / "faces"
    face_dir.mkdir()
    metadata, faces = [], []
    for row, (split, group) in enumerate((("train", "g1"), ("val", "g2"), ("test", "g3"))):
        image = face_dir / f"{row}.png"
        Image.fromarray(np.full((32, 32, 3), 30+row*50, dtype=np.uint8)).save(image)
        item = {"dataset": "ffpp", "video_id": f"v{row}", "frame_idx": 0, "label": row % 2,
                "split": split, "method": "real" if row % 2 == 0 else "fake", "scenario": "s", "group_id": group}
        metadata.append({"row": row, **item, "condition": "clean"})
        faces.append({**item, "crop_path": image.name, "crop_sha256": sha256(image), "face_size": .25})
    pd.DataFrame(metadata).to_csv(cache / "metadata.csv", index=False)
    np.save(cache / "features.npy", np.zeros((3, 1024), np.float32))
    np.save(cache / "covariates.npy", np.zeros((3, 4), np.float32))
    faces_path = face_dir / "faces.csv"
    pd.DataFrame(faces).to_csv(faces_path, index=False)
    from fresco.utils import atomic_json
    atomic_json(cache / "manifest.json", {
        "features_sha256": sha256(cache / "features.npy"), "covariates_sha256": sha256(cache / "covariates.npy"),
        "metadata_sha256": sha256(cache / "metadata.csv"), "faces_sha256": sha256(faces_path),
        "encoder": "ViT-L-14", "pretrained": "openai",
        "encoder_provenance": {"checkpoint_sha256": "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836",
                               "projection_consistency": {"passed": True}}})
    calls = []

    class FakeEncoder:
        def __init__(self, *args, **kwargs):
            calls.append("load")
            self.provenance = {"projection_consistency": {"passed": True},
                               "checkpoint_sha256": "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"}

        def __call__(self, images):
            calls.append(len(images))
            return np.repeat(np.array([np.asarray(i).mean() for i in images], np.float32)[:, None], 1024, axis=1)

        def close(self):
            pass

    monkeypatch.setattr("fresco.extensions.source_cache.CLIPCLSExtractor", FakeEncoder)
    monkeypatch.setattr("fresco.extensions.source_cache.resolve_encoder_checkpoint",
                        lambda *args: SimpleNamespace(identity={"kind": "fixture"}))
    out = tmp_path / "outputs" / "extensions" / "representation" / "fixture" / "source_pairs"
    build_source_cache(cache, faces_path, out, device="cpu", batch_size=2)
    saved = pd.read_csv(out / "metadata.csv")
    assert len(saved) == 6
    assert set(saved.split) == {"train", "val"}
    assert set(saved.condition) == {"jpeg70", "blur1", "resize0.5"}
    assert set(saved.base_row) == {0, 1}
    hashes = [sha256(out / name) for name in ("features.npy", "covariates.npy", "metadata.csv")]
    calls.clear()
    build_source_cache(cache, faces_path, out, device="cpu", batch_size=2)
    assert calls == []
    assert hashes == [sha256(out / name) for name in ("features.npy", "covariates.npy", "metadata.csv")]
    base_meta = pd.read_csv(cache / "metadata.csv")
    inputs = SimpleNamespace(cache=cache, meta=base_meta,
                             preflight={"array_checks": {name+suffix: {"expected_sha256": sha256(cache/(name+suffix))}
                                                         for name, suffix in (("features", ".npy"),
                                                                              ("covariates", ".npy"),
                                                                              ("metadata", ".csv"))}})
    source = load_source_pairs(inputs, out)
    assert len(source["metadata"]) == 6
    # A different clean-cache identity must never be accepted for training.
    inputs.preflight["array_checks"]["metadata.csv"]["expected_sha256"] = "wrong"
    with pytest.raises(ValueError, match="exact original cache"):
        load_source_pairs(inputs, out)