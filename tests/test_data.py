"""Data integrity and geometry checks; no model/network download required."""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fresco.data import load_official_splits, parse_celeb_test_list, safe_relative_path
from fresco.features import degrade_image, quality_covariates
from fresco.preprocess import square_crop


def test_atomic_json_retries_transient_windows_sharing_violation(tmp_path,monkeypatch):
    from pathlib import Path

    from fresco.data import write_json
    original=Path.replace
    attempts=[]
    def transient(source,target):
        attempts.append(str(target))
        if len(attempts)<3:
            raise PermissionError("simulated open Windows reader")
        return original(source,target)
    monkeypatch.setattr(Path,"replace",transient)
    monkeypatch.setattr("fresco.data.time.sleep",lambda _:None)
    write_json(tmp_path/"state.json",{"complete":True})
    assert json.loads((tmp_path/"state.json").read_text())["complete"]
    assert len(attempts)==3


def test_atomic_json_preserves_target_on_persistent_permission_error(tmp_path,monkeypatch):
    from pathlib import Path

    from fresco.data import write_json
    target=tmp_path/"state.json"
    target.write_text('{"old":true}')
    def locked(source,destination):
        raise PermissionError("persistent access denial")
    monkeypatch.setattr(Path,"replace",locked)
    monkeypatch.setattr("fresco.data.time.sleep",lambda _:None)
    with pytest.raises(PermissionError):
        write_json(target,{"new":True})
    assert json.loads(target.read_text())=={"old":True}


@pytest.mark.parametrize("unsafe", ["../x.mp4", "/x.mp4", "C:/x.mp4", "a\\x.mp4", "a/../x", "a//x", "a/./x"])
def test_reject_path_traversal(unsafe):
    with pytest.raises(ValueError):
        safe_relative_path(unsafe)


def test_celeb_official_labels_and_test_only(tmp_path):
    real = tmp_path / "Celeb-real/id1_0000.mp4"
    fake = tmp_path / "Celeb-synthesis/FaceSwap/GHOST/id1_id2_0000.mp4"
    for path in (real, fake):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    listing = tmp_path / "list.txt"
    listing.write_text("1 Celeb-real/id1_0000.mp4\n0 Celeb-synthesis/FaceSwap/GHOST/id1_id2_0000.mp4\n")
    rows = parse_celeb_test_list(listing, tmp_path)
    assert [r["label"] for r in rows] == [0, 1]
    assert {r["split"] for r in rows} == {"test"}
    assert rows[0]["group_id"] == rows[1]["group_id"] == "celeb:id1"


def test_celeb_missing_duplicate_and_wrong_label_fail(tmp_path):
    listing = tmp_path / "list.txt"
    listing.write_text("1 Celeb-real/id1_0000.mp4\n")
    with pytest.raises(FileNotFoundError):
        parse_celeb_test_list(listing, tmp_path)
    path = tmp_path / "Celeb-real/id1_0000.mp4"
    path.parent.mkdir()
    path.touch()
    listing.write_text("1 Celeb-real/id1_0000.mp4\n1 Celeb-real/id1_0000.mp4\n")
    with pytest.raises(ValueError, match="Duplicate"):
        parse_celeb_test_list(listing, tmp_path)
    listing.write_text("0 Celeb-real/id1_0000.mp4\n")
    with pytest.raises(ValueError, match="disagreement"):
        parse_celeb_test_list(listing, tmp_path)


def test_official_pairs_group_and_cross_split_rejected(tmp_path):
    for split, pair in zip(("train", "val", "test"), (("000", "001"), ("002", "003"), ("004", "005"))):
        (tmp_path / f"{split}.json").write_text(json.dumps([pair]))
    splits, groups, _ = load_official_splits(tmp_path)
    assert groups["000"] == groups["001"]
    assert splits["005"] == "test"
    (tmp_path / "test.json").write_text(json.dumps([["001", "005"]]))
    with pytest.raises(ValueError, match="across official splits"):
        load_official_splits(tmp_path)


def test_square_crop_border_preserves_geometry():
    image = np.zeros((60, 100, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(100)[None, :]
    crop = square_crop(image, (-10, -8, 40, 40), size=52, margin=1.3)
    assert crop.shape == (52, 52, 3)
    # 40*1.3=52 already; padding reflects, it does not squash a rectangle.
    assert crop[26, 26, 0] == 10


def test_paired_degradations_and_covariate_finiteness():
    rng = np.random.default_rng(7)
    image = Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))
    clean = np.asarray(degrade_image(image, "clean"))
    assert np.array_equal(clean, np.asarray(image))
    for condition in ("jpeg70", "blur2", "resize0.25"):
        first = degrade_image(image, condition)
        second = degrade_image(image, condition)
        assert first.size == image.size
        assert np.array_equal(np.asarray(first), np.asarray(second))
        covariates = quality_covariates(first, .25)
        assert covariates.shape == (4,)
        assert np.isfinite(covariates).all()
        assert covariates[2] == .25


def test_cache_clean_only_training_paired_test_and_corruption_repair(tmp_path, monkeypatch):
    from fresco import features as module
    from fresco.data import read_csv, sha256_file, write_csv
    calls = []

    class FakeEncoder:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, images):
            calls.append(len(images))
            return np.ones((len(images), 1024), dtype=np.float32)

        def close(self):
            pass

    monkeypatch.setattr(module, "CLIPCLSExtractor", FakeEncoder)
    monkeypatch.setattr(module, "version", lambda name: "test-fixture")
    records = []
    for idx, split in enumerate(("train", "val", "test")):
        crop = tmp_path / f"crop{idx}.png"
        Image.new("RGB", (224, 224), (idx * 50, 20, 180)).save(crop)
        records.append({"video_id": f"ffpp/real/{idx}", "frame_idx": 3, "crop_path": crop.name,
                        "crop_sha256": sha256_file(crop), "face_size": .3, "dataset": "ffpp",
                        "label": 0, "split": split, "method": "real", "scenario": "in_domain", "group_id": f"g{idx}"})
    listing = tmp_path / "faces.csv"
    write_csv(listing, records, list(records[0]))
    cache = module.extract_features(listing, tmp_path / "cache", device="cpu", conditions=["clean", "jpeg70", "blur2"])
    meta = read_csv(cache / "metadata.csv")
    assert len(meta) == 5
    assert [r["condition"] for r in meta if r["split"] != "test"] == ["clean", "clean"]
    assert {r["condition"] for r in meta if r["split"] == "test"} == {"clean", "jpeg70", "blur2"}
    assert np.load(cache / "features.npy").shape == (5, 1024)
    calls.clear()
    module.extract_features(listing, cache, device="cpu", conditions=["clean", "jpeg70", "blur2"])
    assert calls == []
    chunk = next((cache / "chunks").glob("*.features.npy"))
    values = np.load(chunk)
    values[0, 0] = 123
    np.save(chunk, values)
    module.extract_features(listing, cache, device="cpu", conditions=["clean", "jpeg70", "blur2"])
    assert calls
    assert np.load(cache / "features.npy")[0, 0] == 1


@pytest.fixture
def recovery_fixture(tmp_path, monkeypatch):
    import hashlib

    from fresco import preprocess as module
    from fresco.data import INDEX_COLUMNS, sha256_file, write_csv, write_json
    root, out = tmp_path / "source", tmp_path / "prepared"
    source = root / "original/000.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"synthetic source fixture; recovery must not decode it")
    row = dict(zip(INDEX_COLUMNS, ["ffpp", "ffpp/original/000", "original/000.mp4", 0,
                                  "train", "real", "in_domain", "c23", "ffpp:000"]))
    index = tmp_path / "index/index.csv"
    write_csv(index, [row], INDEX_COLUMNS)
    write_json(index.with_name("index.meta.json"), {"roots": {"ffpp": str(root)}})
    key = {"name": "fixture", "model_sha256": "a" * 64}
    monkeypatch.setattr(module, "_detector_key", lambda detector: key)
    video_hash = hashlib.sha256(row["video_id"].encode()).hexdigest()[:24]
    directory = out / "faces/ffpp" / video_hash
    directory.mkdir(parents=True)
    crop = directory / "0000000.png"
    Image.new("RGB", (224, 224), "gray").save(crop)
    face = {**row, "frame_idx": 0, "crop_path": crop.relative_to(out).as_posix(),
            "crop_sha256": sha256_file(crop), "face_size": .3, "source_width": 640, "source_height": 480}
    stat = source.stat()
    digest = sha256_file(source)
    signature = {"version": module.PIPELINE_VERSION, "source_size": stat.st_size, "source_sha256": digest,
                 "source_mtime_ns": stat.st_mtime_ns, "frames": 32, "detector": key,
                 "crop_size": 224, "margin": 1.3, "policy": "largest_face_reflect_square"}
    record = {"video_id": row["video_id"], "source_sha256": digest, "signature": signature,
              "config_hash": hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest(),
              "selected_frames": [0], "faces": [face], "missing_detection": [], "decode_failures": [],
              "status": "complete", "dataset": "ffpp", "method": "real", "label": 0}
    marker = directory / "complete.json"
    write_json(marker, record)
    return module, index, out, source, crop, marker, record


def test_finalize_existing_assembles_without_decoding_or_image_rehash(recovery_fixture, monkeypatch):
    module, index, out, source, crop, _marker, _record = recovery_fixture
    from fresco.data import read_csv
    original_hash = module.sha256_file

    def metadata_hash_only(path):
        assert Path(path) not in {source, crop}, "Recovery must not reread source/crop bytes"
        return original_hash(path)

    def no_video_job(*args, **kwargs):
        raise AssertionError("Recovery must not redetect/decode videos")

    monkeypatch.setattr(module, "sha256_file", metadata_hash_only)
    monkeypatch.setattr(module, "_video_job", no_video_job)
    result = module.finalize_existing(index, out)
    assert len(read_csv(result)) == 1
    summary = json.loads(result.with_suffix(".meta.json").read_text())
    assert summary["videos"] == summary["crops"] == summary["resumed_videos"] == 1
    assert "completion signatures + file presence" in summary["assembly_audit_mode"]
    assert summary["status_counts"] == {"complete": 1}


@pytest.mark.parametrize("corruption", ["missing_marker", "wrong_signature", "missing_png", "missing_crop_sha", "changed_source"])
def test_finalize_existing_refuses_incomplete_or_mismatched_markers(recovery_fixture, corruption):
    module, index, out, source, crop, marker, record = recovery_fixture
    from fresco.data import write_json
    if corruption == "missing_marker":
        marker.unlink()
    elif corruption == "wrong_signature":
        record["signature"]["frames"] = 16
        write_json(marker, record)
    elif corruption == "missing_png":
        crop.unlink()
    elif corruption == "missing_crop_sha":
        del record["faces"][0]["crop_sha256"]
        write_json(marker, record)
    else:
        source.write_bytes(b"changed source bytes and size")
    with pytest.raises((ValueError, FileNotFoundError), match="Recovery refused"):
        module.finalize_existing(index, out)
    assert not (out / "faces.csv").exists()
