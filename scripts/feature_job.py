from __future__ import annotations


import os
import sys
import time
import json
import shutil
import hashlib
import argparse
import traceback
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache/clip_vitl14"
STATUS = ROOT / "data/status/features_job.json"
READY = ROOT / "data/status/features_READY.json"
ROWS, FRAMES, CHUNKS = 2138662, 337699, 8355
CONDITIONS = ["clean", "jpeg90", "jpeg70", "jpeg50", "jpeg30", "blur1", "blur2", "blur3", "resize0.5", "resize0.25"]
CHECKPOINT_SHA = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.2)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def progress():
    markers = list((CACHE / "chunks").glob("*.complete.json"))
    rows = sum(min(256, ROWS - int(p.name.split(".")[0])) for p in markers)
    return {"completed_chunks": len(markers), "expected_chunks": CHUNKS,
            "completed_rows": rows, "expected_rows": ROWS,
            "progress_is_final_validation": False}


def save_status(state, **changes):
    state.update(changes)
    state["updated_at_utc"] = now()
    state["progress"] = progress()
    atomic_json(STATUS, state)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_cache():
    import numpy as np
    from fresco.run import load_cache

    manifest_path = CACHE / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, expected in {"rows": ROWS, "frames": FRAMES, "feature_dim": 1024, "dtype": "float32"}.items():
        require(manifest.get(key) == expected, f"Unexpected manifest {key}: {manifest.get(key)!r}")
    provenance = manifest.get("encoder_provenance") or {}
    require(provenance.get("projection_consistency", {}).get("passed") is True, "Projection consistency was not verified")
    require(provenance.get("activation") == "QuickGELU", "Unexpected encoder activation")
    require(provenance.get("checkpoint_sha256") == CHECKPOINT_SHA, "Unexpected OpenAI checkpoint SHA")
    require(provenance.get("visual_trainable_parameter_count") == 0, "Visual encoder is not frozen")
    require(provenance.get("whole_clip_trainable_parameter_count") == 0, "CLIP is not frozen")
    require(manifest.get("torch_version") == "2.6.0+cu124", "Unexpected torch version")
    require(manifest.get("open_clip_torch_version") == "2.32.0", "Unexpected OpenCLIP version")
    require(manifest.get("conditions") == CONDITIONS, "Unexpected conditions/order")
    for name in ["features", "covariates", "metadata"]:
        require(bool(manifest.get(name + "_sha256")), f"Missing {name} SHA")
    features, covariates, metadata = load_cache(CACHE)
    require(features.shape == (ROWS, 1024) and features.dtype == np.float32, "Invalid feature shape/dtype")
    require(covariates.shape == (ROWS, 4) and covariates.dtype == np.float32, "Invalid covariate shape/dtype")
    frame_keys = ["dataset", "video_id", "frame_idx"]
    require(len(metadata.drop_duplicates(frame_keys)) == FRAMES, "Unexpected number of unique frames")
    require(metadata.video_id.nunique() == 10558, "Unexpected number of videos")
    counts = {str(k): int(v) for k, v in metadata.condition.value_counts().items()}
    require(counts == {c: FRAMES if c == "clean" else 200107 for c in CONDITIONS}, "Unexpected condition row counts")
    source = metadata.loc[metadata.split.isin(["train", "val"])]
    require(source.condition.eq("clean").all(), "Degraded source training/validation rows")
    require(source.split.value_counts().to_dict() == {"train": 115196, "val": 22396}, "Unexpected source row counts")
    test = metadata.loc[metadata.split.eq("test")]
    require(test.groupby(frame_keys, sort=False).size().eq(len(CONDITIONS)).all(), "Incomplete paired test conditions")
    faces = ROOT / "data/faces/faces.csv"
    require(sha(faces) == manifest["faces_sha256"], "Faces manifest SHA changed")
    return {"status": "ready", "validated_at_utc": now(), "cache_dir": str(CACHE),
            "rows": ROWS, "frames": FRAMES, "videos": 10558, "chunks": CHUNKS,
            "feature_dim": 1024, "covariates_dim": 4, "dtype": "float32",
            "condition_counts": counts, "signature": manifest["signature"],
            "manifest": str(manifest_path), "manifest_sha256": sha(manifest_path),
            "features_sha256": manifest["features_sha256"],
            "covariates_sha256": manifest["covariates_sha256"],
            "metadata_sha256": manifest["metadata_sha256"],
            "faces_sha256": manifest["faces_sha256"], "checkpoint_sha256": CHECKPOINT_SHA,
            "uv_lock_sha256": sha(ROOT / "uv.lock"),
            "validation": {"fresco.run.load_cache": "passed", "projection_consistency": "passed",
                           "shapes_dtype_conditions_pairing": "passed", "source_faces_sha256": "passed"},
            "scope": "Preprocessing only; no real-data head training or hypothesis confirmation."}


def document_ready(result):
    marker_start, marker_end = "<!-- FRESCO_FEATURE_JOB_START -->", "<!-- FRESCO_FEATURE_JOB_END -->"
    block = f"""{marker_start}
##  feature cache `{result['manifest_sha256']}`.{marker_end}"""
    for name in ["RUN.md", "docs/VALIDATION.md"]:
        path = ROOT / name
        content = path.read_text(encoding="utf-8")
        old = "CLIP cache"
        content = content.replace(old, "CLIP cache completed")
        if marker_start in content:
            before, rest = content.split(marker_start, 1)
            _, after = rest.split(marker_end, 1)
            content = before + block + after
        else:
            content = content.rstrip() + "\n\n" + block + "\n"
        path.write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true", help="Read-only cache validation; do not publish status or edit docs")
    args = parser.parse_args()
    os.chdir(ROOT)
    for name, relative in {"TEMP": "data/tmp", "TMP": "data/tmp", "UV_CACHE_DIR": ".uv-cache"}.items():
        path = ROOT / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    if args.validate_only:
        print(json.dumps(validate_cache(), ensure_ascii=False, indent=2), flush=True)
        return
    STATUS.parent.mkdir(parents=True, exist_ok=True)

    lock = (STATUS.parent / "features_job.lock").open("a+b")
    lock.seek(0)
    if sys.platform == "win32":
        import msvcrt
        if lock.read(1) == b"":
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    log_dir = ROOT / "data/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    command = [shutil.which("uv") or "uv", "run", "--cache-dir", ".uv-cache", "--frozen",
               "fresco", "stages=[features]", "features.batch_size=4"]
    state = {"status": "running", "phase": "starting", "started_at_utc": now(),
             "supervisor_pid": os.getpid(), "command": command, "cwd": str(ROOT),
             "cache_dir": str(CACHE), "ready_file": str(READY), "exit_code": None,
             "expected_frames": FRAMES, "poll_seconds": 60,
             "environment": {k: os.environ[k] for k in ["TEMP", "TMP", "UV_CACHE_DIR"]}}
    try:
        if READY.exists():
            READY.replace(READY.with_name(f"features_READY.previous_{stamp}.json"))
        save_status(state)
        stdout_path, stderr_path = log_dir / f"features_{stamp}.stdout.log", log_dir / f"features_{stamp}.stderr.log"
        os.environ["PYTHONUNBUFFERED"] = "1"
        os.environ["HYDRA_FULL_ERROR"] = "1"
        with stdout_path.open("ab", buffering=0) as out, stderr_path.open("ab", buffering=0) as err:
            child = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err,
                                     creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
            save_status(state, child_pid=child.pid, stdout_log=str(stdout_path), stderr_log=str(stderr_path), phase="crop_verification_or_extraction")
            print(f"Started feature stage PID={child.pid}; status={STATUS}", flush=True)
            while child.poll() is None:
                time.sleep(60)
                p = progress()
                phase = "assembling_cache" if p["completed_chunks"] == CHUNKS else "extracting" if p["completed_chunks"] else "crop_verification_or_first_chunk"
                save_status(state, phase=phase)
            code = child.wait()
        save_status(state, exit_code=code, child_finished_at_utc=now())
        if code != 0:
            raise RuntimeError(f"Feature stage exited with code {code}; see {stderr_path}. Resume the same command/cache after resolving the cause.")
        save_status(state, phase="validating")
        result = validate_cache()
        document_ready(result)
        atomic_json(READY, result)
        save_status(state, status="ready", phase="complete", ready_at_utc=result["validated_at_utc"], validation=result["validation"], manifest_sha256=result["manifest_sha256"])
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except BaseException as exc:
        save_status(state, status="failed", phase="failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()