from __future__ import annotations

import os
import re
import cv2
import time
import json
import math
import hashlib
import warnings
import argparse
import numpy as np
from pathlib import Path
import onnxruntime as ort
from tqdm.auto import tqdm
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from .data import read_csv, resolve_under, sha256_file, write_csv, write_json


FACE_COLUMNS = ["video_id", "frame_idx", "crop_path", "crop_sha256", "face_size", "source_width", "source_height",
                "dataset", "label", "split", "method", "scenario", "compression", "group_id"]
_DETECTOR = None
_DETECTOR_KEY = None
PIPELINE_VERSION = 4


class YuNetDetector:
    def __init__(self, path: Path):
        self.cv2 = cv2
        self.model = cv2.FaceDetectorYN.create(str(path), "", (640, 640), .8, .3, 5000)

    def detect(self, bgr):
        cv2 = self.cv2
        height, width = bgr.shape[:2]
        scale = min(1., 640. / max(height, width))
        small = cv2.resize(bgr, (max(1, round(width * scale)), max(1, round(height * scale))))
        self.model.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.model.detect(small)
        if faces is None:
            return []
        return [(float(b[0]) / scale, float(b[1]) / scale, float(b[2]) / scale,
                 float(b[3]) / scale, float(b[-1])) for b in faces]


class HaarDetector:
    # no landmark alignment or identity tracking
    def __init__(self):
        self.cv2 = cv2
        self.model_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.model = cv2.CascadeClassifier(str(self.model_path))
        if self.model.empty():
            raise RuntimeError(f"OpenCV Haar cascade unavailable: {self.model_path}")


    def detect(self, bgr):
        cv2 = self.cv2
        height, width = bgr.shape[:2]
        scale = min(1.0, 640.0 / max(height, width))
        small = cv2.resize(bgr, (max(1, round(width * scale)), max(1, round(height * scale))))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        boxes = self.model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                            minSize=(24, 24), flags=cv2.CASCADE_SCALE_IMAGE)
        return [(x / scale, y / scale, w / scale, h / scale, 1.0) for x, y, w, h in boxes]


class SCRFDDetector:
    def __init__(self, path: Path):
        self.cv2 = cv2
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        if len(self.session.get_outputs()) not in (6, 9):
            raise ValueError("SCRFD adapter requires a 3-stride (6/9 output) ONNX export")

    def detect(self, bgr):
        cv2 = self.cv2
        h, w = bgr.shape[:2]
        scale = 640.0 / max(h, w)
        rh, rw = max(1, int(h * scale)), max(1, int(w * scale))
        canvas = np.zeros((640, 640, 3), dtype=np.uint8)
        canvas[:rh, :rw] = cv2.resize(bgr, (rw, rh))
        blob = cv2.dnn.blobFromImage(canvas, 1 / 128., (640, 640), (127.5, 127.5, 127.5), swapRB=True)
        outputs = self.session.run(None, {self.input_name: blob})
        boxes, scores = [], []
        for level, stride in enumerate((8, 16, 32)):
            score = np.asarray(outputs[level]).reshape(-1)
            distance = np.asarray(outputs[level + 3]).reshape(-1, 4) * stride
            side = 640 // stride
            anchors = len(score) // (side * side)
            if anchors not in (1, 2) or distance.shape[0] != len(score):
                raise ValueError("Unsupported SCRFD tensor shapes")
            centers = np.stack(np.mgrid[:side, :side][::-1], axis=-1).reshape(-1, 2) * stride
            centers = np.repeat(centers, anchors, axis=0)
            for idx in np.flatnonzero(score >= .5):
                x, y = centers[idx]
                left_distance, top_distance, right_distance, bottom_distance = distance[idx]
                boxes.append([(x - left_distance) / scale, (y - top_distance) / scale,
                              (left_distance + right_distance) / scale, (top_distance + bottom_distance) / scale])
                scores.append(float(score[idx]))
        if not boxes:
            return []
        keep = np.asarray(cv2.dnn.NMSBoxes(boxes, scores, .5, .4)).reshape(-1)
        return [tuple(boxes[int(i)]) + (scores[int(i)],) for i in keep]


def _detector_key(detector: str) -> dict:
    if detector == "yunet" or detector.startswith("yunet:"):
        path = Path(detector[len("yunet:"):] if detector.startswith("yunet:") else "data/models/face_detection_yunet_2023mar.onnx").resolve()
        if not path.is_file():
            raise FileNotFoundError(f"YuNet model missing: {path}. Download official weights from https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx")
        return {"name": "yunet", "path": str(path), "opencv": cv2.__version__, "model_sha256": sha256_file(path)}
    if detector == "haar":
        path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        return {"name": "haar", "opencv": cv2.__version__, "model_sha256": sha256_file(path)}
    if detector.startswith("scrfd:"):
        path = Path(detector[len("scrfd:"):]).resolve()
        return {"name": "scrfd", "path": str(path), "model_sha256": sha256_file(path)}
    raise ValueError("detector must be 'yunet', 'yunet:/path/model.onnx', 'haar' or 'scrfd:/path/model.onnx'")


def _get_detector(detector: str):
    global _DETECTOR, _DETECTOR_KEY
    if _DETECTOR_KEY != detector:
        if detector == "haar":
            _DETECTOR = HaarDetector()
        elif detector == "yunet" or detector.startswith("yunet:"):
            _DETECTOR = YuNetDetector(Path(detector[len("yunet:"):] if detector.startswith("yunet:") else "data/models/face_detection_yunet_2023mar.onnx"))
        else:
            _DETECTOR = SCRFDDetector(Path(detector[len("scrfd:"):]))
        _DETECTOR_KEY = detector
    return _DETECTOR


def square_crop(bgr, box, size: int = 224, margin: float = 1.3):
    # keep a square FOV at borders using reflected padding
    x, y, width, height = map(float, box[:4])
    side = max(2, int(math.ceil(max(width, height) * margin)))
    left, top = int(round(x + width / 2 - side / 2)), int(round(y + height / 2 - side / 2))
    image_h, image_w = bgr.shape[:2]
    pad_l, pad_t = max(0, -left), max(0, -top)
    pad_r, pad_b = max(0, left + side - image_w), max(0, top + side - image_h)
    if any((pad_l, pad_t, pad_r, pad_b)):
        bgr = cv2.copyMakeBorder(bgr, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT_101)
    left += pad_l
    top += pad_t
    crop = bgr[top:top + side, left:left + side]
    if crop.shape[:2] != (side, side):
        raise RuntimeError("Square crop geometry failed")
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA if side > size else cv2.INTER_CUBIC)


def _video_job(task: tuple) -> dict:
    cv2.setNumThreads(1)
    row, roots, data_dir_str, frames, detector, key, margin = task
    data_dir = Path(data_dir_str)
    path = resolve_under(Path(roots[row["dataset"]]), row["path"])
    video_hash = hashlib.sha256(row["video_id"].encode()).hexdigest()[:24]
    directory = data_dir / "faces" / row["dataset"] / video_hash
    marker = directory / "complete.json"
    stat = path.stat()
    source_sha256 = sha256_file(path)
    signature = {"version": PIPELINE_VERSION, "source_size": stat.st_size, "source_sha256": source_sha256,
                 "source_mtime_ns": stat.st_mtime_ns, "frames": frames, "detector": key,
                 "crop_size": 224, "margin": margin, "policy": "largest_face_reflect_square"}
    config_hash = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    if marker.is_file():
        try:
            previous = json.loads(marker.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            previous = {}
        if previous.get("config_hash") == config_hash and all(
                (data_dir / item["crop_path"]).is_file() and item.get("crop_sha256") == sha256_file(data_dir / item["crop_path"])
                for item in previous.get("faces", [])):
            # split's metadata correction does not require redetection
            for face in previous.get("faces", []):
                for field in ("dataset", "label", "split", "method", "scenario", "compression", "group_id"):
                    face[field] = row[field]
            previous.update({"dataset": row["dataset"], "method": row["method"], "label": row["label"]})
            previous["resumed"] = True
            return previous
    directory.mkdir(parents=True, exist_ok=True)
    result = {"video_id": row["video_id"], "config_hash": config_hash,
              "dataset": row["dataset"], "method": row["method"], "label": row["label"],
              "signature": signature, "source_sha256": source_sha256, "faces": [],
              "missing_detection": [], "decode_failures": [], "status": "failed"}
    decoder_threads = int(os.environ.get("FRESCO_DECODE_THREADS", "0"))
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, decoder_threads])
    result["ffmpeg_decoder_threads"] = int(capture.get(cv2.CAP_PROP_N_THREADS))
    try:
        if not capture.isOpened():
            raise RuntimeError("OpenCV could not open video")
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise RuntimeError("Video has no valid frame-count metadata")
        selected = np.unique(np.linspace(0, total - 1, min(frames, total)).round().astype(int)).tolist()
        result["selected_frames"] = selected
        detector_model = _get_detector(detector)

        sequential = total / max(1, len(selected)) <= 128
        result["decode_strategy"] = "sequential_grab" if sequential else "sparse_seek"
        decoded_index = -1
        for frame_index in selected:
            if sequential:
                ok = True
                while decoded_index < frame_index:
                    ok = capture.grab()
                    decoded_index += 1
                    if not ok:
                        break
                ok, image = capture.retrieve() if ok else (False, None)
            else:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, image = capture.read()
            if not ok or image is None:
                result["decode_failures"].append(frame_index)
                continue
            boxes = detector_model.detect(image)
            if not boxes:
                result["missing_detection"].append(frame_index)
                continue
            box = max(boxes, key=lambda b: (b[2] * b[3], b[4], -b[0], -b[1]))
            crop = square_crop(image, box, margin=margin)
            output_path = directory / f"{frame_index:07d}.png"
            if not cv2.imwrite(str(output_path), crop, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                raise IOError(f"Could not save crop: {output_path}")
            height, width = image.shape[:2]
            record = {key: row.get(key, "") for key in FACE_COLUMNS}
            record.update({"frame_idx": frame_index, "crop_path": output_path.relative_to(data_dir).as_posix(),
                           "crop_sha256": sha256_file(output_path),
                           "face_size": math.sqrt(box[2] * box[3] / (width * height)),
                           "source_width": width, "source_height": height})
            result["faces"].append(record)
        result["status"] = ("empty" if not result["faces"] else
                            "partial" if result["decode_failures"] or result["missing_detection"] else "complete")
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        capture.release()
    if result["status"] != "failed":
        write_json(marker, result)
    else:
        write_json(directory / "failed.json", result)
    return result


def extract_faces(index_csv: Path, data_dir: Path, frames: int = 32, workers: int = 4,
                  detector: str = "yunet", limit: int | None = None, margin: float = 1.3) -> Path:
    index_csv, data_dir = Path(index_csv), Path(data_dir).resolve()
    if frames < 1 or workers < 1 or margin < 1:
        raise ValueError("frames/workers must be positive and margin >= 1")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    meta = json.loads(index_csv.with_name("index.meta.json").read_text(encoding="utf-8"))
    rows = read_csv(index_csv)
    if limit is not None:
        rows = rows[:limit]
    key = _detector_key(detector)
    tasks = [(row, meta["roots"], str(data_dir), frames, detector, key, margin) for row in rows]
    results = []
    start_time = time.monotonic()

    def record_result(result):
        results.append(result)
        if len(results) % 25 == 0 or len(results) == len(tasks):
            elapsed = time.monotonic() - start_time
            progress = {"completed_videos": len(results), "total_videos": len(tasks),
                        "elapsed_seconds": elapsed, "estimated_remaining_seconds": elapsed / len(results) * (len(tasks) - len(results)),
                        "crops": sum(len(r["faces"]) for r in results), "status_counts": dict(Counter(r["status"] for r in results)),
                        "limited": limit is not None, "detector": key, "pid": os.getpid()}
            try:
                write_json(data_dir / "progress.json", progress)
            except PermissionError:
                warnings.warn("Progress file is locked; face extraction continues.", RuntimeWarning)

    if workers == 1:
        for task in tqdm(tasks, desc="Extract faces", unit="video"):
            record_result(_video_job(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_video_job, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Extract faces", unit="video"):
                record_result(future.result())
    return _aggregate_results(index_csv, data_dir, rows, results, key, frames, limit)


def _aggregate_results(index_csv, data_dir, rows, results, key, frames, limit=None,
                       audit_mode="source/crop SHA256 verified on resume; SHA256 computed at extraction"):
    results.sort(key=lambda result: result["video_id"])
    face_rows = [record for result in results for record in result["faces"]]
    path = data_dir / ("faces.csv" if limit is None else f"faces.limit{limit}.csv")
    write_csv(path, face_rows, FACE_COLUMNS)
    counts = Counter(result["status"] for result in results)
    summary = {"schema_version": 1, "detector": key, "frames_requested": frames,
               "index_sha256": sha256_file(index_csv), "manifest_sha256": sha256_file(path),
               "videos": len(rows), "crops": len(face_rows), "status_counts": dict(counts),
               "missing_detections": sum(len(r["missing_detection"]) for r in results),
               "decode_failures": sum(len(r["decode_failures"]) for r in results),
               "resumed_videos": sum(bool(r.get("resumed")) for r in results),
               "failures": [{"video_id": r["video_id"], "error": r.get("error")} for r in results if r["status"] == "failed"],
               "empty_videos": [r["video_id"] for r in results if r["status"] == "empty"],
               "by_method": {method: {"videos": sum(r["method"] == method for r in results),
                                      "empty": sum(r["method"] == method and r["status"] == "empty" for r in results),
                                      "missing_detection": sum(len(r["missing_detection"]) for r in results if r["method"] == method)}
                             for method in sorted({r["method"] for r in results})},
               "limited": limit is not None, "data_root": str(data_dir), "assembly_audit_mode": audit_mode,
               "limitations": ["Largest frontal face may not be the manipulated identity in multi-person scenes.",
                               "Report detection failure/empty-video rates by method; detector selection can bias evaluation.",
                               "Haar is an optional weaker frontal baseline; YuNet is the recommended default.",
                               "Frames with no detection are excluded; no center-crop fallback is used."]}
    write_json(path.with_suffix(".meta.json"), summary)
    write_json(data_dir / ("preprocess_log.json" if limit is None else f"preprocess_log.limit{limit}.json"),
               [{k: v for k, v in result.items() if k != "faces"} for result in results])
    if not face_rows:
        raise RuntimeError(f"No faces extracted; inspect {path.with_suffix('.meta.json')}")
    if counts.get("failed"):
        raise RuntimeError(f"{counts['failed']} videos failed. Successfully processed videos are resumable; see preprocessing log.")
    return path


def finalize_existing(index_csv: Path, data_dir: Path, frames: int = 32,
                      detector: str = "yunet", margin: float = 1.3) -> Path:
    if frames < 1 or margin < 1:
        raise ValueError("frames must be positive and margin >= 1")
    index_csv, data_dir = Path(index_csv), Path(data_dir).resolve()
    meta = json.loads(index_csv.with_name("index.meta.json").read_text(encoding="utf-8"))
    rows = read_csv(index_csv)
    if not rows or len({row["video_id"] for row in rows}) != len(rows):
        raise ValueError("Recovery requires a nonempty index with unique video IDs")
    key, results = _detector_key(detector), []
    for row in tqdm(rows, desc="Recover completed faces", unit="video"):
        video_hash = hashlib.sha256(row["video_id"].encode()).hexdigest()[:24]
        directory = data_dir / "faces" / row["dataset"] / video_hash
        marker = directory / "complete.json"
        if not marker.is_file():
            raise FileNotFoundError(f"Recovery refused: missing completion marker for {row['video_id']}: {marker}")
        result = json.loads(marker.read_text(encoding="utf-8"))
        if not isinstance(result, dict) or result.get("video_id") != row["video_id"]:
            raise ValueError(f"Recovery refused: completion identity mismatch for {row['video_id']}")
        digest = result.get("source_sha256", "")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Recovery refused: missing source SHA256 for {row['video_id']}")
        source = resolve_under(Path(meta["roots"][row["dataset"]]), row["path"])
        stat = source.stat()
        signature = {"version": PIPELINE_VERSION, "source_size": stat.st_size, "source_sha256": digest,
                     "source_mtime_ns": stat.st_mtime_ns, "frames": frames, "detector": key,
                     "crop_size": 224, "margin": margin, "policy": "largest_face_reflect_square"}
        expected_hash = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        if result.get("signature") != signature or result.get("config_hash") != expected_hash:
            raise ValueError(f"Recovery refused: source/config signature mismatch for {row['video_id']}")
        selected = result.get("selected_frames", [])
        faces = result.get("faces", [])
        missing = result.get("missing_detection", [])
        failures = result.get("decode_failures", [])
        if (not selected or len(selected) > frames or selected != sorted(set(selected))
                or any(not isinstance(value, int) or value < 0 for value in selected)):
            raise ValueError(f"Recovery refused: invalid selected frames for {row['video_id']}")
        frame_ids = [face.get("frame_idx") for face in faces]
        combined = frame_ids + missing + failures
        if len(combined) != len(set(combined)) or set(combined) != set(selected):
            raise ValueError(f"Recovery refused: incomplete/duplicate frame accounting for {row['video_id']}")
        expected_status = "empty" if not faces else "partial" if missing or failures else "complete"
        if result.get("status") != expected_status:
            raise ValueError(f"Recovery refused: inconsistent completion status for {row['video_id']}")
        with os.scandir(directory) as entries:
            present = {entry.name: entry.stat(follow_symlinks=False).st_size
                       for entry in entries if entry.name.endswith(".png") and entry.is_file(follow_symlinks=False)}
        for face in faces:
            frame = face["frame_idx"]
            filename = f"{frame:07d}.png"
            expected_path = (directory / filename).relative_to(data_dir).as_posix()
            if face.get("video_id") != row["video_id"] or face.get("crop_path") != expected_path:
                raise ValueError(f"Recovery refused: crop identity/path mismatch for {row['video_id']}")
            if present.get(filename, 0) <= 0:
                raise FileNotFoundError(f"Recovery refused: missing/empty PNG {directory / filename}")
            crop_digest = face.get("crop_sha256", "")
            if not isinstance(crop_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", crop_digest):
                raise ValueError(f"Recovery refused: missing crop SHA256 for {row['video_id']}/{filename}")
            for field in ("dataset", "label", "split", "method", "scenario", "compression", "group_id"):
                face[field] = row[field]
        result.update({"dataset": row["dataset"], "method": row["method"], "label": row["label"], "resumed": True})
        results.append(result)
    audit_mode = ("completion signatures + file presence; per-crop SHA computed at write; "
                  "features stage rechecks SHA; source size/mtime checked, source bytes not rehashed")
    return _aggregate_results(index_csv, data_dir, rows, results, key, frames, audit_mode=audit_mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--detector", default="yunet")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    print(extract_faces(args.index, args.out, args.frames, args.workers, args.detector, args.limit))
