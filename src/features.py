from __future__ import annotations

import os
import io
import cv2
import time
import json
import torch
import hashlib
import warnings
import open_clip
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from PIL import Image, ImageFilter
from importlib.metadata import version
from open_clip.pretrained import download_pretrained_from_url
from .data import read_csv, resolve_under, sha256_file, write_csv, write_json
from .checkpoints import EncoderCheckpoint, resolve_encoder_checkpoint, verify_checkpoint_file


DEFAULT_CONDITIONS = ["clean", "jpeg90", "jpeg70", "jpeg50", "jpeg30", "blur1", "blur2", "blur3", "resize0.5", "resize0.25"]
COVARIATE_NAMES = ["jpeg_blockiness_proxy", "log_laplacian_variance", "face_size", "high_frequency_ratio"]
METADATA_COLUMNS = ["row", "video_id", "frame_idx", "dataset", "label", "split", "method", "scenario", "group_id", "condition"]
FEATURE_VERSION = 6


def _signature(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _save_encoder_provenance(path, signature, identity, provenance, state, recovery=None):
    record = {"schema_version": 1, "signature": signature, "weights_identity": identity,
              "encoder_provenance": provenance, "state": state}
    if recovery is not None:
        record["recovery"] = recovery
    write_json(path, {**record, "record_sha256": _signature(record)})


def _read_encoder_provenance(path, signature, identity):
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("Encoder provenance record must be an object")
    digest = record.pop("record_sha256", None)
    if (digest != _signature(record) or record.get("signature") != signature
            or record.get("weights_identity") != identity
            or not isinstance(record.get("encoder_provenance"), dict)
            or record.get("state") not in {"loaded", "encoded"}):
        raise ValueError("Encoder provenance record is corrupt or belongs to different weights/configuration")
    return record


def _manifest_encoder_provenance(manifest, signatures, identity):
    provenance = manifest.get("encoder_provenance") or {}
    if (manifest.get("signature") in signatures
            and provenance.get("checkpoint_sha256") == identity["sha256"]
            and provenance.get("projection_consistency", {}).get("passed") is True
            and provenance.get("visual_trainable_parameter_count") == 0
            and provenance.get("whole_clip_trainable_parameter_count") == 0):
        return provenance
    return None


def degrade_image(image, condition: str):
    if condition == "clean":
        return image.copy()
    if condition.startswith("jpeg"):
        quality = int(condition[4:])
        if not 1 <= quality <= 100:
            raise ValueError(f"Invalid JPEG quality: {quality}")
        stream = io.BytesIO()
        image.save(stream, format="JPEG", quality=quality, subsampling=0, optimize=False)
        stream.seek(0)
        return Image.open(stream).convert("RGB").copy()
    if condition.startswith("blur"):
        sigma = float(condition[4:])
        if sigma <= 0:
            raise ValueError("Blur sigma must be positive")
        return image.filter(ImageFilter.GaussianBlur(radius=sigma))
    if condition.startswith("resize"):
        ratio = float(condition[6:])
        if not 0 < ratio < 1:
            raise ValueError("Resize ratio must be between 0 and 1")
        original = image.size
        small = (max(1, round(original[0] * ratio)), max(1, round(original[1] * ratio)))
        return image.resize(small, Image.Resampling.BILINEAR).resize(original, Image.Resampling.BILINEAR)
    raise ValueError(f"Unknown degradation: {condition!r}")


def quality_covariates(image, face_size: float):
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    dx, dy = np.abs(np.diff(gray, axis=1)), np.abs(np.diff(gray, axis=0))
    # pixel boundaries 7 -> 8, 15 -> 16 , align to the 8x8 JPEG block grid
    bx, by = dx[:, 7::8], dy[7::8, :]
    boundary = (float(bx.mean()) + float(by.mean())) / 2
    global_difference = (float(dx.mean()) + float(dy.mean())) / 2
    blockiness = boundary / (global_difference + 1e-6)
    loglap = float(np.log1p(cv2.Laplacian(gray, cv2.CV_32F).var() * 255.0**2))
    centered = gray - gray.mean()
    power = np.abs(np.fft.fft2(centered)) ** 2
    fy, fx = np.meshgrid(np.fft.fftfreq(gray.shape[0]), np.fft.fftfreq(gray.shape[1]), indexing="ij")
    high = fx**2 + fy**2 >= .25**2
    ratio = float(power[high].sum() / (power.sum() + 1e-12))
    return np.array([blockiness, loglap, float(face_size), ratio], dtype=np.float32)


class CLIPCLSExtractor:
    def __init__(self, encoder_name: str = "ViT-L-14", pretrained: str = "openai", device: str = "cuda",
                 checkpoint: EncoderCheckpoint | None = None):
        model_cache = Path(os.environ.get("FRESCO_MODEL_CACHE", "data/models/open_clip")).resolve()
        model_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(model_cache / "huggingface"))
        os.environ.setdefault("TORCH_HOME", str(model_cache / "torch"))
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; explicitly use device=cpu for a CPU run")
        checkpoint = checkpoint or resolve_encoder_checkpoint(encoder_name, pretrained, model_cache)
        verify_checkpoint_file(checkpoint)
        pretrained_metadata = checkpoint.metadata
        if pretrained == "openai":
            checkpoint_path = Path(download_pretrained_from_url(checkpoint.identity["url"], cache_dir = str(model_cache)))
            if sha256_file(checkpoint_path) != checkpoint.identity["sha256"]:
                raise ValueError("Official OpenAI checkpoint SHA256 mismatch before loading")
            self.model = open_clip.load_openai_model(encoder_name, precision="fp32", device=self.device,
                                                     cache_dir=str(model_cache))
            self.transform = open_clip.image_transform(self.model.visual.image_size, is_train=False,
                                                       mean=self.model.visual.image_mean, std=self.model.visual.image_std)
            self.provenance = {"loader": "open_clip.load_openai_model(precision=fp32)",
                               "url": pretrained_metadata["url"], "checkpoint_sha256": sha256_file(checkpoint_path),
                               "checkpoint": str(checkpoint_path), "checkpoint_bytes": checkpoint_path.stat().st_size,
                               "activation": "QuickGELU"}
        else:
            self.model, _, self.transform = open_clip.create_model_and_transforms(
                encoder_name, pretrained=str(checkpoint.path), device=self.device, cache_dir=str(model_cache),
                force_quick_gelu=bool(pretrained_metadata.get("quick_gelu", False)),
                pretrained_hf=False, pretrained_image=False, load_weights_only=True,
                image_mean=pretrained_metadata.get("mean"), image_std=pretrained_metadata.get("std"),
                image_interpolation=pretrained_metadata.get("interpolation"),
                image_resize_mode=pretrained_metadata.get("resize_mode"))
            self.provenance = {"loader": "open_clip.create_model_and_transforms", "pretrained": pretrained,
                               "metadata": pretrained_metadata, "checkpoint": str(checkpoint.path),
                               "checkpoint_sha256": checkpoint.identity["sha256"],
                               "checkpoint_bytes": checkpoint.path.stat().st_size}
        self.provenance["weights_identity"] = checkpoint.identity
        self.model.eval().requires_grad_(False)
        self.provenance.update({"visual_parameter_count": sum(p.numel() for p in self.model.visual.parameters()),
                                "visual_trainable_parameter_count": sum(p.numel() for p in self.model.visual.parameters() if p.requires_grad),
                                "whole_clip_parameter_count": sum(p.numel() for p in self.model.parameters()),
                                "whole_clip_trainable_parameter_count": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                                "device_type": self.device.type,
                                "device_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"})
        if not hasattr(self.model.visual, "ln_post") or not hasattr(self.model.visual, "proj"):
            raise ValueError("Encoder must expose visual.ln_post and visual.proj")
        if getattr(self.model.visual, "pool_type", "tok") != "tok":
            raise ValueError("Pre-projection CLS requires token pooling")
        self.captured = None
        self.hook = self.model.visual.ln_post.register_forward_hook(self._capture)
        self.verified = False

    def _capture(self, module, inputs, output):
        self.captured = output

    def __call__(self, images):
        torch = self.torch
        batch = torch.stack([self.transform(image) for image in images]).to(self.device)
        with torch.inference_mode():
            self.captured = None
            projected = self.model.encode_image(batch, normalize=False)
            if self.captured is None:
                raise RuntimeError("CLIP ln_post hook did not run")
            features = self.captured
            if features.ndim == 3:
                features = features[:, 0]
            if features.ndim != 2 or features.shape[1] != 1024:
                raise RuntimeError(f"Expected ViT-L/14 pre-projection CLS [N,1024], got {tuple(features.shape)}")
            if not self.verified:
                reconstructed = features @ self.model.visual.proj
                if not torch.allclose(reconstructed.float(), projected.float(), rtol=2e-4, atol=2e-4):
                    raise RuntimeError("Captured LayerNorm output does not reproduce projected CLIP features")
                self.provenance["projection_consistency"] = {"passed": True, "rtol": 2e-4, "atol": 2e-4,
                                                             "max_absolute_error": float((reconstructed.float() - projected.float()).abs().max().item())}
                self.verified = True
            return features.float().cpu().numpy()

    def close(self):
        self.hook.remove()


def extract_features(faces_csv: Path, cache_dir: Path, device: str = "cuda", batch_size: int = 32,
                     conditions: list[str] | None = None, encoder_name: str = "ViT-L-14",
                     pretrained: str = "openai", recover_provenance: bool = False) -> Path:
    wall_start = time.perf_counter()
    faces_csv, cache_dir = Path(faces_csv).resolve(), Path(cache_dir).resolve()
    model_cache = Path(os.environ.get("FRESCO_MODEL_CACHE", "data/models/open_clip")).resolve()
    checkpoint = resolve_encoder_checkpoint(encoder_name, pretrained, model_cache)
    conditions = list(DEFAULT_CONDITIONS if conditions is None else conditions)
    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("conditions must be nonempty and unique")
    for condition in conditions:
        degrade_image(Image.new("RGB", (224, 224)), condition)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    faces = read_csv(faces_csv)
    if not faces:
        raise ValueError("Faces manifest is empty")
    faces_audit_path = faces_csv.with_suffix(".meta.json")
    faces_audit = json.loads(faces_audit_path.read_text(encoding="utf-8")) if faces_audit_path.is_file() else None
    if faces_audit and faces_audit.get("status_counts", {}).get("failed", 0):
        raise ValueError("Face extraction has unresolved failed videos; resume the faces stage before feature extraction")
    if len({(r["video_id"], r["frame_idx"]) for r in faces}) != len(faces):
        raise ValueError("Duplicate video/frame IDs in faces manifest")
    if "clean" not in conditions:
        raise ValueError("Conditions must include clean for source fitting and paired response curves")
    crop_digests = []
    for row in faces:
        crop = resolve_under(faces_csv.parent, row["crop_path"])
        if not crop.is_file():
            raise FileNotFoundError(row["crop_path"])
        digest = sha256_file(crop)
        if row.get("crop_sha256") and row["crop_sha256"] != digest:
            raise ValueError(f"Crop content changed: {crop}; rerun faces stage to repair it")
        crop_digests.append((row["crop_path"], digest))
    crop_stats = [(r["crop_path"], (faces_csv.parent / r["crop_path"]).stat().st_size,
                   (faces_csv.parent / r["crop_path"]).stat().st_mtime_ns) for r in faces]
    specification = {"version": 5, "faces_sha256": sha256_file(faces_csv),
                     "crop_stats_sha256": hashlib.sha256(json.dumps(crop_stats).encode()).hexdigest(),
                     "crop_content_sha256": hashlib.sha256(json.dumps(crop_digests).encode()).hexdigest(),
                     "encoder": encoder_name, "pretrained": pretrained, "conditions": conditions,
                     "activation_policy": "Match pretrained metadata quick_gelu; OpenAI CLIP uses QuickGELU, not default GELU.",
                     "checkpoint_policy": "OpenAI: original SHA-verified URL via load_openai_model(fp32); other tags: factory.",
                     "open_clip_torch_version": version("open_clip_torch"), "torch_version": torch.__version__,
                     "feature": "visual.ln_post CLS BEFORE visual.proj, unnormalized fp32, 1024 dimensions",
                     "covariates": COVARIATE_NAMES, "chunk_size": 256}
    specification["condition_policy"] = "train/val: clean only; test: all configured paired conditions"
    legacy_signature = _signature(specification)
    specification.update({"version": FEATURE_VERSION, "weights_identity": checkpoint.identity})
    signature = _signature(specification)
    cache_dir.mkdir(parents=True, exist_ok=True)
    spec_path = cache_dir / "specification.json"
    signatures = {signature}
    if spec_path.is_file():
        prior = json.loads(spec_path.read_text(encoding="utf-8"))
        if prior.get("signature") != signature:
            if prior.get("signature") != legacy_signature or checkpoint.identity["kind"] != "official_openai":
                raise ValueError(f"Cache configuration/weights changed; select a new cache directory: {cache_dir}")
            signatures.add(legacy_signature)
        elif prior.get("legacy_chunk_signature") == legacy_signature and checkpoint.identity["kind"] == "official_openai":
            signatures.add(legacy_signature)
    metadata = []
    row_faces = []
    for face_index, face in enumerate(faces):
        selected_conditions = ["clean"] if face["split"] in {"train", "val"} else conditions
        for condition in selected_conditions:
            record = {key: face.get(key, "") for key in METADATA_COLUMNS}
            record.update({"row": len(metadata), "condition": condition})
            metadata.append(record)
            row_faces.append(face_index)
    total = len(metadata)
    chunk_dir = cache_dir / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    extractor = None
    effective_batch_size = batch_size
    encoder_load_seconds = 0.
    encoder_call_seconds = 0.
    encoded_rows = 0
    previous_manifest = cache_dir / "manifest.json"
    manifest = json.loads(previous_manifest.read_text(encoding="utf-8")) if previous_manifest.is_file() else {}
    provenance_path = cache_dir / "encoder_provenance.json"
    provenance_record = _read_encoder_provenance(provenance_path, signature, checkpoint.identity)
    encoder_provenance = (provenance_record["encoder_provenance"] if provenance_record else
                          _manifest_encoder_provenance(manifest, signatures, checkpoint.identity))
    provenance_encoded = bool(provenance_record and provenance_record["state"] == "encoded")
    if encoder_provenance is not None and provenance_record is None:
        provenance_encoded = True
        _save_encoder_provenance(provenance_path, signature, checkpoint.identity, encoder_provenance,
                                 "encoded", {"source": "previous manifest with checkpoint SHA and projection evidence"})

    def chunk_paths(start):
        return tuple(chunk_dir / f"{start:09d}.{suffix}"
                     for suffix in ("features.npy", "covariates.npy", "complete.json"))

    valid_chunks = set()
    for start in range(0, total, 256):
        end = min(total, start + 256)
        chunk_f, chunk_c, marker = chunk_paths(start)
        try:
            completion = json.loads(marker.read_text(encoding="utf-8"))
            if (completion.get("signature") in signatures
                    and completion.get("feature_sha256") == sha256_file(chunk_f)
                    and completion.get("covariates_sha256") == sha256_file(chunk_c)):
                f, c = np.load(chunk_f, mmap_mode="r"), np.load(chunk_c, mmap_mode="r")
                if (f.shape == (end - start, 1024) and c.shape == (end - start, 4)
                        and f.dtype == np.float32 and c.dtype == np.float32):
                    valid_chunks.add(start)
        except (OSError, ValueError, KeyError, TypeError):
            pass

    if valid_chunks and not provenance_encoded and not recover_provenance:
        raise ValueError("Completed feature chunks lack verified durable encoder provenance. "
                         "Use features.recover_provenance=true for an explicit encoder/cached-row verification, "
                         "or choose a new cache directory.")

    def load_extractor():
        nonlocal extractor, encoder_provenance, encoder_load_seconds
        load_start = time.perf_counter()
        extractor = CLIPCLSExtractor(encoder_name, pretrained, device, checkpoint=checkpoint)
        encoder_provenance = getattr(extractor, "provenance", {"loader": "test fixture"})
        encoder_load_seconds += time.perf_counter() - load_start
        if not provenance_encoded:
            _save_encoder_provenance(provenance_path, signature, checkpoint.identity, encoder_provenance, "loaded")

    def persist_encoded(recovery=None):
        nonlocal encoder_provenance, provenance_encoded
        encoder_provenance = getattr(extractor, "provenance", {"loader": "test fixture"})
        _save_encoder_provenance(provenance_path, signature, checkpoint.identity, encoder_provenance, "encoded", recovery)
        provenance_encoded = True

    try:
        if valid_chunks and not provenance_encoded:
            warnings.warn("Explicitly recovering encoder provenance by loading the configured weights and "
                          "comparing a cached row; this does not independently recompute every cached row.", RuntimeWarning)
            load_extractor()
            sample = min(valid_chunks)
            face = faces[row_faces[sample]]
            with Image.open(faces_csv.parent / face["crop_path"]) as source:
                image = degrade_image(source.convert("RGB"), metadata[sample]["condition"])
            recovered = extractor([image])
            existing = np.load(chunk_paths(sample)[0], mmap_mode="r")[0]
            covariate = quality_covariates(image, float(face["face_size"]))
            existing_covariate = np.load(chunk_paths(sample)[1], mmap_mode="r")[0]
            if (np.asarray(recovered).shape != (1, 1024)
                    or not np.allclose(recovered[0], existing, rtol=2e-4, atol=2e-4)
                    or not np.allclose(covariate, existing_covariate, rtol=2e-4, atol=2e-4)):
                raise ValueError("Encoder provenance recovery failed: newly encoded row differs from cached data")
            persist_encoded({"source": "explicit fresh encoder verification", "cached_row": sample,
                             "rtol": 2e-4, "atol": 2e-4,
                             "scope": "one row compared; other chunks retain config and SHA verification"})
        stored_spec = {"signature": signature, **specification}
        if legacy_signature in signatures:
            stored_spec["legacy_chunk_signature"] = legacy_signature
        write_json(spec_path, stored_spec)
        for start in tqdm(range(0, total, 256), desc="Frozen CLIP cache", unit="chunk"):
            end = min(total, start + 256)
            chunk_f, chunk_c, marker = chunk_paths(start)
            if start in valid_chunks:
                continue
            if extractor is None:
                load_extractor()
            features = np.empty((end - start, 1024), dtype=np.float32)
            covariates = np.empty((end - start, 4), dtype=np.float32)
            position = start
            while position < end:
                batch_end = min(end, position + effective_batch_size)
                images = []
                for absolute in range(position, batch_end):
                    face = faces[row_faces[absolute]]
                    with Image.open(faces_csv.parent / face["crop_path"]) as source:
                        modified = degrade_image(source.convert("RGB"), metadata[absolute]["condition"])
                    images.append(modified)
                    covariates[absolute - start] = quality_covariates(modified, float(face["face_size"]))
                try:
                    encode_start = time.perf_counter()
                    features[position - start:batch_end - start] = extractor(images)
                    encoder_call_seconds += time.perf_counter() - encode_start
                    encoded_rows += batch_end - position
                    if not provenance_encoded:
                        persist_encoded()
                except torch.cuda.OutOfMemoryError:
                    if effective_batch_size == 1:
                        raise
                    effective_batch_size = max(1, effective_batch_size // 2)
                    torch.cuda.empty_cache()
                    continue
                position = batch_end
            if not np.isfinite(features).all() or not np.isfinite(covariates).all():
                raise ValueError(f"Nonfinite features/covariates in chunk beginning at {start}; no completion marker written")
            for target, values in ((chunk_f, features), (chunk_c, covariates)):
                with target.with_suffix(target.suffix + ".tmp").open("wb") as stream:
                    np.save(stream, values)
                target.with_suffix(target.suffix + ".tmp").replace(target)
            write_json(marker, {"signature": signature, "start": start, "end": end,
                                "feature_sha256": sha256_file(chunk_f), "covariates_sha256": sha256_file(chunk_c)})
    finally:
        if extractor is not None:
            extractor.close()
    for name, width in (("features", 1024), ("covariates", 4)):
        target = cache_dir / f"{name}.npy"
        mmap = np.lib.format.open_memmap(str(target.with_suffix(".npy.tmp")), mode="w+", dtype=np.float32, shape=(total, width))
        for start in range(0, total, 256):
            values = np.load(chunk_dir / f"{start:09d}.{name}.npy", mmap_mode="r")
            mmap[start:start + len(values)] = values
        mmap.flush()
        del mmap
        target.with_suffix(".npy.tmp").replace(target)
    write_csv(cache_dir / "metadata.csv", metadata, METADATA_COLUMNS)
    write_json(cache_dir / "manifest.json", {"signature": signature, **specification, "synthetic": False,
                                             "rows": total, "frames": len(faces), "feature_dim": 1024, "dtype": "float32",
                                             "effective_batch_size": effective_batch_size, "device": device,
                                             "encoder_provenance": encoder_provenance,
                                             "performance": {"stage_wall_seconds": time.perf_counter() - wall_start,
                                                             "encoder_load_seconds": encoder_load_seconds, "encoder_call_seconds": encoder_call_seconds,
                                                             "newly_encoded_rows": encoded_rows,
                                                             "encoder_rows_per_second": encoded_rows / encoder_call_seconds if encoder_call_seconds else None,
                                                             "scope": "encoder call includes OpenCLIP image transform, stack, device copy, forward and CPU synchronization; excludes video/face extraction, degradation/covariates, checkpoint load/download; stage wall includes cache validation, load/download, extraction, final arrays and hashing."},
                                             "features": "features.npy", "covariates_file": "covariates.npy", "metadata": "metadata.csv",
                                             "features_sha256": sha256_file(cache_dir / "features.npy"),
                                             "covariates_sha256": sha256_file(cache_dir / "covariates.npy"),
                                             "metadata_sha256": sha256_file(cache_dir / "metadata.csv"),
                                             "face_extraction_coverage": ({key: faces_audit.get(key) for key in
                                                                           ["videos", "crops", "status_counts", "missing_detections", "decode_failures", "empty_videos", "by_method", "limited"]}
                                                                          if faces_audit else None),
                                             "faces_audit_sha256": sha256_file(faces_audit_path) if faces_audit_path.is_file() else None,
                                             "standardization": "None here: fit feature/covariate scaler on FF++ train rows only in each experiment.",
                                             "paired_conditions": "Every TEST frame has every configured condition; train/val have clean only."})
    return cache_dir