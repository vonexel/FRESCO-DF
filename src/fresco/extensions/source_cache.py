from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from importlib.metadata import version
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from fresco.checkpoints import resolve_encoder_checkpoint
from fresco.data import resolve_under
from fresco.features import CLIPCLSExtractor, degrade_image, quality_covariates
from fresco.utils import atomic_json
from .inventory import sha256


CONDITIONS = ("jpeg70", "blur1", "resize0.5")
CHUNK = 192


def _input_identity(cache, faces_csv):
    manifest_path = cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in ("features.npy", "covariates.npy", "metadata.csv"):
        expected = manifest.get(name.split(".")[0]+"_sha256")
        if not expected or sha256(cache/name) != expected:
            raise ValueError(f"Original cache bytes do not match manifest: {name}")
    faces_hash = sha256(faces_csv)
    if manifest.get("faces_sha256") != faces_hash:
        raise ValueError("Face manifest does not match the original feature-cache provenance")
    encoder = manifest.get("encoder_provenance", {})
    if (manifest.get("encoder") != "ViT-L-14" or manifest.get("pretrained") != "openai"
            or encoder.get("checkpoint_sha256") != "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
            or not encoder.get("projection_consistency", {}).get("passed")):
        raise ValueError("Original cache is not the verified OpenAI CLIP ViT-L/14 pre-projection cache")
    return manifest


def _source_faces(cache, faces_csv):
    use = ["row", "dataset", "video_id", "frame_idx", "label", "split", "method", "scenario", "group_id", "condition"]
    source = []
    for chunk in pd.read_csv(cache / "metadata.csv", usecols=use, keep_default_na=False, chunksize=131072):
        take = chunk.loc[(chunk.dataset == "ffpp") & chunk.split.isin(["train", "val"])
                         & (chunk.condition == "clean")]
        if not take.empty:
            source.append(take)
    if not source:
        raise ValueError("No FF++ source train/validation clean rows")
    source = pd.concat(source, ignore_index=True).rename(columns={"row": "base_row"}).drop(columns="condition")
    faces = pd.read_csv(faces_csv, keep_default_na=False)
    faces = faces.loc[(faces.dataset == "ffpp") & faces.split.isin(["train", "val"])]
    cols = ["video_id", "frame_idx", "crop_path", "crop_sha256", "face_size",
            "dataset", "split", "label", "method", "scenario", "group_id"]
    faces = faces[cols]
    keys = ["dataset", "video_id", "frame_idx"]
    result = source.merge(faces, on=keys, how="left", suffixes=("", "_face"), validate="one_to_one")
    if len(result) != len(source) or result.crop_path.isna().any():
        raise ValueError("Original source clean rows do not all have their original face crops")
    for name in ("split", "label", "method", "scenario", "group_id"):
        if not result[name].equals(result[name+"_face"]):
            raise ValueError(f"Source crop provenance mismatch for {name}")
    groups = result.groupby("group_id").split.nunique()
    if (groups > 1).any():
        raise ValueError("Source group overlaps train and validation")
    return result.drop(columns=[name+"_face" for name in ("split", "label", "method", "scenario", "group_id")])


def build_source_cache(cache: Path, faces_csv: Path, output: Path, *, device="cuda", batch_size=4):
    cache, faces_csv, output = (Path(p).resolve() for p in (cache, faces_csv, output))
    root = (Path.cwd() / "outputs" / "extensions" / "representation").resolve()
    if not output.is_relative_to(root) or output == root:
        raise ValueError("Source-pair cache must be inside outputs/extensions/representation/<run_id>/")
    if output.is_relative_to(cache) or cache.is_relative_to(output) or output.is_relative_to(faces_csv.parent):
        raise ValueError("Source-pair cache overlaps historical inputs")
    if batch_size < 1 or (device.startswith("cuda") and not torch.cuda.is_available()):
        raise ValueError("A positive batch size and an available requested device are required")
    baseline = _input_identity(cache, faces_csv)
    faces = _source_faces(cache, faces_csv)
    model_dir = Path(os.environ.get("FRESCO_MODEL_CACHE", "data/models/open_clip")).resolve()
    checkpoint = resolve_encoder_checkpoint("ViT-L-14", "openai", model_dir)
    identity = {"schema": 1, "original_manifest_sha256": sha256(cache/"manifest.json"),
                "original_cache_hashes": {name: baseline[name+"_sha256"]
                                          for name in ("features", "covariates", "metadata")},
                "faces_sha256": baseline["faces_sha256"], "conditions": list(CONDITIONS),
                "weights_identity": checkpoint.identity, "torch": torch.__version__,
                "open_clip_torch": version("open_clip_torch"), "device": device,
                "source_module_sha256": sha256(Path(__file__)), "chunk_rows": CHUNK}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    specification = output / "specification.json"
    if specification.exists():
        if json.loads(specification.read_text(encoding="utf-8")).get("signature") != signature:
            raise ValueError("Existing source cache has different provenance; choose a new run-id")
    else:
        atomic_json(specification, {"signature": signature, "identity": identity})
    expanded = faces.loc[faces.index.repeat(len(CONDITIONS))].reset_index(drop=True)
    expanded["condition"] = np.tile(CONDITIONS, len(faces))
    expanded["row"] = np.arange(len(expanded))
    chunk_dir = output / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    extractor = None
    encoder_provenance = None
    start_time = time.perf_counter()
    encoded = 0
    effective = batch_size
    try:
        for start in tqdm(range(0, len(expanded), CHUNK), desc="FF++ source-only CLIP", unit="chunk"):
            end = min(start+CHUNK, len(expanded))
            fpath, qpath = (chunk_dir / f"{start:09d}.{name}.npy" for name in ("features", "covariates"))
            done = chunk_dir / f"{start:09d}.json"
            if done.is_file():
                marker = json.loads(done.read_text(encoding="utf-8"))
                if (marker.get("signature") == signature and fpath.is_file() and qpath.is_file()
                        and marker.get("features_sha256") == sha256(fpath)
                        and marker.get("covariates_sha256") == sha256(qpath)):
                    continue
                raise ValueError(f"Corrupt or incompatible source-cache chunk: {start}")
            if extractor is None:
                extractor = CLIPCLSExtractor("ViT-L-14", "openai", device, checkpoint=checkpoint)
            embeddings = np.empty((end-start, 1024), dtype=np.float32)
            quality = np.empty((end-start, 4), dtype=np.float32)
            position = start
            while position < end:
                stop = min(end, position+effective)
                images = []
                for offset, row in enumerate(expanded.iloc[position:stop].itertuples(index=False)):
                    crop = resolve_under(faces_csv.parent, row.crop_path)
                    if sha256(crop) != row.crop_sha256:
                        raise ValueError("Source crop checksum disagrees with original face manifest")
                    with Image.open(crop) as source:
                        image = degrade_image(source.convert("RGB"), row.condition)
                    images.append(image)
                    quality[position-start+offset] = quality_covariates(image, float(row.face_size))
                try:
                    embeddings[position-start:stop-start] = extractor(images)
                except torch.cuda.OutOfMemoryError:
                    if effective == 1:
                        raise
                    effective = max(1, effective//2)
                    torch.cuda.empty_cache()
                    continue
                position = stop
                encoded += len(images)
            if not (np.isfinite(embeddings).all() and np.isfinite(quality).all()):
                raise ValueError("Nonfinite source transformed features or quality")
            encoder_provenance = extractor.provenance
            for target, values in ((fpath, embeddings), (qpath, quality)):
                temp = target.with_suffix(".npy.tmp")
                with temp.open("wb") as stream:
                    np.save(stream, values)
                temp.replace(target)
            atomic_json(done, {"signature": signature, "start": start, "end": end,
                               "features_sha256": sha256(fpath), "covariates_sha256": sha256(qpath),
                               "encoder_provenance": encoder_provenance})
    finally:
        if extractor is not None:
            extractor.close()
    for name, width in (("features", 1024), ("covariates", 4)):
        target = output / f"{name}.npy"
        temp = target.with_suffix(".npy.tmp")
        merged = np.lib.format.open_memmap(str(temp), mode="w+", dtype=np.float32, shape=(len(expanded), width))
        for start in range(0, len(expanded), CHUNK):
            values = np.load(chunk_dir / f"{start:09d}.{name}.npy", mmap_mode="r")
            merged[start:start+len(values)] = values
        merged.flush()
        del merged
        temp.replace(target)
    cols = ["row", "base_row", "dataset", "video_id", "frame_idx", "label", "split", "method", "scenario",
            "group_id", "condition"]
    expanded[cols].to_csv(output / "metadata.csv", index=False)
    first_marker = json.loads((chunk_dir / "000000000.json").read_text(encoding="utf-8"))
    provenance = encoder_provenance or first_marker.get("encoder_provenance")
    if not provenance or not provenance.get("projection_consistency", {}).get("passed"):
        raise ValueError("Source-cache chunks lack verified projection provenance")
    atomic_json(output / "manifest.json", {"signature": signature, "identity": identity,
                "source_only": True, "rows": len(expanded), "frames": len(faces),
                "conditions": list(CONDITIONS), "encoder_provenance": provenance,
                "features_sha256": sha256(output/"features.npy"),
                "covariates_sha256": sha256(output/"covariates.npy"),
                "metadata_sha256": sha256(output/"metadata.csv"),
                "elapsed_seconds_this_invocation": time.perf_counter()-start_time, "encoded_rows_this_invocation": encoded,
                "note": "No test rows or clean rows re-encoded; clean anchors remain in original cache."})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--faces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({"conditions": CONDITIONS, "cache": str(args.cache), "faces": str(args.faces),
                          "output": str(args.output), "device": args.device, "source_only": True}, indent=2))
    else:
        print(build_source_cache(args.cache, args.faces, args.output, device=args.device, batch_size=args.batch_size))


if __name__ == "__main__":
    main()