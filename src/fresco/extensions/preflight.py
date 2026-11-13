from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from .inventory import sha256


SEEDS = list(range(10))
HEADS = ("kan_spline", "mlp_gelu")
ARTIFACTS = ("features.npy", "covariates.npy", "metadata.csv")


def _id_hash(values):
    return hashlib.sha256("\n".join(sorted(map(str, values))).encode()).hexdigest()


def scan_metadata(path: Path, conditions):
    """Check split, label, frame pairs and provenance using bounded CSV chunks."""
    required = ["row", "dataset", "split", "video_id", "frame_idx", "condition", "label", "method", "group_id"]
    condition_bits = {name: 1 << i for i, name in enumerate(conditions)}
    if "clean" not in condition_bits or len(condition_bits) != len(conditions):
        raise ValueError("Cache conditions must be unique and include clean")
    pairs, videos, groups = {}, {}, {}
    counts = Counter()
    offset = 0
    for part in pd.read_csv(path, usecols=required, keep_default_na=False, chunksize=131072):
        if not np.array_equal(part.row.to_numpy(), np.arange(offset, offset+len(part))):
            raise ValueError("Cache metadata row indices differ from array order")
        offset += len(part)
        if not part.label.isin([0, 1]).all():
            raise ValueError("Labels are not binary")
        if (part.group_id == "").any():
            raise ValueError("Missing source/video group identifier")
        if not ((part.label == 0) == (part.method == "real")).all():
            raise ValueError("Label/method polarity mismatch: expected real=0 and fake=1")
        if ((part.dataset != "ffpp") & (part.split != "test")).any():
            raise ValueError("External data found outside test")
        if not part.condition.isin(conditions).all():
            raise ValueError("Unknown transformation in metadata")
        # id's are only returned as hashes
        for row in part.itertuples(index=False):
            identity = (row.dataset, row.video_id)
            provenance = (row.split, int(row.label), row.method, row.group_id)
            old = videos.setdefault(identity, provenance)
            if old != provenance:
                raise ValueError("A video has inconsistent label/split/method/group provenance")
            if row.dataset == "ffpp":
                prior = groups.setdefault(row.group_id, row.split)
                if prior != row.split:
                    raise ValueError("Source group spans FF++ splits")
            key = (row.dataset, row.video_id, int(row.frame_idx))
            bit = condition_bits[row.condition]
            previous = pairs.get(key, 0)
            if previous & bit:
                raise ValueError("Duplicated frame-condition pair")
            pairs[key] = previous | bit
            counts[(row.dataset, row.split, row.condition)] += 1
    full = sum(condition_bits.values())
    clean = condition_bits["clean"]
    for key, flags in pairs.items():
        split = videos[key[:2]][0]
        expected = full if split == "test" else clean
        if flags != expected:
            raise ValueError(f"Missing or extra conditions for {split} frame")
    video_sets = {}
    group_sets = {}
    for (dataset, video), (split, _label, _method, group) in videos.items():
        name = f"{dataset}/{split}"
        video_sets.setdefault(name, set()).add(video)
        group_sets.setdefault(name, set()).add(group)
    return {"rows": offset, "frames": len(pairs), "videos": len(videos),
            "counts": [{"dataset": a, "split": b, "condition": c, "rows": n}
                       for (a, b, c), n in sorted(counts.items())],
            "video_hashes": {k: _id_hash(v) for k, v in sorted(video_sets.items())},
            "group_hashes": {k: _id_hash(v) for k, v in sorted(group_sets.items())},
            "group_counts": {k: len(v) for k, v in sorted(group_sets.items())},
            "polarity": "0=real, 1=fake", "ffpp_group_split_overlap": False,
            "pairing": "each cached clean test frame has every declared condition",
            "group_limit": "FF++ groups are source-video pairs, not verified identities; Celeb donor identities may recur"}


def inspect(cache: Path, main: Path, *, source_cache: Path | None = None,
            replay_provenance: Path | None = None, full_hash=False, scan=False):
    cache, main = Path(cache).resolve(), Path(main).resolve()
    result = {"status": "INCOMPLETE", "issues": [], "warnings": [], "teachers": [],
              "cache": str(cache), "main": str(main)}
    issues = result["issues"]
    for path in (cache / "manifest.json", main / "manifest.json"):
        if not path.is_file():
            issues.append(f"Missing {path}")
    if issues:
        return result
    cache_manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    main_manifest = json.loads((main / "manifest.json").read_text(encoding="utf-8"))
    declared = main_manifest.get("config", {}).get("experiment", {}).get("seeds")
    if declared != SEEDS:
        issues.append(f"Original teacher seeds must be exactly 0..9; got {declared}")
    result["cache_manifest_sha256"] = sha256(cache / "manifest.json")
    result["main_manifest_sha256"] = sha256(main / "manifest.json")
    result["array_checks"] = {}
    for name in ARTIFACTS:
        p = cache / name
        key = name.split(".")[0] + "_sha256"
        expected = main_manifest.get("cache", {}).get(name)
        recorded = cache_manifest.get(key)
        status = "RECORDED_MATCH" if expected and expected == recorded and p.is_file() else "MISMATCH_OR_MISSING"
        if status != "RECORDED_MATCH":
            issues.append(f"Cache manifest/original experiment mismatch for {name}")
        elif full_hash:
            status = "VERIFIED" if sha256(p) == expected else "BYTE_HASH_MISMATCH"
            if status != "VERIFIED":
                issues.append(f"Cache bytes mismatch for {name}")
        result["array_checks"][name] = {"status": status, "expected_sha256": expected}
    common = main / "k16_cov1" / "heldout_none"
    coordinates = common / "coordinates.npz"
    fit = common / "fit_provenance.json"
    if not coordinates.is_file() or not fit.is_file():
        issues.append("Original coordinates or fit provenance missing")
    else:
        result["coordinates_sha256"] = sha256(coordinates)
        provenance = json.loads(fit.read_text(encoding="utf-8"))
        result["coordinate_fit"] = {key: provenance.get(key) for key in ("dataset", "split", "condition", "held_out_method")}
        if (provenance.get("dataset"), provenance.get("split"), provenance.get("condition")) != ("ffpp", "train", "clean"):
            issues.append("Original coordinates are not documented as FF++ train clean")
    evidence = None
    if replay_provenance is not None and Path(replay_provenance).is_file():
        evidence_path = Path(replay_provenance)
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        result["replay_evidence_sha256"] = sha256(evidence_path)
        if (evidence.get("original_manifest_sha256") != result["main_manifest_sha256"]
                or evidence.get("coordinates_sha256") != result.get("coordinates_sha256")
                or evidence.get("selected_seeds") != SEEDS
                or evidence.get("declared_seeds") != SEEDS
                or evidence.get("complete_declared_seed_coverage") is not True
                or evidence.get("retrained") is not False
                or evidence.get("original_outputs_modified") is not False
                or evidence.get("cache_hashes", evidence.get("cache")) != main_manifest.get("cache")):
            issues.append("Saved replay provenance is not bound to these main/cache/coordinates/seeds")
            evidence = None
    else:
        result["warnings"].append("Legacy checkpoints need verified all-frame replay evidence or a new replay")
    evidence_runs = {(r.get("run", {}).get("head"), r.get("run", {}).get("seed")): r
                     for r in (evidence or {}).get("runs", [])}
    for head in HEADS:
        for seed in SEEDS:
            path = common / head / "params" / f"seed{seed}"
            entry = {"head": head, "seed": seed, "status": "BLOCKED"}
            try:
                marker = json.loads((path / "DONE.json").read_text(encoding="utf-8"))
                metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
                identity = metrics["run"]
                expected = {"head": head, "seed": seed, "budget": "params", "k": 16,
                            "covariates": True, "held_out": None}
                if any(identity.get(k) != v for k, v in expected.items()):
                    raise ValueError("run identity mismatch")
                for name in ("checkpoint.pt", "metrics.json"):
                    if sha256(path / name) != marker["files"][name]:
                        raise ValueError(f"checksum mismatch: {name}")
                entry["checkpoint_sha256"] = marker["files"]["checkpoint.pt"]
                replay = evidence_runs.get((head, seed), {})
                reproduced = replay.get("original_logit_reproduction", replay.get("replay_validation", {}))
                if (replay.get("checkpoint_sha256") == entry["checkpoint_sha256"]
                        and reproduced.get("evidence") == "all saved test-frame logits reproduced"
                        and reproduced.get("frames", 0) > 0):
                    entry["status"] = "CHECKSUM_AND_REPLAY_VERIFIED"
                    entry["replayed_test_frame_conditions"] = reproduced["frames"]
                else:
                    entry["status"] = "CHECKSUM_VERIFIED_REPLAY_NOT_VERIFIED"
            except (OSError, KeyError, ValueError, TypeError) as exc:
                entry["reason"] = str(exc)
                issues.append(f"{head}/seed{seed}: {exc}")
            result["teachers"].append(entry)
    if any(item["status"] != "CHECKSUM_AND_REPLAY_VERIFIED" for item in result["teachers"]):
        result["warnings"].append("One or more teacher/coordinate bindings lack full saved replay evidence")
    policy = cache_manifest.get("condition_policy", "")
    result["original_source_condition_policy"] = policy
    if source_cache is None or not (Path(source_cache) / "manifest.json").is_file():
        result["source_transformed_cache"] = {"status": "MISSING", "path": str(source_cache) if source_cache else None}
    else:
        source_cache = Path(source_cache).resolve()
        if source_cache == cache:
            issues.append("Transformed source cache must be separate from the historical cache")
        source_manifest = json.loads((source_cache / "manifest.json").read_text(encoding="utf-8"))
        result["source_transformed_cache"] = {"status": "PRESENT_UNVERIFIED", "path": str(source_cache),
                                              "manifest_sha256": sha256(source_cache / "manifest.json"),
                                              "condition_policy": source_manifest.get("condition_policy")}
        result["warnings"].append("Source-transformed cache presence does not establish pairing or exact signature")
    if scan:
        result["metadata_audit"] = scan_metadata(cache / "metadata.csv", cache_manifest["conditions"])
        expected_replay = sum(item["rows"] for item in result["metadata_audit"]["counts"]
                              if item["split"] == "test")
        for entry in result["teachers"]:
            if (entry["status"] == "CHECKSUM_AND_REPLAY_VERIFIED"
                    and entry.get("replayed_test_frame_conditions") != expected_replay):
                entry["status"] = "CHECKSUM_VERIFIED_REPLAY_COUNT_MISMATCH"
                issues.append(f"{entry['head']}/seed{entry['seed']}: replay count differs from full test metadata")
    else:
        result["warnings"].append("Metadata split/pair audit not run")
    if not full_hash:
        result["warnings"].append("Array bytes not hashed; matching manifest declarations are not byte verification")
    result["status"] = "FAILED" if issues else ("VERIFIED" if not result["warnings"] else "INCOMPLETE")
    return result