from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from fresco.utils import atomic_json
from .inventory import sha256
from .preflight import HEADS, SEEDS


def repair(path: Path):
    path = Path(path).resolve()
    root = (Path.cwd() / "outputs" / "extensions" / "response").resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Status correction is restricted to a response extension run")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    signature = manifest["signature"]
    status_path = path / "seed_status.json"
    backup = path / "seed_status.before_correction.json"
    evidence_path = path / "seed_status_correction.json"
    if backup.exists() or evidence_path.exists():
        raise FileExistsError("A status correction already exists; do not overwrite it")
    original_hash = sha256(status_path)
    verified = {}
    names = ["video.csv", "strata.csv", "components.csv", "agreement.csv", "support.csv"]
    for head in HEADS:
        for seed in SEEDS:
            directory = path / head / f"seed{seed}"
            marker = json.loads((directory / "DONE.json").read_text(encoding="utf-8"))
            if marker.get("signature") != signature:
                raise ValueError(f"Seed signature mismatch: {directory}")
            hashes = {name: sha256(directory / name) for name in names}
            if any(hashes[n] != marker.get("files", {}).get(n) for n in names):
                raise ValueError(f"Seed artifact mismatch: {directory}")
            verified[f"{head}/seed{seed}"] = {"status": "COMPLETED"}
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    if metrics.get("seeds") != SEEDS or metrics.get("status") != "RUN":
        raise ValueError("Run metrics do not document all ten seeds")
    shutil.copyfile(status_path, backup)
    atomic_json(status_path, verified)
    atomic_json(evidence_path, {"reason": "A reused loop variable wrote completion to the wrong ledger key",
                                "original_sha256": original_hash, "backup_sha256": sha256(backup),
                                "corrected_sha256": sha256(status_path), "verified_seed_count": len(verified),
                                "verification": "signature and all five DONE-listed SHA256 values per seed",
                                "scientific_tables_modified": False})
    return len(verified)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(f"Corrected status after verifying {repair(args.run)} seed artifacts")


if __name__ == "__main__":
    main()