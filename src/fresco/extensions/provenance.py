from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from .inventory import sha256


def snapshot_code(destination, source_directory, expected_hashes):
    destination, source_directory = Path(destination), Path(source_directory)
    destination.mkdir(parents=True, exist_ok=True)
    for name, digest in expected_hashes.items():
        if Path(name).name != name or not name.endswith(".py"):
            raise ValueError("Invalid code snapshot filename")
        source, target = source_directory/name, destination/name
        if sha256(source) != digest:
            raise ValueError(f"Run source changed before snapshot: {name}")
        if target.exists():
            if sha256(target) != digest:
                raise ValueError(f"Existing run source snapshot is corrupted: {name}")
        else:
            shutil.copyfile(source, target)
            if sha256(target) != digest:
                raise ValueError(f"Failed to copy verified run source: {name}")


def workspace_provenance(directory=None):
    directory = Path(directory or Path.cwd()).resolve()
    try:
        commit = subprocess.run(["git", "-C", str(directory), "rev-parse", "--verify", "HEAD"],
                                capture_output=True, text=True, check=False, timeout=10)
        if commit.returncode:
            return {"status": "UNAVAILABLE", "commit": None, "dirty": None,
                    "reason": "Git checkout metadata or a committed HEAD is unavailable"}
        status = subprocess.run(["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=normal"],
                                capture_output=True, text=True, check=False, timeout=30)
        if status.returncode:
            return {"status": "PARTIAL", "commit": commit.stdout.strip(), "dirty": None,
                    "reason": "Git dirty state could not be read"}
        return {"status": "AVAILABLE", "commit": commit.stdout.strip(), "dirty": bool(status.stdout.strip())}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "UNAVAILABLE", "commit": None, "dirty": None, "reason": type(exc).__name__}