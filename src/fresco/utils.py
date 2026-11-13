import os
import json
import torch
import random
import hashlib
import platform
import numpy as np
from pathlib import Path
from .data import replace_file
from __future__ import annotations
from importlib.metadata import version, PackageNotFoundError


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(x):
        if isinstance(x, dict):
            return {str(k): convert(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)):
            return [convert(v) for v in x]
        if isinstance(x, np.ndarray):
            return convert(x.tolist())
        if isinstance(x, np.generic):
            return convert(x.item())
        if isinstance(x, float) and not np.isfinite(x):
            return None
        if isinstance(x, Path):
            return str(x)
        return x
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(convert(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    replace_file(tmp, path)


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:20]


def source_fingerprint() -> str:
    return fingerprint({p.name: digest_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def complete(path: Path, key: str) -> bool:
    try:
        record = json.loads((path / "DONE.json").read_text(encoding="utf-8"))
        return record["key"] == key and all(
            (path / name).is_file() and digest_file(path / name) == sha
            for name, sha in record["files"].items())
    except (OSError, KeyError, ValueError):
        return False


def mark_complete(path: Path, key: str, names: list[str]) -> None:
    atomic_json(path / "DONE.json", {"key": key, "files": {n: digest_file(path / n) for n in names}})


def environment() -> dict:
    versions={}
    for name in ["torch","numpy","scipy","pandas","scikit-learn","hydra-core","matplotlib",
                 "opencv-python-headless","open-clip-torch"]:
        try:
            versions[name]=version(name)
        except PackageNotFoundError:
            versions[name]=None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "source_sha256": source_fingerprint(), "deterministic": True,"packages":versions}