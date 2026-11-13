from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_files(root: Path):
    for name in ("pyproject.toml", "uv.lock", "README.md", "RUN.md", "DECISIONS.md", "Makefile"):
        path = root / name
        if path.is_file():
            yield path
    for directory, pattern in (("configs", "*.yaml"), ("docs", "*.md"),
                               ("src/fresco", "*.py"), ("scripts", "*.py"),
                               ("scripts", "*.sh"), ("tests", "*.py")):
        for path in (root / directory).rglob(pattern):
            rel = path.relative_to(root)
            if path.is_file() and "extensions" not in rel.parts and not path.name.startswith("test_extensions"):
                yield path
    output = root / "outputs"
    if output.is_dir():
        for path in output.rglob("*"):
            rel = path.relative_to(root)
            if path.is_file() and "extensions" not in rel.parts:
                yield path


def create(root: Path, destination: Path, *, project: bool):
    root, destination = root.resolve(), destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to replace baseline: {destination}")
    files = project_files(root) if project else (p for p in root.rglob("*") if p.is_file())
    records = {}
    for path in files:
        if path.resolve() == destination:
            raise ValueError("Inventory must be outside the tree being catalogued")
        records[path.relative_to(root).as_posix()] = {"size": path.stat().st_size, "sha256": sha256(path)}
    payload = {"schema": 1, "scope": "project legacy files" if project else "complete supplied root",
               "root_at_creation": str(root), "files": dict(sorted(records.items()))}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return len(records)


def verify(root: Path, inventory: Path):
    root = root.resolve()
    payload = json.loads(inventory.read_text(encoding="utf-8"))
    if payload.get("schema") != 1 or not payload.get("files"):
        raise ValueError("Invalid or empty inventory")
    errors = []
    for name, expected in payload["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe inventory path: {name}")
        path = root / relative
        if not path.is_file():
            errors.append(f"missing: {name}")
        elif path.stat().st_size != expected["size"] or sha256(path) != expected["sha256"]:
            errors.append(f"changed: {name}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("create", "verify"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--project", action="store_true", help="select historical project files and outputs")
    args = parser.parse_args()
    if args.mode == "create":
        print(f"Inventoried {create(args.root, args.inventory, project=args.project)} files")
    else:
        errors = verify(args.root, args.inventory)
        print("\n".join(errors) if errors else "Legacy inventory verified")
        if errors:
            raise SystemExit(1)


if __name__ == "__main__":
    main()