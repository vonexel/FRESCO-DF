"""Patch integrity and crash recovery do not require server data or an archive."""
import importlib.util
import json
from pathlib import Path

import pytest


def patcher():
    path = Path(__file__).resolve().parents[1]/"scripts"/"patch_linux_fp32_tests.py"
    spec = importlib.util.spec_from_file_location("linux_fp32_patch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fixture(tmp_path):
    module = patcher()
    root = tmp_path/"project"
    files = {"tests/test_covariate_controls.py": b"old first test\r\n",
             "tests/test_replay_audit.py": b"old second test\n",
             "src/fresco/training.py": b"unchanged computation\n",
             "configs/config.yaml": b"unchanged config\n"}
    for name, content in files.items():
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    manifest = {"schema_version": 1, "source_bytes": sum(map(len, files.values())),
                "files": {name: {"bytes": len(content), "sha256": module.sha256(content)}
                          for name, content in files.items()}}
    data = module._json_bytes(manifest)
    (root/module.MANIFEST).write_bytes(data)
    replacements = {name: b"corrected FP32 test\n" + name.encode() for name in module.TARGETS}
    options = {"baseline_manifest": data, "expected_baseline_sha256": module.sha256(data),
               "replacements": replacements}
    return module, root, files, options


def snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def test_apply_checks_manifest_preserves_source_and_is_idempotent(fixture):
    module, root, files, options = fixture
    first = module.apply_known_patch(root, **options)
    assert first["status"] == "PATCH_APPLIED" and first["verified_files"] == 4
    manifest = json.loads((root/module.MANIFEST).read_bytes())
    assert manifest["source_bytes"] == sum(entry["bytes"] for entry in manifest["files"].values())
    for name, entry in manifest["files"].items():
        assert module.sha256((root/name).read_bytes()) == entry["sha256"]
    for name in ["src/fresco/training.py", "configs/config.yaml"]:
        assert (root/name).read_bytes() == files[name]
    backup = root/".patch_backups"/module.PATCH_ID
    for name in module.TARGETS:
        assert (backup/name).read_bytes() == files[name]
    assert (backup/module.MANIFEST).read_bytes() == options["baseline_manifest"]
    before = snapshot(root)
    assert module.apply_known_patch(root, **options)["status"] == "ALREADY_APPLIED"
    assert snapshot(root) == before


def test_unexpected_other_file_corruption_fails_before_any_write(fixture):
    module, root, _, options = fixture
    (root/"src/fresco/training.py").write_bytes(b"unexpected experiment changes")
    before = snapshot(root)
    with pytest.raises(ValueError, match="corrupted project file"):
        module.apply_known_patch(root, **options)
    assert snapshot(root) == before
    assert not (root/".patch_backups").exists()


def test_unrecognized_version_manifest_fails_before_any_write(fixture):
    module, root, _, options = fixture
    (root/module.MANIFEST).write_text('{"schema_version":2}', encoding="utf-8")
    before = snapshot(root)
    with pytest.raises(ValueError, match="Unrecognized project manifest"):
        module.apply_known_patch(root, **options)
    assert snapshot(root) == before
    assert not (root/".patch_backups").exists()


def test_interrupted_application_recovers_without_overwriting_backups(fixture, monkeypatch):
    module, root, files, options = fixture
    write = module._atomic_write
    count = 0

    def interrupted(path, content):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("simulated interruption")
        write(path, content)

    monkeypatch.setattr(module, "_atomic_write", interrupted)
    with pytest.raises(OSError, match="interruption"):
        module.apply_known_patch(root, **options)
    assert (root/module.MANIFEST).read_bytes() == options["baseline_manifest"]
    monkeypatch.setattr(module, "_atomic_write", write)
    assert module.apply_known_patch(root, **options)["status"] == "PATCH_APPLIED"
    for name in module.TARGETS:
        assert (root/".patch_backups"/module.PATCH_ID/name).read_bytes() == files[name]
