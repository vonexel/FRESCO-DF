"""A launch-only patch must preserve prior fixes, trained results and strict identity."""
import importlib.util
import json
from pathlib import Path

import pytest


def patcher():
    path = Path(__file__).resolve().parents[1]/"scripts"/"patch_analyze_output_path.py"
    spec = importlib.util.spec_from_file_location("analyze_patch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fixture(tmp_path):
    module = patcher()
    root = tmp_path/"project"
    files = {module.SHELL: b"#!/usr/bin/env bash\nold_preflight\n",
             module.FP32_TARGETS[0]: b"old first test\r\n",
             module.FP32_TARGETS[1]: b"old second test\n",
             "src/fresco/training.py": b"unchanged experiment\n"}
    for name, content in files.items():
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    manifest = {"schema_version": 1, "source_bytes": sum(map(len, files.values())),
                "files": {name: {"bytes": len(content), "sha256": module.sha256(content)}
                          for name, content in files.items()}}
    baseline = module._json_bytes(manifest)
    (root/module.MANIFEST).write_bytes(baseline)
    replacements = {name: b"known FP32 corrected test\n" + name.encode() for name in module.FP32_TARGETS}
    options = {"baseline_manifest": baseline, "expected_baseline_sha256": module.sha256(baseline),
               "fp32_replacements": replacements, "new_shell": b"#!/usr/bin/env bash\nnew_preflight\n"}
    return module, root, files, options


def snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("with_fp32", [False, True])
def test_apply_and_idempotence_preserve_existing_fp32_and_training(fixture, with_fp32):
    module, root, files, options = fixture
    if with_fp32:
        variants = module.manifest_variants(options["baseline_manifest"],
            options["fp32_replacements"], options["new_shell"])
        previous = next(s["source"] for s in variants.values()
                        if json.loads(s["source"]).get("applied_patches"))
        (root/module.MANIFEST).write_bytes(previous)
        for name, content in options["fp32_replacements"].items():
            (root/name).write_bytes(content)
    original_tests = {name: (root/name).read_bytes() for name in module.FP32_TARGETS}
    results = root/"outputs/response_clean/completed/checkpoint.pt"
    results.parent.mkdir(parents=True)
    results.write_bytes(b"eight-hour features and two-hour training are retained")
    first = module.apply_known_patch(root, **options)
    assert first["status"] == "PATCH_APPLIED"
    assert first["preserved_fp32_patch"] is with_fp32
    assert (root/"src/fresco/training.py").read_bytes() == files["src/fresco/training.py"]
    assert results.read_bytes() == b"eight-hour features and two-hour training are retained"
    for name in module.FP32_TARGETS:
        assert (root/name).read_bytes() == original_tests[name]
    manifest = json.loads((root/module.MANIFEST).read_bytes())
    assert len(manifest["applied_patches"]) == (2 if with_fp32 else 1)
    assert manifest["applied_patches"][-1]["patch_id"] == module.PATCH_ID
    for name, entry in manifest["files"].items():
        assert module.sha256((root/name).read_bytes()) == entry["sha256"]
    before = snapshot(root)
    assert module.apply_known_patch(root, **options)["status"] == "ALREADY_APPLIED"
    assert snapshot(root) == before


@pytest.mark.parametrize("corrupt", ["src/fresco/training.py", "tests/test_covariate_controls.py",
                                    "TRANSFER_MANIFEST.json"])
def test_unknown_file_or_manifest_is_rejected_before_any_write(fixture, corrupt):
    module, root, _, options = fixture
    (root/corrupt).write_bytes(b"unrecognized modification")
    before = snapshot(root)
    with pytest.raises(ValueError, match="Unrecognized"):
        module.apply_known_patch(root, **options)
    assert snapshot(root) == before
    assert not (root/".patch_backups").exists()


def test_committed_shell_with_original_manifest_recovers_without_rewriting_tests(fixture, monkeypatch):
    module, root, files, options = fixture
    write = module._atomic_write

    def interrupted(path, content):
        if path.name == module.MANIFEST:
            raise OSError("simulated interruption before manifest")
        write(path, content)

    monkeypatch.setattr(module, "_atomic_write", interrupted)
    with pytest.raises(OSError, match="interruption"):
        module.apply_known_patch(root, **options)
    assert (root/module.SHELL).read_bytes() == options["new_shell"]
    assert (root/module.MANIFEST).read_bytes() == options["baseline_manifest"]
    monkeypatch.setattr(module, "_atomic_write", write)
    assert module.apply_known_patch(root, **options)["status"] == "PATCH_APPLIED"
    assert (root/".patch_backups"/module.PATCH_ID/module.SHELL).read_bytes() == files[module.SHELL]
    for name in module.FP32_TARGETS:
        assert (root/name).read_bytes() == files[name]
