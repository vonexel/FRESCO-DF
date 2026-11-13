"""Transfer integrity is tested independently of datasets and GPU availability."""
import importlib.util
import io
import tarfile
from pathlib import Path

import pytest


def script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def transfer_tree(tmp_path):
    root = tmp_path / "project"
    required = ["pyproject.toml", "uv.lock", "src/fresco/run.py", "configs/config.yaml",
                "scripts/server_preflight.py", "scripts/run_research_gpu.sh", "docs/UBUNTU_GPU_RUN.md"]
    for name in required:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source\n", encoding="utf-8")
    return root


def test_build_extract_and_reject_tampering(transfer_tree, tmp_path):
    bundle = script("build_transfer_bundle")
    excluded = transfer_tree / "src/fresco/__pycache__/secret.pyc"
    excluded.parent.mkdir(parents=True)
    excluded.write_bytes(b"not-transferable")
    data = transfer_tree / "data/personal.csv"
    data.parent.mkdir()
    data.write_text("not-transferable", encoding="utf-8")
    archive = tmp_path / "bundle.tar.gz"
    result = bundle.build(transfer_tree, archive, "project")
    assert result["status"] == "VERIFIED"
    assert archive.with_name(archive.name + ".sha256").is_file()
    extracted = tmp_path / "extracted"
    bundle.verify_archive(archive, extract_to=extracted)
    assert bundle.verify_directory(extracted / "project")["files"] == 7
    assert not (extracted / "project/data").exists()
    assert not (extracted / "project/src/fresco/__pycache__").exists()
    (extracted / "project/src/fresco/run.py").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or modified"):
        bundle.verify_directory(extracted / "project")
    with pytest.raises(FileExistsError):
        bundle.build(transfer_tree, archive, "project")


@pytest.mark.parametrize("name", ["../escape", "/absolute", "project/../escape", "project\\escape"])
def test_archive_rejects_path_traversal_before_extraction(tmp_path, name):
    bundle = script("build_transfer_bundle")
    archive = tmp_path / "malformed.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo(name)
        member.size = 1
        stream.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError):
        bundle.verify_archive(archive, extract_to=tmp_path / "extracted")
    assert not (tmp_path / "escape").exists()


def test_preflight_rejects_output_overlapping_source(tmp_path):
    preflight = script("server_preflight")
    source = tmp_path / "original/cache"
    with pytest.raises(ValueError, match="overlaps"):
        preflight.reject_output_overlap(source / "new-output", [source])
    with pytest.raises(ValueError, match="overlaps"):
        preflight.reject_output_overlap(source.parent, [source])
    preflight.reject_output_overlap(tmp_path / "separate-output", [source])


def test_preflight_lock_requires_linux_cuda_wheels(tmp_path):
    preflight = script("server_preflight")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nversion="0.1.1"\nrequires-python=">=3.11,<3.12"\n', encoding="utf-8")
    lines = ['[[package]]\nname="fresco"\nversion="0.1.1"\n']
    for name, version in [("torch", "2.6.0+cu124"), ("torchvision", "0.21.0+cu124")]:
        lines.append(f'[[package]]\nname="{name}"\nversion="{version}"\n'
                     'source={registry="https://download.pytorch.org/whl/cu124"}\n'
                     'wheels=[{url="https://example.test/cp311-cp311-linux_x86_64.whl"}]\n')
    path = tmp_path / "uv.lock"
    path.write_text("\n".join(lines), encoding="utf-8")
    assert preflight.check_lock(tmp_path)["locked"]["torch"] == "2.6.0+cu124"
    path.write_text(path.read_text(encoding="utf-8").replace("linux_x86_64", "win_amd64"), encoding="utf-8")
    with pytest.raises(ValueError, match="Linux"):
        preflight.check_lock(tmp_path)


def test_shell_has_lf_and_no_absolute_windows_paths():
    path = Path(__file__).resolve().parents[1] / "scripts/run_research_gpu.sh"
    content = path.read_bytes()
    assert b"\r\n" not in content
    assert b"E:\\" not in content
    assert content.startswith(b"#!/usr/bin/env bash\n")
