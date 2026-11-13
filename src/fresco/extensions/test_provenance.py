from types import SimpleNamespace
import pytest
from fresco.extensions.inventory import sha256
from fresco.extensions.provenance import snapshot_code, workspace_provenance


def test_git_metadata_available_or_explicitly_unavailable(tmp_path, monkeypatch):
    responses = iter([SimpleNamespace(returncode=0, stdout="abc123\n"),
                      SimpleNamespace(returncode=0, stdout=" M README.md\n")])
    monkeypatch.setattr("fresco.extensions.provenance.subprocess.run", lambda *a, **k: next(responses))
    assert workspace_provenance(tmp_path) == {"status": "AVAILABLE", "commit": "abc123", "dirty": True}
    monkeypatch.setattr("fresco.extensions.provenance.subprocess.run",
                        lambda *a, **k: SimpleNamespace(returncode=128, stdout=""))
    result = workspace_provenance(tmp_path)
    assert result["status"] == "UNAVAILABLE" and result["commit"] is None and result["dirty"] is None


def test_source_snapshot_verifies_bytes_and_resume(tmp_path):
    source, destination = tmp_path/"source", tmp_path/"snapshot"
    source.mkdir()
    (source/"analysis.py").write_text("x = 1\n", encoding="utf-8")
    checksums = {"analysis.py": sha256(source/"analysis.py")}
    snapshot_code(destination, source, checksums)
    snapshot_code(destination, source, checksums)
    assert (destination/"analysis.py").read_bytes() == (source/"analysis.py").read_bytes()
    (destination/"analysis.py").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="snapshot is corrupted"):
        snapshot_code(destination, source, checksums)