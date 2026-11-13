import json
from omegaconf import OmegaConf
from fresco.extensions.blocked import record_blocked


def test_blocked_records_seed_denominator_and_preserves_started_run(tmp_path, monkeypatch):
    monkeypatch.setattr("fresco.extensions.blocked.environment", lambda: {"python": "test"})
    cfg = OmegaConf.create({"seeds": list(range(10))})
    record_blocked(tmp_path, cfg, "fixture", "BLOCKED_INPUT: missing cache", range(10), {"cache": None})
    manifest = json.loads((tmp_path/"manifest.json").read_text())
    statuses = json.loads((tmp_path/"seed_status.json").read_text())
    assert not manifest["scientific_run_started"]
    assert len(statuses) == 10 and all(item["status"] == "BLOCKED" for item in statuses.values())
    before = (tmp_path/"manifest.json").read_bytes()
    record_blocked(tmp_path, cfg, "fixture", "BLOCKED_DEPENDENCY", range(10), {})
    assert (tmp_path/"manifest.json").read_bytes() == before