from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pandas as pd
from omegaconf import OmegaConf
from fresco.utils import atomic_json, environment
from .provenance import workspace_provenance


def record_blocked(out, cfg, label, reason, seed_keys, input_paths):
    out = Path(out)
    if (out / "manifest.json").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    status = "BLOCKED_DEPENDENCY" if "BLOCKED_DEPENDENCY" in reason else "BLOCKED_INPUT"
    identity = {"analysis": label, "status": status, "reason": reason,
                "input_paths": {key: str(value) if value else None for key, value in input_paths.items()},
                "config": OmegaConf.to_container(cfg, resolve=True), "environment": environment(),
                "workspace": workspace_provenance()}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    atomic_json(out / "manifest.json", {"signature": signature, "identity": identity,
                                        "scientific_run_started": False})
    OmegaConf.save(cfg, out / "resolved.yaml", resolve=True)
    statuses = {str(key): {"status": "BLOCKED", "reason": reason} for key in seed_keys}
    atomic_json(out / "seed_status.json", statuses)
    atomic_json(out / "metrics.json", {"status": status, "reason": reason, "completed_new_seeds": 0})
    pd.DataFrame([{"configuration": key, **value} for key, value in statuses.items()]).to_csv(
        out / "metrics.csv", index=False)
    (out / "REPORT.md").write_text(
        f"# {label}: {status}\n\n{reason}\n\n"
        "No scientific result was produced. Use a new run ID after correcting the input.\n\n"
        "## Limitations\n\nThis is an input/dependency gate record, not an experiment or empirical result.\n",
        encoding="utf-8")