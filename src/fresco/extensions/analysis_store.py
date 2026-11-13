from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from omegaconf import OmegaConf

from fresco.utils import atomic_json, environment

from .inventory import sha256
from .provenance import snapshot_code, workspace_provenance


def create_analysis_run(cfg, inputs, run_id, analysis, *, extra=None):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Invalid run-id")
    if analysis not in {"pd_ale", "surrogates", "symbolic"}:
        raise ValueError("Unknown explanation analysis")
    root = (Path.cwd() / "outputs" / "extensions" / "explanations").resolve()
    out = (root / run_id / analysis).resolve()
    if not out.is_relative_to(root) or any(out.is_relative_to(p) or p.is_relative_to(out)
                                          for p in (inputs.cache, inputs.main)):
        raise ValueError("Explanation output overlaps a protected input")
    code_dir = Path(__file__).resolve().parent
    code_names = {"__main__.py", "runtime.py", "analysis_store.py", "preflight.py", "paired.py", "inventory.py",
                  "figure_manifest.py", "provenance.py"}
    code_names |= ({"effects.py", "effect_run.py", "effect_visuals.py", "response_run.py"}
                   if analysis == "pd_ale" else
                   {"symbolic.py", "symbolic_run.py", "symbolic_model.py", "symbolic_eval.py",
                    "symbolic_visuals_v2.py"}
                   if analysis == "symbolic" else
                   {"surrogates.py", "surrogate_run.py", "ebm_worker.py", "source_pairs.py", "source_cache.py",
                    "surrogate_visuals.py", "surrogate_report.py"})
    identity = {"analysis": analysis, "config": OmegaConf.to_container(cfg, resolve=True),
                "original_manifest_sha256": inputs.preflight["main_manifest_sha256"],
                "coordinates_sha256": inputs.preflight["coordinates_sha256"],
                "cache_hashes": {k: v["expected_sha256"] for k, v in inputs.preflight["array_checks"].items()},
                "teacher_hashes": [r["checkpoint_sha256"] for r in inputs.preflight["teachers"]],
                "code": {p.name: sha256(p) for p in sorted(code_dir.glob("*.py")) if p.name in code_names},
                "environment": environment(), "extra": extra or {}}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest = out / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")).get("signature") != signature:
            raise ValueError("Existing explanation run differs; select a new run-id")
    else:
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(manifest, {"signature": signature, "identity": identity, "workspace": workspace_provenance(),
                               "legacy_verdict": "NOT_SUPPORTED; unchanged"})
        OmegaConf.save(cfg, out / "resolved.yaml", resolve=True)
        atomic_json(out / "preflight.json", inputs.preflight)
    snapshot_code(out/"source_snapshot", code_dir, identity["code"])
    return out, signature