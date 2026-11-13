from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd

from .source_pairs import load_source_pairs
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from .preflight import SEEDS, inspect
from fresco.utils import atomic_json
from .blocked import record_blocked
from .surrogate_run import ebm_environment
from .response_run import read_historical_curves
from .representation_run import validate_reuse_candidates, validate_saved_factorial_artifacts
from .factorial_cache import audit_factorial_cache
from .representation_run import run_representation
from .response_run import run_response
from .effect_run import run_pd_ale
from .surrogate_run import run_surrogates
from .symbolic_run import run_symbolic
from .symbolic_eval import evaluate_symbolic





def _config_dir():
    package = Path(__file__).resolve().parents[1] / "configs" / "extensions"
    project = Path(__file__).resolve().parents[3] / "configs" / "extensions"
    return project if project.is_dir() else package


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("representation", "response", "explanations"))
    parser.add_argument("--analysis", choices=("pd_ale", "surrogates", "symbolic"))
    parser.add_argument("--cache", type=Path, default=Path(os.getenv("FRESCO_CACHE", "data/cache/clip_vitl14")))
    parser.add_argument("--main", type=Path, default=os.getenv("FRESCO_MAIN_EXPERIMENT"))
    parser.add_argument("--source-cache", type=Path, default=os.getenv("FRESCO_SOURCE_CACHE")
                        or os.getenv("FRESCO_RESPONSE_CACHE"))
    parser.add_argument("--replay-provenance", type=Path, default=os.getenv("FRESCO_REPLAY_PROVENANCE"))
    parser.add_argument("--curve-dir", type=Path, default=os.getenv("FRESCO_CURVES_V2"))
    parser.add_argument("--device", default=os.getenv("FRESCO_DEVICE", "cuda"))
    parser.add_argument("--ebm-python", type=Path, default=os.getenv("FRESCO_EBM_PYTHON"))
    parser.add_argument("--clean-run", type=Path, default=os.getenv("FRESCO_CLEAN_RUN"))
    parser.add_argument("--augmented-run", type=Path, default=os.getenv("FRESCO_AUGMENTED_RUN"))
    parser.add_argument("--historical-code", type=Path, default=os.getenv("FRESCO_HISTORICAL_CODE"))
    parser.add_argument("--symbolic-attempt", type=Path, default=os.getenv("FRESCO_SYMBOLIC_ATTEMPT"))
    parser.add_argument("--symbolic-coordinates-run", type=Path, default=os.getenv("FRESCO_SYMBOLIC_COORDINATES_RUN"))
    parser.add_argument("--historical-factorial-cache", action="store_true",
                        help="use a separately audited, numerically distinct historical cache for all factorial cells")
    parser.add_argument("--run-id", default="source_attempt_01")
    parser.add_argument("--full-check", action="store_true", help="hash full arrays and scan all metadata")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if (args.experiment == "explanations") != bool(args.analysis):
        parser.error("--analysis is required exactly for the explanations experiment")
    if args.historical_factorial_cache and args.experiment != "representation":
        parser.error("--historical-factorial-cache applies only to representation")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        parser.error("--run-id may contain only letters, digits, underscores and hyphens")
    with initialize_config_dir(version_base="1.3", config_dir=str(_config_dir())):
        cfg = compose(config_name=args.experiment)
    if list(cfg.seeds) != SEEDS:
        raise ValueError("Scientific runs require exactly the fixed ten seeds 0..9")
    status = "IMPLEMENTED"
    plan = {"experiment": args.experiment, "analysis": args.analysis, "execution_status": status,
            "seeds": SEEDS, "cache": str(args.cache), "main": str(args.main) if args.main else None,
            "source_cache": str(args.source_cache) if args.source_cache else None,
            "replay_provenance": str(args.replay_provenance) if args.replay_provenance else None,
            "curve_dir": str(args.curve_dir) if args.curve_dir else None, "device": args.device,
            "ebm_python": str(args.ebm_python) if args.ebm_python else None,
            "clean_run": str(args.clean_run) if args.clean_run else None,
            "augmented_run": str(args.augmented_run) if args.augmented_run else None,
            "historical_code": str(args.historical_code) if args.historical_code else None,
            "historical_factorial_cache": args.historical_factorial_cache,
            "symbolic_attempt": str(args.symbolic_attempt) if args.symbolic_attempt else None,
            "symbolic_coordinates_run": str(args.symbolic_coordinates_run) if args.symbolic_coordinates_run else None,
            "output": str(Path.cwd() / "outputs" / "extensions" / args.experiment / args.run_id),
            "resolved_config": OmegaConf.to_container(cfg, resolve=True)}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.main:
        print(json.dumps({**plan, "status": "BLOCKED_INPUT", "reason": "Set --main or FRESCO_MAIN_EXPERIMENT"}, indent=2))
        return 2
    if args.curve_dir is None:
        args.curve_dir = Path(args.main).resolve().parent.parent / "research_followup_20261028" / "curves_v2"
    if args.validate_only:
        factorial_reuse = None
        saved_factorial_runs = None
        checks = inspect(args.cache, args.main, source_cache=args.source_cache,
                         replay_provenance=args.replay_provenance,
                         full_hash=args.full_check, scan=args.full_check)
        if args.analysis == "surrogates":
            try:
                checks["ebm_dependency"] = {"status": "AVAILABLE", "versions": ebm_environment(args.ebm_python)}
            except (OSError, RuntimeError, ValueError) as exc:
                checks["ebm_dependency"] = {"status": "BLOCKED_DEPENDENCY", "reason": str(exc)}
                checks["issues"].append(str(exc))
        if args.experiment == "response":
            try:
                _, hashes = read_historical_curves(args.curve_dir, checks)
                checks["historical_curves"] = {"status": "VERIFIED", "count": len(hashes)}
            except (OSError, ValueError, KeyError) as exc:
                checks["issues"].append(f"Historical curves: {exc}")
        if args.experiment == "representation":
            try:
                reuse = validate_reuse_candidates(args.clean_run, args.augmented_run, args.source_cache,
                                                  args.historical_code)
                if args.historical_factorial_cache:
                    saved_factorial_runs = validate_saved_factorial_artifacts(args.clean_run, args.augmented_run,
                                                                              Path(args.main))
                    if saved_factorial_runs != 80:
                        raise ValueError("Historical factorial saved-run coverage is incomplete")
                    factorial_reuse = reuse
                checks["reuse"] = {name: report["status"] for name, report in reuse.items()}
            except (OSError, ValueError, KeyError) as exc:
                checks["issues"].append(str(exc))
        required_cache = args.experiment == "representation" or args.analysis == "surrogates"
        cache_status = checks.get("source_transformed_cache", {}).get("status")
        if required_cache and cache_status == "PRESENT_UNVERIFIED" and args.full_check:
            try:
                inputs = SimpleNamespace(cache=Path(args.cache).resolve(), preflight=checks,
                                         meta=pd.read_csv(Path(args.cache)/"metadata.csv", keep_default_na=False),
                                         features=np.load(Path(args.cache)/"features.npy", mmap_mode="r"),
                                         quality=np.load(Path(args.cache)/"covariates.npy", mmap_mode="r"))
                if args.historical_factorial_cache:
                    if factorial_reuse is None:
                        raise ValueError("Historical k16 models and saved factorial scores must be verified first")
                    source = audit_factorial_cache(inputs, args.source_cache, factorial_reuse, device=args.device)
                    source["saved_factorial_runs_verified"] = saved_factorial_runs
                    checks["source_transformed_cache"].update(status="VERIFIED_FACTORIAL_ONLY",
                        source_pairs=source["source_pairs"], manifest_sha256=source["manifest_sha256"],
                        factorial_cache_audit=source)
                    checks["limitations"] = [source["limitation"]]
                else:
                    source = load_source_pairs(inputs, args.source_cache)
                    checks["source_transformed_cache"].update(status="VERIFIED",
                        source_pairs=len(source["metadata"]), manifest_sha256=source["manifest_sha256"])
                checks["warnings"] = [warning for warning in checks.get("warnings", [])
                                      if warning != "Source-transformed cache presence does not establish pairing or exact signature"]
            except (OSError, ValueError, KeyError, TypeError, RuntimeError, IndexError) as exc:
                checks["issues"].append(f"Source pair validation: {exc}")
        if required_cache and cache_status not in {"PRESENT_UNVERIFIED", "VERIFIED"}:
            checks["issues"].append("Required separately cached source-train transformations are missing")
        checks["status"] = "FAILED" if checks["issues"] else "INCOMPLETE" if checks.get("warnings") else "VERIFIED"
        if args.historical_factorial_cache and checks["status"] == "VERIFIED":
            checks["status"] = "VERIFIED_WITH_LIMITATIONS"
        checks["implementation_status"] = status
        print(json.dumps({"plan": plan, "input_preflight": checks}, indent=2))
        return 2 if checks["issues"] else 0
    if args.experiment == "response":
        try:
            path = run_response(cfg, args.cache, Path(args.main), args.replay_provenance,
                                args.curve_dir, args.run_id, args.device)
        except Exception as exc:
            status_path = Path.cwd() / "outputs" / "extensions" / "response" / args.run_id / "seed_status.json"
            if status_path.is_file():
                statuses = json.loads(status_path.read_text(encoding="utf-8"))
                for item in statuses.values():
                    if item.get("status") == "RUNNING":
                        item.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}")
                atomic_json(status_path, statuses)
            raise
        print(f"REPORT: {path / 'REPORT.md'}")
        return 0
    if args.experiment == "representation":
        try:
            path = run_representation(cfg, args.cache, Path(args.main), args.replay_provenance,
                                      args.source_cache, args.clean_run, args.augmented_run,
                                      args.historical_code, args.run_id, args.device,
                                      historical_factorial_cache=args.historical_factorial_cache)
        except Exception as exc:
            destination = Path.cwd() / "outputs" / "extensions" / "representation" / args.run_id
            status_path = destination / "seed_status.json"
            if not status_path.is_file():
                keys = [f"k{k}/{recipe}/{head}/cov{int(cov)}/seed{seed}"
                        for k in (16, 128) for recipe in ("clean", "augmented")
                        for head in ("kan_spline", "mlp_gelu") for cov in (False, True) for seed in SEEDS]
                record_blocked(destination, cfg, "representation", f"{type(exc).__name__}: {exc}", keys,
                               {"cache": args.cache, "main": args.main, "source_cache": args.source_cache,
                                "clean_run": args.clean_run, "augmented_run": args.augmented_run,
                                "historical_code": args.historical_code})
            else:
                statuses = json.loads(status_path.read_text(encoding="utf-8"))
                for item in statuses.values():
                    if item.get("status") in {"PENDING", "RUNNING"}:
                        item.update(status="BLOCKED", reason=f"Run interrupted: {type(exc).__name__}: {exc}")
                atomic_json(status_path, statuses)
                if not (destination/"metrics.json").exists():
                    counts = {value: sum(item.get("status") == value for item in statuses.values())
                              for value in ("COMPLETED", "REUSED", "FAILED", "BLOCKED")}
                    atomic_json(destination/"metrics.json", {"status": "PARTIAL_FAILURE", "counts": counts,
                                                               "reason": str(exc)})
                    (destination/"REPORT.md").write_text(
                        "# Representation: incomplete\n\n"+str(exc)+
                        "\n\n## Limitations\n\nNo complete factorial result is claimed; "
                        "see seed_status.json before resuming.\n", encoding="utf-8")
            raise
        print(f"REPORT: {path / 'REPORT.md'}")
        return 0
    if args.experiment == "explanations" and args.analysis == "pd_ale":

        try:
            path = run_pd_ale(cfg, args.cache, Path(args.main), args.replay_provenance, args.run_id, args.device)
        except Exception as exc:
            status_path = Path.cwd() / "outputs" / "extensions" / "explanations" / args.run_id / "pd_ale" / "seed_status.json"
            if status_path.is_file():
                statuses = json.loads(status_path.read_text(encoding="utf-8"))
                for item in statuses.values():
                    if item.get("status") == "RUNNING":
                        item.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}")
                atomic_json(status_path, statuses)
            raise
        print(f"REPORT: {path / 'REPORT.md'}")
        return 0
    if args.experiment == "explanations" and args.analysis == "surrogates":
        try:
            path = run_surrogates(cfg, args.cache, Path(args.main), args.replay_provenance,
                                  args.source_cache, args.ebm_python, args.run_id, args.device)
        except Exception as exc:
            destination = Path.cwd() / "outputs" / "extensions" / "explanations" / args.run_id / "surrogates"
            status_path = destination / "seed_status.json"
            if status_path.is_file():
                statuses = json.loads(status_path.read_text(encoding="utf-8"))
                for item in statuses.values():
                    if item.get("status") == "RUNNING":
                        item.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}")
                atomic_json(status_path, statuses)
            else:
                keys = [f"{h}/seed{s}" for h in ("kan_spline", "mlp_gelu") for s in SEEDS]
                record_blocked(destination, cfg, "explanations/surrogates", f"{type(exc).__name__}: {exc}", keys,
                               {"cache": args.cache, "main": args.main, "source_cache": args.source_cache,
                                "ebm_python": args.ebm_python})
            raise
        print(f"REPORT: {path / 'REPORT.md'}")
        return 0
    if args.experiment == "explanations" and args.analysis == "symbolic":
        if args.symbolic_attempt:
            attempt = args.symbolic_attempt
        else:
            attempt = run_symbolic(cfg, args.cache, Path(args.main), args.replay_provenance,
                                   Path.cwd() / "outputs" / "extensions", args.run_id + "_source")
        try:
            path = evaluate_symbolic(cfg, args.cache, Path(args.main), args.replay_provenance,
                                     attempt, args.run_id, args.device, args.symbolic_coordinates_run)
        except Exception as exc:
            status_path = Path.cwd() / "outputs" / "extensions" / "explanations" / args.run_id / "symbolic" / "seed_status.json"
            if status_path.is_file():
                statuses = json.loads(status_path.read_text(encoding="utf-8"))
                for item in statuses.values():
                    if item.get("status") == "RUNNING":
                        item.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}")
                atomic_json(status_path, statuses)
            raise
        outcome = json.loads((path / "metrics.json").read_text(encoding="utf-8"))["status"]
        print(f"{outcome}: {path / 'REPORT.md'}")
        return 0 if outcome in {"RUN", "FULL_SYMBOLIFICATION_NOT_EVALUATED"} else 2
    print(json.dumps({**plan, "status": "BLOCKED_IMPLEMENTATION",
                      "reason": "The complete scientific orchestration and required reports are not yet implemented; "
                                "no training or result generation was started."}, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())