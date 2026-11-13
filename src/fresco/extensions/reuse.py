from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from fresco.utils import environment
from .inventory import sha256
from .preflight import SEEDS


HEADS = ("kan_spline", "mlp_gelu")
CONDITIONS = ["clean", "jpeg70", "blur1", "resize0.5"]


def source_fingerprint(directory: Path):
    hashes = {p.name: sha256(p) for p in sorted(directory.glob("*.py"))}
    if not hashes:
        raise ValueError(f"No Python source files in {directory}")
    return hashlib.sha256(json.dumps(hashes, sort_keys=True, default=str).encode()).hexdigest()[:20]


def inspect_candidate(experiment: Path, cache: Path | None, code: Path, regime: str):
    experiment = Path(experiment).resolve()
    output = {"status": "BLOCKED_INPUT", "experiment": str(experiment), "regime": regime,
              "reasons": [], "completed": [], "missing_or_invalid": []}
    reasons = output["reasons"]
    if regime not in {"clean", "augmented"}:
        raise ValueError("Regime must be clean or augmented")
    manifest_path = experiment / "manifest.json"
    if not manifest_path.is_file():
        reasons.append("Historical manifest missing")
        return output
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    cfg = record.get("config", {})
    exp, train, model = cfg.get("experiment", {}), cfg.get("train", {}), cfg.get("model", {})
    expected = {
        "experiment.name": (exp.get("name"), "response_" + regime),
        "experiment.synthetic": (exp.get("synthetic"), False),
        "experiment.seeds": (exp.get("seeds"), SEEDS),
        "experiment.heads": (exp.get("heads"), list(HEADS)),
        "experiment.ks": (exp.get("ks"), [16]),
        "experiment.covariates": (sorted(exp.get("covariates", [])), [False, True]),
        "experiment.budgets": (exp.get("budgets"), ["params"]),
        "experiment.held_out_methods": (exp.get("held_out_methods"), [None]),
        "experiment.source_conditions": (exp.get("source_conditions"), CONDITIONS),
        "train.conditions": (train.get("conditions"), ["clean"] if regime == "clean" else CONDITIONS),
        "train.target_params": (train.get("target_params"), 1585),
        "train.epochs": (train.get("epochs"), 30),
        "train.patience": (train.get("patience"), 6),
        "train.device": (train.get("device"), "cuda"),
        "train.target_hidden": (train.get("target_hidden"), 72),
        "train.threads": (train.get("threads"), 4),
        "train.wall_seconds": (train.get("wall_seconds"), 60),
        "train.covariate_indices": (train.get("covariate_indices"), None),
        "train.covariate_scaling": (train.get("covariate_scaling"), "standard"),
        "train.scaling_conditions": (train.get("scaling_conditions"), None),
        "train.range_quantile": (train.get("range_quantile"), .995),
        "train.range_limit": (train.get("range_limit"), 2.5),
        "train.max_steps": (train.get("max_steps"), 3000),
        "train.validation_every_steps": (train.get("validation_every_steps"), 100),
        "train.batch_size": (train.get("batch_size"), 1024),
        "train.lrs": (train.get("lrs"), [0.001, 0.0003, 0.003]),
        "train.weight_decay": (train.get("weight_decay"), 0.0001),
        "model": (model, {"grid_size": 5, "spline_degree": 3, "grid_min": -3.0, "grid_max": 3.0}),
    }
    output["config_checks"] = {name: {"actual": actual, "expected": wanted, "match": actual == wanted}
                               for name, (actual, wanted) in expected.items()}
    reasons.extend(f"Configuration mismatch: {name}" for name, value in output["config_checks"].items()
                   if not value["match"])
    try:
        actual_source = source_fingerprint(Path(code))
        output["source"] = {"saved": record.get("source"), "candidate": actual_source}
        if actual_source != record.get("source"):
            reasons.append("Source fingerprint mismatch")
    except (OSError, ValueError) as exc:
        reasons.append(f"Cannot verify source fingerprint: {exc}")
    output["dependency_checks"] = {}
    for name in ("uv.lock", "pyproject.toml"):
        path = Path.cwd() / name
        saved = record.get("dependency_files", {}).get(name)
        actual = sha256(path) if path.is_file() else None
        output["dependency_checks"][name] = {"saved": saved, "actual": actual}
        if not saved or saved != actual:
            reasons.append(f"Dependency-file mismatch: {name}")
    recorded_runtime = record.get("runtime", {})
    current_runtime = environment()
    package_names = ("torch", "numpy", "scipy", "pandas", "scikit-learn", "hydra-core", "open-clip-torch")
    differences = {n: [recorded_runtime.get("packages", {}).get(n), current_runtime["packages"].get(n)]
                   for n in package_names
                   if recorded_runtime.get("packages", {}).get(n) != current_runtime["packages"].get(n)}
    if recorded_runtime.get("python") != current_runtime.get("python"):
        differences["python"] = [recorded_runtime.get("python"), current_runtime.get("python")]
    if recorded_runtime.get("gpu") != current_runtime.get("gpu"):
        differences["gpu"] = [recorded_runtime.get("gpu"), current_runtime.get("gpu")]
    if recorded_runtime.get("cuda") != current_runtime.get("cuda"):
        differences["cuda"] = [recorded_runtime.get("cuda"), current_runtime.get("cuda")]
    output["runtime_differences"] = differences
    if differences:
        reasons.append("Runtime package/Python versions differ")
    output["cache_checks"] = {}
    for name in ("features.npy", "covariates.npy", "metadata.csv"):
        path = Path(cache) / name if cache else None
        saved = record.get("cache", {}).get(name)
        actual = sha256(path) if path is not None and path.is_file() else None
        output["cache_checks"][name] = {"saved": saved, "actual": actual}
        if not saved or saved != actual:
            reasons.append(f"Cache absent or byte hash differs: {name}")
    if cache is not None and (Path(cache)/"manifest.json").is_file():
        current_manifest = json.loads((Path(cache)/"manifest.json").read_text(encoding="utf-8"))
        saved_manifest = record.get("cache_provenance", {})
        for field in ("signature", "encoder", "pretrained", "feature", "covariates", "source_conditions"):
            if current_manifest.get(field) != saved_manifest.get(field) or field not in current_manifest:
                reasons.append(f"Cache feature provenance mismatch: {field}")
    for covariates in (False, True):
        for head in HEADS:
            common = experiment / f"k16_cov{int(covariates)}" / "heldout_none" / head / "params"
            try:
                hpo = json.loads((common / "hpo.json").read_text(encoding="utf-8"))
                marker = json.loads((common / "DONE.json").read_text(encoding="utf-8"))
                if sha256(common / "hpo.json") != marker["files"]["hpo.json"]:
                    raise ValueError("HPO checksum mismatch")
                lr = hpo["best_lr"]
                if lr not in [0.001, 0.0003, 0.003]:
                    raise ValueError("Selected learning rate is outside fixed HPO")
                if (hpo.get("selection_seed") != 0 or len(hpo.get("trials", [])) != 3
                        or [t.get("lr") for t in hpo["trials"]] != [0.001, 0.0003, 0.003]):
                    raise ValueError("Historical source-validation HPO budget differs")
            except (OSError, KeyError, ValueError, TypeError) as exc:
                output["missing_or_invalid"].append(f"{head}/cov{int(covariates)} HPO: {exc}")
                continue
            for seed in SEEDS:
                path = common / f"seed{seed}"
                identity = {"head": head, "seed": seed, "covariates": covariates, "k": 16,
                            "held_out": None, "budget": "params", "lr": lr}
                try:
                    marker = json.loads((path / "DONE.json").read_text(encoding="utf-8"))
                    details = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
                    if any(details["run"].get(key) != value for key, value in identity.items()):
                        raise ValueError("run identity/HPO mismatch")
                    for name in ("checkpoint.pt", "metrics.json"):
                        if sha256(path / name) != marker["files"][name]:
                            raise ValueError(f"checksum mismatch: {name}")
                    output["completed"].append(identity)
                except (OSError, KeyError, ValueError, TypeError) as exc:
                    output["missing_or_invalid"].append(f"{head}/cov{int(covariates)}/seed{seed}: {exc}")
    if len(output["completed"]) != 40 or output["missing_or_invalid"]:
        reasons.append("Historical factorial recipe is not complete and verified for all 40 runs")
    output["status"] = "POSSIBLE_MATCH_REVIEW_REQUIRED" if not reasons else "NOT_VERIFIED_DO_NOT_RETRAIN_AUTOMATICALLY"
    output["limitation"] = ("Exact matching applies only to this predeclared k16 recipe; "
                            "does not make k128 or other recipes equivalent.")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--cache", type=Path, help="the historical transformed source cache")
    parser.add_argument("--historical-code", type=Path, required=True, help="source directory containing legacy *.py")
    parser.add_argument("--regime", choices=("clean", "augmented"), required=True)
    args = parser.parse_args(argv)
    result = inspect_candidate(args.experiment, args.cache, args.historical_code, args.regime)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "POSSIBLE_MATCH_REVIEW_REQUIRED" else 2


if __name__ == "__main__":
    raise SystemExit(main())