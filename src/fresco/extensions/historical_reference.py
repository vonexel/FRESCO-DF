from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
from fresco.utils import atomic_json
from .inventory import sha256


def load_full_feature_reference(experiment):
    experiment = Path(experiment).resolve()
    stage = experiment/"report_stage"
    marker = json.loads((stage/"DONE.json").read_text(encoding="utf-8"))
    artifacts = stage/"artifacts.json"
    if sha256(artifacts) != marker.get("files", {}).get("artifacts.json"):
        raise ValueError("Historical report-stage completion hash does not match")
    files = json.loads(artifacts.read_text(encoding="utf-8"))
    table = experiment/"results_summary.csv"
    if sha256(table) != files.get("results_summary.csv"):
        raise ValueError("Historical full-feature reference table failed its checksum")
    result = pd.read_csv(table)
    mask = ((result.budget == "params") & (result.k == 1024) & result["head"].isin(["linear", "mlp_gelu"])
            & result.dataset.isin(["ffpp", "celebdfpp"]) & (result.condition == "clean")
            & (result.level == "video") & (result.method == "all") & (result.calibration == "raw"))
    selected = result.loc[mask, ["head", "covariates", "dataset", "auc_mean", "auc_std", "seeds", "params"]].copy()
    if (len(selected) != 8 or not selected.seeds.eq(10).all()
            or selected.duplicated(["head", "covariates", "dataset"]).any()):
        raise ValueError("Historical full-feature context is not the complete 2×2×2 ten-seed table")
    selected["role"] = "historical_reference_only_not_factorial"
    return selected, {"status": "VERIFIED_HISTORICAL_REFERENCE", "experiment": str(experiment),
                      "experiment_manifest_sha256": sha256(experiment/"manifest.json"),
                      "report_stage_done_sha256": sha256(stage/"DONE.json"),
                      "report_artifacts_sha256": sha256(artifacts), "table_sha256": sha256(table)}


def discover_from_main(main):
    root = Path(main).resolve().parent.parent/"full_features"
    latest = root/"LATEST.json"
    if not latest.is_file():
        return None, {"status": "NOT_AVAILABLE", "reason": "No historical full-feature LATEST.json"}
    try:
        key = json.loads(latest.read_text(encoding="utf-8"))["key"]
        if not isinstance(key, str) or not key or Path(key).name != key:
            raise ValueError("Invalid historical full-feature run key")
        return load_full_feature_reference(root/key)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        return None, {"status": "INVALID_DO_NOT_USE", "reason": f"{type(exc).__name__}: {exc}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = (Path.cwd()/"outputs"/"extensions"/"representation").resolve()
    out = args.output.resolve()
    if not out.is_relative_to(root) or out == root or out.exists():
        raise ValueError("Choose a new output directory under outputs/extensions/representation")
    table, audit = load_full_feature_reference(args.experiment)
    out.mkdir(parents=True)
    table.to_csv(out/"historical_full_feature_reference.csv", index=False)
    atomic_json(out/"manifest.json", {**audit, "scientific_training": False,
                                      "output_table_sha256": sha256(out/"historical_full_feature_reference.csv")})
    (out/"REPORT.md").write_text(
        "# Historical full-feature references\n\n"
        "Checksum-verified existing 1024-dimensional linear and GELU probes are shown as context only; "
        "they are not members of the matched k16/k128 factorial design. No training was performed.\n\n"
        "## Limitations\n\nA difference in feature dimension is not a pure causal estimate of information loss.\n",
        encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()