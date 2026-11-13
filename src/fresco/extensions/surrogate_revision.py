from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from fresco.utils import atomic_json
from .figure_manifest import write_figure_manifest
from .inventory import sha256
from .surrogate_visuals import surrogate_figures


def revise(source, name):
    source = Path(source).resolve()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Invalid figure-revision name")
    root = (Path.cwd()/"outputs"/"extensions"/"explanations").resolve()
    if not source.is_relative_to(root):
        raise ValueError("Source must be an extension explanation run")
    manifest = json.loads((source/"manifest.json").read_text(encoding="utf-8"))
    statuses = json.loads((source/"seed_status.json").read_text(encoding="utf-8"))
    expected = {f"{head}/seed{seed}" for head in ("kan_spline", "mlp_gelu") for seed in range(10)}
    if (manifest.get("identity", {}).get("analysis") != "surrogates" or set(statuses) != expected
            or any(item.get("status") != "COMPLETED" for item in statuses.values())):
        raise ValueError("Source is not a completed ten-seed surrogate series")
    figures = source/"figures"
    recorded = json.loads((figures/"figure_manifest.json").read_text(encoding="utf-8"))
    source_hashes = {item["source"]: item["source_sha256"] for item in recorded["figures"]}
    names = ("surrogate_fidelity.csv", "surrogate_agreement.csv", "surrogate_transfer.csv", "surrogate_functions.csv")
    if any(source_hashes.get(filename) != sha256(figures/filename) for filename in names):
        raise ValueError("Source figure CSV failed its existing completion hash")
    out = source/"figure_revisions"/name
    if out.exists():
        raise FileExistsError("Figure revision already exists; choose a new name")
    dest = out/"figures"
    dest.mkdir(parents=True)
    for filename in names:
        shutil.copyfile(figures/filename, dest/filename)
    surrogate_figures(dest)
    write_figure_manifest(dest, "surrogates")
    shutil.copyfile(Path(__file__), out/"surrogate_revision.py")
    atomic_json(out/"manifest.json", {"status": "RENDERED_FROM_VERIFIED_CSV", "seed_count_per_head": 10,
                "source_run_manifest_sha256": sha256(source/"manifest.json"),
                "source_figure_manifest_sha256": sha256(figures/"figure_manifest.json"),
                "figure_manifest_sha256": sha256(dest/"figure_manifest.json"),
                "revision_code_sha256": sha256(Path(__file__)), "metrics_recomputed": False})
    (out/"REPORT.md").write_text(
        "# Surrogate figure revision\n\nExisting checksum-verified per-seed result tables were reused. "
        "No model, metric or scientific result was recomputed. Figure labels now name all quality features "
        "and show test-pair or video counts.\n\n## Limitations\n\nThe analysis is exploratory; "
        "source-fitted quality functions are not causal effects. See the main report for estimands and intervals.\n",
        encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    print(revise(args.source, args.name))


if __name__ == "__main__":
    main()