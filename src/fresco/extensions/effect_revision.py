from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from fresco.utils import atomic_json
from .effect_visuals import effect_figures
from .figure_manifest import write_figure_manifest
from .inventory import sha256


def revise(source, name):
    source = Path(source).resolve()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Invalid figure-revision name")
    root = (Path.cwd()/"outputs"/"extensions"/"explanations").resolve()
    if not source.is_relative_to(root):
        raise ValueError("Source must be an extension explanation run")
    manifest = json.loads((source/"manifest.json").read_text(encoding="utf-8"))
    statuses = json.loads((source/"seed_status.json").read_text(encoding="utf-8"))
    expected = {f"{h}/seed{s}" for h in ("kan_spline", "mlp_gelu") for s in range(10)}
    if (manifest.get("identity", {}).get("analysis") != "pd_ale" or set(statuses) != expected
            or any(x.get("status") != "COMPLETED" for x in statuses.values())):
        raise ValueError("Source is not a completed ten-seed PD/ALE series")
    figures = source/"figures"
    recorded = json.loads((figures/"figure_manifest.json").read_text(encoding="utf-8"))
    source_hashes = {x["source"]: x["source_sha256"] for x in recorded["figures"]}
    names = ("pd_ale_fidelity.csv", "pd_ale_curves.csv", "pd_ale_coverage.csv")
    if any(source_hashes.get(n) != sha256(figures/n) for n in names):
        raise ValueError("Source figure CSV failed its existing completion hash")
    out = source/"figure_revisions"/name
    if out.exists():
        raise FileExistsError("Figure revision already exists; choose a new name")
    dest = out/"figures"
    dest.mkdir(parents=True)
    for filename in names:
        shutil.copyfile(figures/filename, dest/filename)
    effect_figures(dest)
    write_figure_manifest(dest, "pd_ale")
    figure_manifest = json.loads((dest/"figure_manifest.json").read_text(encoding="utf-8"))
    for item in figure_manifest["figures"]:
        if item["figure"] == "pd_ale_fidelity.svg":
            item["additional_sources"] = [{"source": "pd_ale_coverage.csv",
                                           "sha256": sha256(dest/"pd_ale_coverage.csv")}]
    atomic_json(dest/"figure_manifest.json", figure_manifest)
    shutil.copyfile(Path(__file__), out/"effect_revision.py")
    atomic_json(out/"manifest.json", {"status": "RENDERED_FROM_VERIFIED_CSV", "seed_count_per_head": 10,
                "source_run_manifest_sha256": sha256(source/"manifest.json"),
                "source_figure_manifest_sha256": sha256(figures/"figure_manifest.json"),
                "figure_manifest_sha256": sha256(dest/"figure_manifest.json"),
                "revision_code_sha256": sha256(Path(__file__)),
                "metrics_recomputed": False})
    (out/"REPORT.md").write_text(
        "# PD/ALE figure revision\n\n"
        "Existing, checksum-verified per-seed result tables were reused. No model, metric or scientific "
        "result was recomputed. Fidelity labels now include common-support fractions. Curves are separated "
        "by teacher, show seed SD (not a confidence interval), source-bin density and regions outside "
        "the fitted support.\n\n## Limitations\n\nThe analysis remains exploratory and non-causal; "
        "see the original run REPORT.md for populations and bootstrap assumptions.\n", encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    print(revise(args.source, args.name))


if __name__ == "__main__":
    main()