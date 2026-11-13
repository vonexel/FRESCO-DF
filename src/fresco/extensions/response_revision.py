from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from fresco.utils import atomic_json
from .figure_manifest import write_figure_manifest
from .inventory import sha256
from .visuals import response_figures


def revise(source, name):
    source = Path(source).resolve()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Invalid figure-revision name")
    root = (Path.cwd()/"outputs"/"extensions"/"response").resolve()
    if not source.is_relative_to(root):
        raise ValueError("Source must be an extension response run")
    manifest = json.loads((source/"manifest.json").read_text(encoding="utf-8"))
    statuses = json.loads((source/"seed_status.json").read_text(encoding="utf-8"))
    expected = {f"{h}/seed{s}" for h in ("kan_spline", "mlp_gelu") for s in range(10)}
    if (manifest.get("identity", {}).get("experiment") != "response" or set(statuses) != expected
            or any(x.get("status") != "COMPLETED" for x in statuses.values())):
        raise ValueError("Source is not a completed ten-seed response series")
    figures = source/"figures"
    recorded = json.loads((figures/"figure_manifest.json").read_text(encoding="utf-8"))
    source_hashes = {x["source"]: x["source_sha256"] for x in recorded["figures"]}
    names = ("coverage.csv", "fidelity_vs_support.csv", "response_agreement.csv", "response_components.csv")
    if any(source_hashes.get(n) != sha256(figures/n) for n in names):
        raise ValueError("Source figure CSV failed its existing completion hash")
    out = source/"figure_revisions"/name
    if out.exists():
        raise FileExistsError("Figure revision already exists; choose a new name")
    dest = out/"figures"
    dest.mkdir(parents=True)
    for filename in names:
        shutil.copyfile(figures/filename, dest/filename)
    response_figures(dest)
    write_figure_manifest(dest, "response")
    shutil.copyfile(Path(__file__), out/"response_revision.py")
    atomic_json(out/"manifest.json", {"status": "RENDERED_FROM_VERIFIED_CSV", "seed_count_per_head": 10,
                "source_run_manifest_sha256": sha256(source/"manifest.json"),
                "source_figure_manifest_sha256": sha256(figures/"figure_manifest.json"),
                "figure_manifest_sha256": sha256(dest/"figure_manifest.json"),
                "revision_code_sha256": sha256(Path(__file__)), "metrics_recomputed": False})
    (out/"REPORT.md").write_text(
        "# Response figure revision\n\nExisting, checksum-verified per-seed tables were reused. "
        "No model, metric or scientific result was recomputed. Coverage explicitly includes unsupported "
        "pairs. Component panels separate heads and datasets; wide translucent bars describe paired-frame SD, "
        "and thin error bars describe seed SD. Neither is a confidence interval.\n\n## Limitations\n\n"
        "These descriptive distributions are not causal; see the original report for matched populations "
        "and bootstrap assumptions.\n", encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    print(revise(args.source, args.name))


if __name__ == "__main__":
    main()