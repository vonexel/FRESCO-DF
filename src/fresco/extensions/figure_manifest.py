from __future__ import annotations

import argparse
import importlib.metadata
import xml.etree.ElementTree as ET
from pathlib import Path
from fresco.utils import atomic_json
from .inventory import sha256


PLOTS = {
    "response": ("visuals.py", ("coverage", "fidelity_vs_support", "response_agreement", "response_components")),
    "pd_ale": ("effect_visuals.py", ("pd_ale_fidelity", "pd_ale_curves", "pd_ale_coverage")),
    "surrogates": ("surrogate_visuals.py", ("surrogate_fidelity", "surrogate_agreement", "surrogate_transfer",
                                             "surrogate_functions")),
    "representation": ("representation_visuals.py", ("covariate_effect", "factor_interaction", "transfer_gap",
                                                        "robustness")),
    "symbolic": ("symbolic_visuals_v2.py", ("symbolic_tradeoff", "symbolic_edges")),
}
ADDITIONAL_SOURCES = {"pd_ale": {"pd_ale_fidelity": ("pd_ale_coverage.csv",)}}


def write_figure_manifest(directory, analysis):
    directory = Path(directory)
    script, stems = PLOTS[analysis]
    if analysis == "symbolic" and ((directory/"symbolic_transfer.csv").exists()
                                   or (directory/"symbolic_transfer.svg").exists()):
        stems = (*stems, "symbolic_transfer")
    script_path = Path(__file__).resolve().with_name(script)
    snapshot = directory / (analysis + "_figure_generator.py")
    if snapshot.exists():
        if sha256(snapshot) != sha256(script_path):
            raise ValueError("Figure generator changed; use a new output/run directory")
    else:
        snapshot.write_bytes(script_path.read_bytes())
    records = []
    for stem in stems:
        svg, csv = directory / (stem+".svg"), directory / (stem+".csv")
        ET.parse(svg)
        extras = [directory/name for name in ADDITIONAL_SOURCES.get(analysis, {}).get(stem, ())]
        records.append({"figure": svg.name, "figure_sha256": sha256(svg), "source": csv.name,
                        "source_sha256": sha256(csv),
                        "additional_sources": [{"source": item.name, "sha256": sha256(item)} for item in extras]})
    atomic_json(directory / "figure_manifest.json", {
        "analysis": analysis, "script": script, "script_snapshot": snapshot.name,
        "script_sha256": sha256(script_path),
        "matplotlib": importlib.metadata.version("matplotlib"),
        "plotting": {"format": "vector SVG", "svg_fonttype": "none", "save_date": None,
                     "seed_and_uncertainty_labels": "see each figure and REPORT.md"},
        "figures": records})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", choices=PLOTS)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    write_figure_manifest(args.directory, args.analysis)


if __name__ == "__main__":
    main()