from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from fresco.utils import atomic_json
from .inventory import sha256


def _gap(report):
    value = report.get("head_a_mean")
    return f"{value:.4f}" if value is not None else "undefined"


def _contrast(report):
    value = report.get("difference")
    ci = report.get("crossed_seeds_clusters", {}).get("ci95")
    if value is None:
        return "undefined"
    return f"{value:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else f"{value:+.4f} [CI unavailable]"


def render_report(result):
    if result.get("status") != "RUN" or result.get("seeds") != list(range(10)):
        raise ValueError("A completed ten-seed surrogate summary is required")
    reports = result["contrasts"]
    rows = []
    for head in ("kan_spline", "mlp_gelu"):
        for dataset in ("ffpp", "celebdfpp"):
            r = reports[f"{head}/{dataset}"]
            rows.append(f"| {head} | {dataset} | {_gap(r['ridge_minus_zero'])} | {_gap(r['gam_minus_zero'])} | "
                        f"{_gap(r['ebm_minus_zero'])} | {_contrast(r['gam_minus_ridge'])} | "
                        f"{_contrast(r['ebm_minus_ridge'])} |")
    kan = reports["kan_spline/celebdfpp"]
    mlp = reports["mlp_gelu/celebdfpp"]
    return (
        "# Quality-only response surrogates\n\n"
        f"All ten frozen teacher seeds per head were evaluated on {result['pairs']} identical held-out "
        "frame-condition pairs. Zero, Ridge, additive spline GAM and additive EBM use the same eight "
        "quality features and source train/validation pairs. The EBM with two interactions is exploratory.\n\n"
        "Lower G means a more faithful response prediction; zero-change has G=1 when the response "
        "denominator is defined. Each G averages video-level squared errors before the ratio and square root. "
        "Differences are mean paired-seed G differences; brackets are exploratory 95% crossed seed × "
        "test-cluster percentile intervals. The fixed-seed cluster intervals are also in metrics.json.\n\n"
        "| Frozen teacher | Test set | Ridge G | GAM G | EBM G | GAM − Ridge [95% CI] | EBM − Ridge [95% CI] |\n"
        "|---|---|---:|---:|---:|---:|---:|\n" + "\n".join(rows) + "\n\n"
        "On external Celeb-DF++, EBM − Ridge is " + _contrast(kan["ebm_minus_ridge"]) + " for KAN and "
        + _contrast(mlp["ebm_minus_ridge"]) + " for MLP. EBM − zero is "
        + _contrast(kan["ebm_minus_zero"]) + " for KAN and " + _contrast(mlp["ebm_minus_zero"]) + " for MLP. "
        "These results distinguish relative improvement over Ridge from fidelity against the no-change baseline.\n\n"
        "metrics.csv records per-seed and per-condition MAE, RMSE, bias, sign exclusions and correlations; "
        "teacher_score_distribution.csv separates teacher scores by class. The SVG figures and their source "
        "tables are in figures/; fitted response models are not PD/ALE curves or detector AUROC comparisons.\n\n"
        "## Limitations\n\nAll comparisons are exploratory after target inspection and unadjusted for "
        "multiple testing. Intervals condition on the observed source data, test clusters and generators; "
        "parent identities may remain dependent. Quality-only surrogates do not establish causal effects or "
        "a KAN-specific structural advantage.\n")


def revise_report(directory):
    directory = Path(directory).resolve()
    root = (Path.cwd()/"outputs"/"extensions"/"explanations").resolve()
    if not directory.is_relative_to(root):
        raise ValueError("Report directory must be an extension explanation run")
    metrics = directory/"metrics.json"
    result = json.loads(metrics.read_text(encoding="utf-8"))
    statuses = json.loads((directory/"seed_status.json").read_text(encoding="utf-8"))
    expected = {f"{head}/seed{seed}" for head in ("kan_spline", "mlp_gelu") for seed in range(10)}
    if set(statuses) != expected or any(row.get("status") != "COMPLETED" for row in statuses.values()):
        raise ValueError("All 20 seed/head combinations must be completed")
    report, backup, audit = directory/"REPORT.md", directory/"REPORT.original.md", directory/"report_revision.json"
    if backup.exists() or audit.exists():
        raise FileExistsError("A prior report revision already exists")
    text = render_report(result)
    previous = sha256(report)
    shutil.copyfile(report, backup)
    report.write_text(text, encoding="utf-8")
    shutil.copyfile(Path(__file__), directory/"report_generator.py")
    atomic_json(audit, {"reason": "add numeric findings from the completed frozen summary",
                        "metrics_sha256": sha256(metrics), "previous_report_sha256": previous,
                        "revised_report_sha256": sha256(report), "generator_sha256": sha256(Path(__file__)),
                        "scientific_tables_changed": False})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(revise_report(parser.parse_args().directory))


if __name__ == "__main__":
    main()