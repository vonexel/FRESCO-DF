from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


matplotlib.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "fresco-effects", "font.size": 10,
                            "axes.spines.top": False, "axes.spines.right": False,
                            "figure.constrained_layout.use": True})
COLORS = {"pd": "#0072B2", "ale": "#D55E00", "common": "#009E73"}


def effect_figures(path):
    path = Path(path)
    data = pd.read_csv(path / "pd_ale_fidelity.csv")
    coverage = pd.read_csv(path / "pd_ale_coverage.csv")
    totals = coverage.groupby("dataset")[["paired", "common_supported"]].sum()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, head in zip(axes, ("kan_spline", "mlp_gelu")):
        part = data[data["head"] == head]
        for offset, method in ((-.15, "pd"), (.15, "ale")):
            values = part[part.method == method].set_index("dataset").reindex(["ffpp", "celebdfpp"])
            x = np.arange(2)+offset
            ax.plot(x, values.G, marker="o", linestyle="none", color=COLORS[method], label=method)
            ax.vlines(x, values.ci_low, values.ci_high, color=COLORS[method], linewidth=1)
            for i, row in enumerate(values.itertuples()):
                if np.isfinite(row.G):
                    fraction = totals.loc[row.Index, "common_supported"]/totals.loc[row.Index, "paired"]
                    ax.annotate(f"n={int(row.videos)}; cov={fraction:.0%}", (x[i], row.G), xytext=(0, 6),
                                textcoords="offset points", fontsize=7, ha="center")
        ax.axhline(1, color="0.4", linestyle="--", linewidth=.8)
        ax.set(title=head, xticks=np.arange(2), xticklabels=["FF++", "Celeb-DF++"], ylabel="G on common pairs")
    axes[0].legend()
    fig.suptitle("PD vs ALE; 95% crossed seed × cluster percentile CI; zero-change G=1")
    fig.savefig(path / "pd_ale_fidelity.svg", metadata={"Date": None})
    plt.close(fig)

    curves = pd.read_csv(path / "pd_ale_curves.csv")
    features = list(curves.feature.drop_duplicates())
    labels = {"jpeg_blockiness_proxy": "Blockiness proxy", "log_laplacian_variance": "Log Laplacian variance",
              "face_size": "Face size", "high_frequency_ratio": "High-frequency ratio"}
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex="col")
    for row, head in enumerate(("kan_spline", "mlp_gelu")):
        for col, feature in enumerate(features):
            ax = axes[row, col]
            rows = curves[(curves.feature == feature) & (curves["head"] == head)]
            for method in ("pd", "ale"):
                part = rows[rows.method == method]
                summary = part.groupby("grid").response.agg(["mean", "std"]).reset_index()
                ax.plot(summary.grid, summary["mean"], label=method, color=COLORS[method])
                ax.fill_between(summary.grid, summary["mean"]-summary["std"],
                                summary["mean"]+summary["std"], color=COLORS[method], alpha=.15)
            occupancy = rows.groupby("grid", as_index=False).bin_count.first().fillna(0)
            density = ax.twinx()
            edges = occupancy.grid.to_numpy(float)
            counts = occupancy.bin_count.to_numpy(float)[:-1]
            widths = np.diff(edges)
            heights = np.divide(counts, counts.sum()*widths, out=np.zeros_like(counts),
                                where=(widths > 0) & (counts.sum() > 0))
            density.stairs(heights, edges, fill=True, color="0.7", alpha=.2)
            density.tick_params(axis="y", labelsize=6, colors="0.4")
            low, high = occupancy.grid.min(), occupancy.grid.max()
            pad = max((high-low)*.05, 1e-8)
            ax.axvspan(low-pad, low, color="0.8", alpha=.25)
            ax.axvspan(high, high+pad, color="0.8", alpha=.25)
            ax.set_xlim(low-pad, high+pad)
            ax.set(title=f"{'KAN' if head == 'kan_spline' else 'MLP'}: {labels.get(feature, feature)}",
                   xlabel="Source-standardized quality",
                   ylabel="Centered model effect (logit)")
    axes.flat[0].legend()
    fig.suptitle("PD vs ALE; shading: seed SD, not CI; grey: source density / outside fitted range")
    fig.savefig(path / "pd_ale_curves.svg", metadata={"Date": None})
    plt.close(fig)

    coverage = coverage.groupby(["dataset", "condition"], as_index=False)[
        ["paired", "pd_supported", "ale_supported", "common_supported"]].sum()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, dataset in zip(axes, ("ffpp", "celebdfpp")):
        rows = coverage[coverage.dataset == dataset].sort_values("condition")
        x = np.arange(len(rows))
        for offset, method in ((-.23, "pd"), (0, "ale"), (.23, "common")):
            vals = rows[method+"_supported"]/rows.paired
            ax.bar(x+offset, vals, .22, label=method, color=COLORS[method])
        for i, count in enumerate(rows.paired):
            ax.annotate(str(int(count)), (i, 1), xytext=(0, 2), textcoords="offset points",
                        rotation=90, fontsize=6, ha="center")
        ax.set(title=dataset, xticks=x, xticklabels=rows.condition, ylim=(0, 1.2),
               xlabel="Degradation condition", ylabel="Supported fraction")
        ax.tick_params(axis="x", rotation=45)
    axes[0].legend()
    fig.suptitle("Native and intersected support; labels: paired-frame denominator")
    fig.savefig(path / "pd_ale_coverage.svg", metadata={"Date": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    effect_figures(args.directory)


if __name__ == "__main__":
    main()