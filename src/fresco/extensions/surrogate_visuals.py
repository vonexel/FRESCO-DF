from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


matplotlib.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "fresco-surrogates", "font.size": 9,
                            "axes.spines.top": False, "axes.spines.right": False,
                            "figure.constrained_layout.use": True})
METHODS = ("zero", "ridge", "gam", "ebm", "ebm_interactions")
COLORS = {"zero": "#777777", "ridge": "#0072B2", "gam": "#009E73", "ebm": "#D55E00",
          "ebm_interactions": "#CC79A7"}
FEATURES = ("Blockiness proxy", "Log Laplacian variance", "Face size", "High-frequency ratio")


def surrogate_figures(path):
    path = Path(path)
    fidelity = pd.read_csv(path / "surrogate_fidelity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, head in zip(axes, ("kan_spline", "mlp_gelu")):
        rows = fidelity[fidelity["head"] == head]
        for offset, method in zip(np.linspace(-.32, .32, len(METHODS)), METHODS):
            p = rows[rows.method == method].set_index("dataset").reindex(["ffpp", "celebdfpp"])
            x = np.arange(2)+offset
            ax.plot(x, p.G, marker="o", linestyle="none", color=COLORS[method], label=method)
            ax.vlines(x, p.ci_low, p.ci_high, color=COLORS[method], linewidth=1)
        ax.axhline(1, color="0.4", linestyle="--", linewidth=.8)
        counts = rows.groupby("dataset").videos.first()
        ax.set(title=f"{head}; videos: {int(counts['ffpp'])} / {int(counts['celebdfpp'])}",
               xticks=np.arange(2), xticklabels=["FF++", "Celeb-DF++"], ylabel="G on same pairs")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Ten seeds; 95% crossed seed × cluster CI; interactions variant exploratory")
    fig.savefig(path / "surrogate_fidelity.svg", metadata={"Date": None})
    plt.close(fig)

    agreement = pd.read_csv(path / "surrogate_agreement.csv")
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
    for i, head in enumerate(("kan_spline", "mlp_gelu")):
        for j, dataset in enumerate(("ffpp", "celebdfpp")):
            ax = axes[i, j]
            rows = agreement[(agreement["head"] == head) & (agreement.dataset == dataset)]
            for method in ("ridge", "gam", "ebm"):
                p = rows.assign(weighted=rows[method]*rows.n, weighted_x=rows.observed*rows.n)
                p = p.groupby("observed_bin", as_index=False)[["n", "weighted", "weighted_x"]].sum()
                p = p[p.n > 0]
                ax.plot(p.weighted_x/p.n, p.weighted/p.n, label=method, color=COLORS[method])
            ax.plot([-10, 10], [-10, 10], linestyle="--", color="0.2", linewidth=.8)
            ax.axhline(0, color="0.6", linewidth=.6)
            ax.set(title=f"{head}, {dataset}", xlabel="Observed Δlogit (binned)", ylabel="Predicted Δlogit")
            if not rows.empty:
                pairs = int(rows.groupby("seed").n.sum().iloc[0])
                ax.text(.03, .96, f"n={pairs} pairs/seed", transform=ax.transAxes, va="top", fontsize=8)
    axes[0, 0].legend()
    fig.savefig(path / "surrogate_agreement.svg", metadata={"Date": None})
    plt.close(fig)

    transfer = pd.read_csv(path / "surrogate_transfer.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, head in zip(axes, ("kan_spline", "mlp_gelu")):
        rows = transfer[transfer["head"] == head].set_index("method").reindex(METHODS)
        x = np.arange(len(METHODS))
        ax.scatter(x, rows.difference, c=[COLORS[m] for m in METHODS])
        ax.vlines(x, rows.ci_low, rows.ci_high, color="0.25", linewidth=1)
        ax.axhline(0, color="0.4", linestyle="--", linewidth=.8)
        reference = rows.dropna(subset=["source_videos", "target_videos"])
        counts = (f"; videos: {int(reference.source_videos.iloc[0])} / {int(reference.target_videos.iloc[0])}"
                  if not reference.empty else "")
        ax.set(title=head+counts, xticks=x, xticklabels=METHODS, ylabel="G(Celeb) − G(FF++)")
        ax.tick_params(axis="x", rotation=35)
    fig.suptitle("Independent dataset-cluster bootstrap; paired seeds; 95% percentile CI")
    fig.savefig(path / "surrogate_transfer.svg", metadata={"Date": None})
    plt.close(fig)

    functions = pd.read_csv(path / "surrogate_functions.csv")
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for feature, ax in enumerate(axes.flat):
        rows = functions[functions.feature == feature]
        for method in ("gam", "ebm"):
            # Source grids can differ across teacher seeds only in theory; they
            # are generated from the same source rows here.
            values = rows[rows.method == method].groupby("grid").response.agg(["mean", "std"]).reset_index()
            ax.plot(values.grid, values["mean"], label=method, color=COLORS[method])
            ax.fill_between(values.grid, values["mean"]-values["std"], values["mean"]+values["std"],
                            color=COLORS[method], alpha=.15)
        source = rows.groupby("grid", as_index=False).source_bin_count.first()
        density = ax.twinx()
        density.fill_between(source.grid, 0, source.source_bin_count, color="0.7", alpha=.2)
        density.tick_params(axis="y", labelsize=6, colors="0.4")
        ax.set(title=("Clean: " if feature < 4 else "Changed: ") + FEATURES[feature % 4],
               xlabel="Source-supported quality", ylabel="Centered response (logit)")
    axes.flat[0].legend()
    fig.suptitle("Terms: across-teacher/seed SD, not CI; grey: source-train bin counts (right axis)")
    fig.savefig(path / "surrogate_functions.svg", metadata={"Date": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    surrogate_figures(args.directory)


if __name__ == "__main__":
    main()