from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


matplotlib.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "fresco-symbolic", "font.size": 9,
                            "axes.spines.top": False, "axes.spines.right": False,
                            "figure.constrained_layout.use": True})
COLORS = {"pruned": "#0072B2", "mixed": "#D55E00"}


def symbolic_figures(path):
    path = Path(path)
    data = pd.read_csv(path / "symbolic_tradeoff.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, dataset in zip(axes, ("ffpp", "celebdfpp")):
        part = data[data.dataset == dataset]
        for clone in ("pruned", "mixed"):
            selected = part[part.clone == clone]
            edges = (selected.total_edges - selected.pruned_edges if clone == "pruned"
                     else selected.retained_spline_edges)
            ax.scatter(edges, selected.logit_rmse_equal_video, label=clone, color=COLORS[clone], alpha=.7)
        ax.set(title=dataset, xlabel="Original spline edges retained", ylabel="Equal-video logit RMSE vs teacher")
    axes[0].legend()
    fig.suptitle("Pruned and mixed KAN copies on clean test; one point per seed")
    fig.savefig(path / "symbolic_tradeoff.svg", metadata={"Date": None})
    plt.close(fig)

    edges = pd.read_csv(path / "symbolic_edges.csv")
    groups = list(edges.groupby(["layer", "output", "input"], sort=True))
    fig, axes = plt.subplots(1, len(groups), figsize=(5*len(groups), 4), squeeze=False)
    for ax, (key, part) in zip(axes.flat, groups):
        part = part.sort_values("x")
        ax.plot(part.x, part.actual, color="#0072B2", label="Original spline edge")
        ax.plot(part.x, part.approximation, color="#D55E00", linestyle="--", label="Source-fitted candidate")
        occupancy = ax.twinx()
        occupancy.fill_between(part.x, 0, part.source_bin_count, color="0.7", alpha=.2)
        occupancy.set_ylabel("Source-train bin count", color="0.4")
        status = "accepted" if bool(part.accepted.iloc[0]) else "rejected"
        ax.set(title=f"Layer {key[0]}, edge {key[1]}←{key[2]}: {status}", xlabel="Edge input",
               ylabel="Edge output")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Predeclared seed 0 edges; occupancy is source train")
    fig.savefig(path / "symbolic_edges.svg", metadata={"Date": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    symbolic_figures(parser.parse_args().directory)


if __name__ == "__main__":
    main()