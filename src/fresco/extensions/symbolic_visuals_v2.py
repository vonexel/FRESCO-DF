from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


matplotlib.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "fresco-symbolic-v2", "font.size": 9,
                            "axes.spines.top": False, "axes.spines.right": False,
                            "figure.constrained_layout.use": True})
COLORS = {"pruned": "#0072B2", "mixed": "#D55E00"}


def _frontier(data):
    best, points = float("inf"), []
    for x, y in sorted(set(zip(data.retained_edges, data.logit_rmse_equal_video))):
        if np.isfinite(x) and np.isfinite(y) and y < best:
            points.append((x, y))
            best = y
    return np.asarray(points, dtype=float).reshape(-1, 2)


def symbolic_figures(path):
    path = Path(path)
    data = pd.read_csv(path/"symbolic_tradeoff.csv")
    data["retained_edges"] = np.where(data.clone == "pruned", data.total_edges-data.pruned_edges,
                                       data.retained_spline_edges)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for col, dataset in enumerate(("ffpp", "celebdfpp")):
        part = data[data.dataset == dataset]
        for clone in ("pruned", "mixed"):
            points = part[part.clone == clone]
            axes[0, col].scatter(points.retained_edges, points.logit_rmse_equal_video,
                                 color=COLORS[clone], label=clone, alpha=.7)
            axes[1, col].scatter(points.retained_edges, points.auc_change,
                                 color=COLORS[clone], label=clone, alpha=.7)
        frontier = _frontier(part)
        if len(frontier):
            axes[0, col].plot(frontier[:, 0], frontier[:, 1], color="0.35", linestyle="--", linewidth=.8,
                              label="Observed frontier")
        axes[1, col].axhline(0, color="0.4", linestyle=":", linewidth=.8)
        axes[0, col].set(title=dataset, ylabel="Equal-video logit RMSE vs teacher")
        axes[1, col].set(xlabel="Original spline edges retained", ylabel="Video AUROC change vs teacher")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Pruned / mixed KAN copies on clean test; one point per seed; unsmoothed observed frontier")
    fig.savefig(path/"symbolic_tradeoff.svg", metadata={"Date": None})
    plt.close(fig)

    edges = pd.read_csv(path/"symbolic_edges.csv")
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
        family, rmse = part.family.iloc[0], part.edge_validation_rmse.iloc[0]
        ax.set(title=f"Layer {key[0]}, edge {key[1]}←{key[2]}: {status}\n{family}; val RMSE={rmse:.3g}",
               xlabel="Edge input", ylabel="Edge output")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Predeclared seed 0 edges; occupancy is source train")
    fig.savefig(path/"symbolic_edges.svg", metadata={"Date": None})
    plt.close(fig)
    transfer_path = path/"symbolic_transfer.csv"
    if transfer_path.is_file():
        result = pd.read_csv(transfer_path).iloc[0]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axvline(0, color="0.4", linestyle="--", linewidth=.8)
        ax.plot(result["difference"], 0, "o", color=COLORS["mixed"])
        ax.hlines(0, result.ci_low, result.ci_high, color=COLORS["mixed"])
        ax.set(yticks=[0], yticklabels=["Complete symbolic KAN"],
               xlabel="Celeb − FF++ equal-video logit RMSE", title="Crossed seed × cluster 95% CI")
        ax.annotate(f"{int(result.seeds)} seeds; n={int(result.source_videos)}/{int(result.target_videos)} videos",
                    (result["difference"], 0), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8)
        fig.savefig(path/"symbolic_transfer.svg", metadata={"Date": None})
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    symbolic_figures(parser.parse_args().directory)


if __name__ == "__main__":
    main()