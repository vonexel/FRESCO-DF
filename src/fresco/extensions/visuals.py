from __future__ import annotations


import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


matplotlib.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "fresco-extensions", "font.size": 10, "axes.spines.top": False,
                            "axes.spines.right": False, "figure.constrained_layout.use": True})
COLORS = {"kan_spline": "#0072B2", "mlp_gelu": "#D55E00", "ffpp": "#0072B2", "celebdfpp": "#D55E00"}


def response_figures(path):
    path = Path(path)
    coverage = pd.read_csv(path / "coverage.csv")
    coverage = coverage.groupby(["dataset", "condition"], as_index=False)[["paired", "supported"]].sum()
    coverage["fraction"] = coverage.supported/coverage.paired
    fig, ax = plt.subplots(figsize=(10, 4.3))
    conditions = sorted(coverage.condition.unique())
    x = np.arange(len(conditions))
    for offset, dataset in ((-.18, "ffpp"), (.18, "celebdfpp")):
        rows = coverage[coverage.dataset == dataset].set_index("condition").reindex(conditions)
        bars = ax.bar(x+offset, rows.fraction, .36, label=f"{dataset}: common", color=COLORS[dataset])
        ax.bar(x+offset, 1-rows.fraction, .36, bottom=rows.fraction, color="white", hatch="//",
               edgecolor=COLORS[dataset], linewidth=.5, label=f"{dataset}: unsupported")
        for bar, count in zip(bars, rows.paired):
            if pd.notna(count):
                ax.text(bar.get_x()+bar.get_width()/2, 1, str(int(count)),
                        rotation=90, ha="center", va="bottom", fontsize=6)
    ax.set(xlabel="Actual degradation", ylabel="Common-support fraction", xticks=x, xticklabels=conditions,
           ylim=(0, 1.12), title="Coverage-v2 intersection; labels show paired frames")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.savefig(path / "coverage.svg", metadata={"Date": None})
    plt.close(fig)

    support = pd.read_csv(path / "fidelity_vs_support.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, dataset in zip(axes, ("ffpp", "celebdfpp")):
        rows = support[support.dataset == dataset]
        for head in ("kan_spline", "mlp_gelu"):
            part = rows[rows["head"] == head]
            for _, seed in part.groupby("seed"):
                ax.plot(seed.support_bin, seed.G, color=COLORS[head], alpha=.2, linewidth=.7)
            means = part.groupby("support_bin").agg(g=("G", "mean"), n=("pairs", "first")).reset_index()
            ax.plot(means.support_bin, means.g, marker="o", color=COLORS[head], label=head)
            for point in means.itertuples():
                ax.annotate(f"n={int(point.n)}", (point.support_bin, point.g), fontsize=6, xytext=(0, 4),
                            textcoords="offset points", ha="center")
        ax.axhline(1, color="0.35", linestyle="--", linewidth=.8)
        ax.set(title=dataset, xlabel="Source-trained support-distance bin", ylabel="G on common support")
    axes[0].legend()
    fig.suptitle("Thin lines: 10 paired seeds; dashed: zero-change G=1")
    fig.savefig(path / "fidelity_vs_support.svg", metadata={"Date": None})
    plt.close(fig)

    agreement = pd.read_csv(path / "response_agreement.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
    for ax, dataset in zip(axes, ("ffpp", "celebdfpp")):
        rows = agreement[agreement.dataset == dataset]
        for head in ("kan_spline", "mlp_gelu"):
            p = rows[rows["head"] == head]
            p = p.assign(weighted=p.predicted*p.n, weighted_x=p.observed*p.n)
            p = p.groupby("observed_bin", as_index=False)[["n", "weighted", "weighted_x"]].sum()
            p = p[p.n > 0]
            ax.plot(p.weighted_x/p.n, p.weighted/p.n, color=COLORS[head], label=head)
        ax.axhline(0, color="0.5", linewidth=.8)
        ax.axvline(0, color="0.5", linewidth=.8)
        ax.plot([-10, 10], [-10, 10], linestyle="--", color="0.2", linewidth=.8)
        ax.set(title=dataset, xlabel="Observed Δlogit (binned mean)", ylabel="Predicted Δlogit (binned mean)")
        reference = rows[rows["head"] == "kan_spline"]
        if not reference.empty:
            pairs = int(reference.groupby("seed").n.sum().iloc[0])
            ax.text(.03, .96, f"n={pairs} pairs/seed", transform=ax.transAxes, va="top", fontsize=8)
    axes[0].legend()
    fig.suptitle("Common-support pairs; dashed: perfect agreement")
    fig.savefig(path / "response_agreement.svg", metadata={"Date": None})
    plt.close(fig)

    components = pd.read_csv(path / "response_components.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for row, head in enumerate(("kan_spline", "mlp_gelu")):
        for col, dataset in enumerate(("ffpp", "celebdfpp")):
            ax = axes[row, col]
            part = components[(components.dataset == dataset) & (components["head"] == head)]
            order = sorted(part.condition.unique())
            for offset, (name, color) in zip((-.2, 0., .2),
                    (("cov", "#0072B2"), ("embed", "#D55E00"), ("interaction", "#009E73"))):
                pooled = []
                for (seed, condition), group in part.groupby(["seed", "condition"], observed=True):
                    n = group.pairs.to_numpy(float)
                    mu = group[name+"_mean"].to_numpy(float)
                    std = group[name+"_std"].fillna(0).to_numpy(float)
                    mean = float((n*mu).sum()/n.sum())
                    variance = float(((n-1)*std**2+n*(mu-mean)**2).sum()/max(n.sum()-1, 1))
                    pooled.append({"seed": seed, "condition": condition, "mean": mean,
                                   "pair_sd": np.sqrt(max(variance, 0))})
                summary = pd.DataFrame(pooled).groupby("condition").agg(
                    mean=("mean", "mean"), seed_sd=("mean", "std"), pair_sd=("pair_sd", "mean")).reindex(order)
                x = np.arange(len(order))+offset
                ax.vlines(x, summary["mean"]-summary.pair_sd, summary["mean"]+summary.pair_sd,
                          color=color, alpha=.2, linewidth=5)
                ax.errorbar(x, summary["mean"], yerr=summary.seed_sd, marker="o", capsize=2,
                            linestyle="none", label=name, color=color)
            ax.axhline(0, color="0.4", linewidth=.8)
            ax.set(title=f"{head}: {dataset}", xticks=np.arange(len(order)), xticklabels=order,
                   xlabel="Degradation condition", ylabel="Response component (logit)")
            ax.tick_params(axis="x", rotation=45)
    axes[0, 0].legend()
    fig.suptitle("Wide translucent bars: paired-frame SD; thin bars: seed SD; neither is a confidence interval")
    fig.savefig(path / "response_components.svg", metadata={"Date": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("response",))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    response_figures(args.directory)


if __name__ == "__main__":
    main()