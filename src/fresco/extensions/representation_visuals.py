from __future__ import annotations


import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


matplotlib.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "fresco-representation", "font.size": 9,
                            "axes.spines.top": False, "axes.spines.right": False,
                            "figure.constrained_layout.use": True})
COLORS = {"kan_spline": "#0072B2", "mlp_gelu": "#D55E00", "ffpp": "#009E73", "celebdfpp": "#CC79A7"}


def representation_tables(metrics, reports, path):
    path = Path(path)
    external = metrics[(metrics.dataset == "celebdfpp") & (metrics.condition == "clean")]
    rows = []
    for k in (16, 128):
        report = reports[f"k{k}/celebdfpp"]
        for recipe in ("clean", "augmented"):
            for head in ("kan_spline", "mlp_gelu"):
                name = f"{recipe}: covariate effect, {head}"
                record = next(r for r in report["contrasts"] if r["contrast"] == name)
                ci = record.get("crossed_seeds_clusters", {}).get("ci95")
                rows.append({"k": k, "recipe": recipe, "head": head, "effect": record["difference"],
                             "ci_low": ci[0] if ci else np.nan, "ci_high": ci[1] if ci else np.nan,
                             "videos": report["videos"], "clusters": report["clusters"], "seeds": 10})
    pd.DataFrame(rows).to_csv(path/"covariate_effect.csv", index=False)
    pivot = external.pivot_table(index=["k", "recipe", "head", "seed"], columns="covariates", values="auc")
    pivot["effect"] = pivot[True]-pivot[False]
    pivot.reset_index().to_csv(path/"factor_interaction.csv", index=False)
    transfer = metrics[(metrics.k == 16) & (metrics.recipe == "clean") & metrics.covariates
                       & (metrics.condition == "clean")]
    transfer.to_csv(path/"transfer_gap.csv", index=False)
    robustness = metrics[(metrics.k == 16) & (metrics.recipe == "clean") & metrics.covariates]
    robustness.to_csv(path/"robustness.csv", index=False)


def representation_figures(path):
    path = Path(path)
    cov = pd.read_csv(path/"covariate_effect.csv")
    fig, ax = plt.subplots(figsize=(8, 4.7))
    cov = cov.sort_values(["k", "recipe", "head"]).reset_index(drop=True)
    y = np.arange(len(cov))
    for i, row in cov.iterrows():
        ax.plot(row.effect, i, "o", color=COLORS[row["head"]])
        if np.isfinite(row.ci_low) and np.isfinite(row.ci_high):
            ax.hlines(i, row.ci_low, row.ci_high, color=COLORS[row["head"]])
        ax.annotate(f"n={int(row.videos)}", (row.effect, i), xytext=(6, 3), textcoords="offset points", fontsize=7)
    ax.axvline(0, color="0.4", linestyle="--")
    ax.set(yticks=y, yticklabels=[f"k{r.k} {r.recipe} {r.head}" for r in cov.itertuples()],
           xlabel="Celeb clean video AUROC: covariates on − off", title="Crossed seed × cluster 95% CI; 10 seeds")
    fig.savefig(path/"covariate_effect.svg", metadata={"Date": None})
    plt.close(fig)

    interaction = pd.read_csv(path/"factor_interaction.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, k in zip(axes, (16, 128)):
        for head in ("kan_spline", "mlp_gelu"):
            part = interaction[(interaction.k == k) & (interaction["head"] == head)]
            for _, seed in part.groupby("seed"):
                p = seed.set_index("recipe").reindex(["clean", "augmented"])
                ax.plot([0, 1], p.effect, color=COLORS[head], alpha=.22, linewidth=.7)
            p = part.groupby("recipe").effect.mean().reindex(["clean", "augmented"])
            ax.plot([0, 1], p, marker="o", color=COLORS[head], label=head)
        ax.axhline(0, color="0.5", linestyle="--", linewidth=.8)
        ax.set(title=f"PCA k={k}", xticks=[0, 1], xticklabels=["clean", "augmented"],
               ylabel="Celeb AUROC cov-on − cov-off")
    axes[0].legend()
    fig.suptitle("Thin trajectories: paired training seeds; no smoothing")
    fig.savefig(path/"factor_interaction.svg", metadata={"Date": None})
    plt.close(fig)

    transfer = pd.read_csv(path/"transfer_gap.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    for head in ("kan_spline", "mlp_gelu"):
        p = transfer[transfer["head"] == head]
        for _, row in p.groupby("seed"):
            row = row.set_index("dataset").reindex(["ffpp", "celebdfpp"])
            ax.plot([0, 1], row.auc, color=COLORS[head], alpha=.3, linewidth=.8)
        means = p.groupby("dataset").auc.mean().reindex(["ffpp", "celebdfpp"])
        ax.plot([0, 1], means, marker="o", color=COLORS[head], label=head)
    ax.set(xticks=[0, 1], xticklabels=["FF++ test", "Celeb-DF++ test"], ylabel="Video AUROC",
           title="Predeclared k16 clean, covariates on; paired seeds")
    ax.legend()
    fig.savefig(path/"transfer_gap.svg", metadata={"Date": None})
    plt.close(fig)

    robustness = pd.read_csv(path/"robustness.csv")
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    families = (("JPEG", ("jpeg90", "jpeg70", "jpeg50", "jpeg30")),
                ("Blur", ("blur1", "blur2", "blur3")),
                ("Resize", ("resize0.5", "resize0.25")))
    for i, dataset in enumerate(("ffpp", "celebdfpp")):
        for j, (title, conditions) in enumerate(families):
            ax = axes[i, j]
            for head in ("kan_spline", "mlp_gelu"):
                p = robustness[(robustness.dataset == dataset) & (robustness["head"] == head)]
                q = p.groupby("condition").auc.agg(["mean", "std"]).reindex(conditions)
                ax.errorbar(np.arange(len(conditions)), q["mean"], yerr=q["std"], marker="o", capsize=2,
                            color=COLORS[head], label=head)
            ax.set(title=f"{dataset}: {title}", xticks=np.arange(len(conditions)), xticklabels=conditions,
                   ylabel="Video AUROC")
            ax.tick_params(axis="x", rotation=35)
    axes[0, 0].legend()
    fig.suptitle("Predeclared k16 clean, covariates on; bars show seed SD, not CI")
    fig.savefig(path/"robustness.svg", metadata={"Date": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    representation_figures(args.directory)


if __name__ == "__main__":
    main()