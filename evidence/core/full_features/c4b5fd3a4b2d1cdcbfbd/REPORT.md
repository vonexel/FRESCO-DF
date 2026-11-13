# FRESCO empirical report

Measured results from the supplied face videos. No target data were used for model selection.

Seeds = 10 (paper §6 recommends ≥10). Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].

Primary interpretation question: how much of the real paired image response is captured by an observed-quality partial response? A disagreement is an explanation-coverage gap; the algebraic decomposition does not establish causality.

## Measurements

| Head | Budget | k | Covariates | Held-out method | Video AUROC mean ± seed SD | Parameters | Head b256 ms |
|---|---|---:|---|---|---:|---:|---:|
|linear|params|1024|False|nan|0.8019 ± 0.0059|1025|0.1220|
|linear|params|1024|True|nan|0.8050 ± 0.0061|1029|0.1160|
|linear|time|1024|False|nan|0.8031 ± 0.0053|1025|0.1194|
|linear|time|1024|True|nan|0.8074 ± 0.0047|1029|0.1197|
|mlp_gelu|params|1024|False|nan|0.8154 ± 0.0080|73873|0.2365|
|mlp_gelu|params|1024|True|nan|0.8151 ± 0.0055|74161|0.2487|
|mlp_gelu|time|1024|False|nan|0.8154 ± 0.0042|73873|0.2402|
|mlp_gelu|time|1024|True|nan|0.8112 ± 0.0089|74161|0.2390|

## Hypothesis assessment

See [stats.json](stats.json) for paired conditional intervals and budget feasibility. TOST uses 90% intervals. Failure to demonstrate equivalence is inconclusive. Comparators are selected with FF++ validation only. Mean single-seed AUC contrasts are used; scores are not ensembled for inference.

| Comparison | Budget | k | Covariates | Held-out method | Verdict | Reason |
|---|---|---|---|---|---|---|

## Primary paired response agreement

Status: **INCONCLUSIVE**. Primary configuration absent or incomplete

Prespecified contrast: spline KAN minus GELU MLP, k16, observed quality covariates, parameter budget. Each video has equal weight; seed-specific agreement ratios are averaged.


## Article criteria and limitations

A: frame/video AUROC, AP, EER, Brier, NLL and equal-width ECE. EER is the FPR=FNR crossing of the linearly interpolated empirical ROC, retaining all thresholds. P: every trainable coefficient counted; fixed CLIP and PCA state are not trainable head parameters. C: separate parameter and train-wall-time budgets, head b1/b256 latency and training memory. Encoder/crop timing must be read from the feature/preprocessing manifests; head latency alone is not pipeline latency.

R: paired degradations and cross-dataset transfer. I: observed-covariate response stability and intervention decomposition. Symbolic PD approximation is a diagnostic, not validated full-model symbolification. No universal monotonic direction of fake score versus JPEG quality is assumed. E: resolved configs, source/data hashes, initialization checkpoints, completed-stage checksums and failed-run logs.

Intervals are conditional on the fixed seeds; training-seed dispersion must be reported separately. Identity clustering holds target identities together; cross-role identity/source dependencies can remain. DeLong video intervals are secondary descriptive intervals. Unknown-method detection and cross-dataset shift are distinct tasks; the latter confounds dataset and generator. The adapted first-layer plasticity score does not reproduce the complete Canevaro pipeline.

Non-goals retained transparently: no LN tuning, no claim of exact upstream KAN implementations, no fabricated c40 condition when c40 videos are absent, no automatic Pareto recommendation from point estimates alone.

## Artifacts

[Results](results.csv), [seed summary](results_summary.csv), [descriptive Pareto table](pareto.csv), [novelty results](openset_results.csv), [surrogate fidelity](symbolic_fidelity.csv), [environment and data provenance](manifest.json).

- [auc_by_method_heatmap](figures/auc_by_method_heatmap.svg) · [source data](figures/auc_by_method_heatmap.csv)
- [auc_celebdfpp_by_head](figures/auc_celebdfpp_by_head.svg) · [source data](figures/auc_celebdfpp_by_head.csv)
- [calibration_reliability](figures/calibration_reliability.svg) · [source data](figures/calibration_reliability.csv)
- [covariate_ablation](figures/covariate_ablation.svg) · [source data](figures/covariate_ablation.csv)
- [pareto_auc_latency](figures/pareto_auc_latency.svg) · [source data](figures/pareto_auc_latency.csv)
- [response_curves_covariates](figures/response_curves_covariates.svg) · [source data](figures/response_curves_covariates.csv)
- [robustness_curves](figures/robustness_curves.svg) · [source data](figures/robustness_curves.csv)
- [stability](figures/stability.svg) · [source data](figures/stability.csv)

Completed runs: 80. Failure records: 0. Negative outcomes are retained.
