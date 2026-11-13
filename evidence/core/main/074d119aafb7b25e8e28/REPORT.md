# FRESCO empirical report

Measured results from the supplied face videos. No target data were used for model selection.

Seeds = 10 (paper §6 recommends ≥10). Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].

Primary interpretation question: how much of the real paired image response is captured by an observed-quality partial response? A disagreement is an explanation-coverage gap; the algebraic decomposition does not establish causality.

## Measurements

| Head | Budget | k | Covariates | Held-out method | Video AUROC mean ± seed SD | Parameters | Head b256 ms |
|---|---|---:|---|---|---:|---:|---:|
|kan_fast|params|16|True|nan|0.4431 ± 0.0984|1525|0.7048|
|kan_fast|time|16|True|nan|0.5272 ± 0.1120|9145|0.7309|
|kan_rational|params|16|True|nan|0.4910 ± 0.0727|1577|0.5413|
|kan_rational|time|16|True|nan|0.5551 ± 0.0703|1665|0.5470|
|kan_spline|params|16|True|nan|0.4795 ± 0.1299|1521|2.2873|
|kan_spline|time|16|True|nan|0.5221 ± 0.0888|13681|2.2972|
|kan_wavelet|params|16|True|nan|0.5050 ± 0.1102|1616|0.3337|
|kan_wavelet|time|16|True|nan|0.3835 ± 0.0633|6121|0.3283|
|linear|params|16|True|nan|0.5447 ± 0.0161|21|0.1199|
|linear|time|16|True|nan|0.5284 ± 0.0191|21|0.1168|
|mlp_bspline|params|16|True|nan|0.4360 ± 0.0733|1569|1.2312|
|mlp_bspline|time|16|True|nan|0.4215 ± 0.0560|2305|1.2246|
|mlp_gelu|params|16|True|nan|0.5767 ± 0.0652|1585|0.2341|
|mlp_gelu|time|16|True|nan|0.5726 ± 0.0366|1585|0.2337|
|mlp_relu|params|16|True|nan|0.5567 ± 0.0662|1585|0.2408|
|mlp_relu|time|16|True|nan|0.5812 ± 0.0544|1585|0.2314|

## Hypothesis assessment

See [stats.json](stats.json) for paired conditional intervals and budget feasibility. TOST uses 90% intervals. Failure to demonstrate equivalence is inconclusive. Comparators are selected with FF++ validation only. Mean single-seed AUC contrasts are used; scores are not ensembled for inference.

| Comparison | Budget | k | Covariates | Held-out method | Verdict | Reason |
|---|---|---|---|---|---|---|
|kan_fast vs mlp_gelu|params|16|True|None|INCONCLUSIVE|Equivalence not demonstrated; this does not establish inequality|
|kan_rational vs mlp_gelu|params|16|True|None|INCONCLUSIVE|Equivalence not demonstrated; this does not establish inequality|
|kan_spline vs mlp_gelu|params|16|True|None|INCONCLUSIVE|Equivalence not demonstrated; this does not establish inequality|
|kan_wavelet vs mlp_gelu|params|16|True|None|INCONCLUSIVE|Equivalence not demonstrated; this does not establish inequality|
|kan_fast vs mlp_gelu|time|16|True|None|INCONCLUSIVE|Equivalence not demonstrated; this does not establish inequality|
|kan_rational vs mlp_gelu|time|16|True|None|INCONCLUSIVE|Equivalence not demonstrated; this does not establish inequality|
|kan_spline vs mlp_gelu|time|16|True|None|INCONCLUSIVE|Equivalence not demonstrated; this does not establish inequality|
|kan_wavelet vs mlp_gelu|time|16|True|None|INCONCLUSIVE|Equivalence not demonstrated; this does not establish inequality|

## Primary paired response agreement

Status: **NOT_SUPPORTED**. Joint predeclared agreement, noninferiority, stability and budget criteria

Prespecified contrast: spline KAN minus GELU MLP, k16, observed quality covariates, parameter budget. Each video has equal weight; seed-specific agreement ratios are averaged.

G(KAN) = 0.7911; G(MLP) = 0.6802; difference = 0.1109, 95% cluster CI = [0.08979351093082467, 0.13107656364837525].

Median curve stability = 0.989279017792665. AUROC noninferiority contrast: {'difference': -0.09721636718414961, 'seed_differences': [-0.16927695342653748, 0.1656960288189382, 0.016692040483746506, -0.1863185093061155, -0.1654612316665237, -0.2411409640620979, 0.053273758469851606, -0.1436808474140151, -0.3566901106441376, 0.054743116905394906], 'uncertainty': 'test-cluster sampling conditional on fixed seeds', 'ci90': [-0.12613954390764218, -0.07171636659946441], 'ci95': [-0.13230058541110995, -0.06705782490751257], 'margin': 0.01, 'equivalent_tost_percentile': False, 'superior_beyond_margin': False, 'clusters': 129, 'resamples_valid': 10000, 'unit': 'source/identity cluster; paired videos'}.

Curve stability coverage complete: True; covariates 4/4; defined seed pairs 180/180. Constant, unavailable or incomplete curves make the primary verdict INCONCLUSIVE. See stats.json for per-feature reasons.


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
- [intervention_response](figures/intervention_response.svg) · [source data](figures/intervention_response.csv)
- [openset_auroc](figures/openset_auroc.svg) · [source data](figures/openset_auroc.csv)
- [pareto_auc_latency](figures/pareto_auc_latency.svg) · [source data](figures/pareto_auc_latency.csv)
- [response_curves_covariates](figures/response_curves_covariates.svg) · [source data](figures/response_curves_covariates.csv)
- [robustness_curves](figures/robustness_curves.svg) · [source data](figures/robustness_curves.csv)
- [stability](figures/stability.svg) · [source data](figures/stability.csv)
- [symbolic_fidelity](figures/symbolic_fidelity.svg) · [source data](figures/symbolic_fidelity.csv)

Completed runs: 160. Failure records: 0. Negative outcomes are retained.
