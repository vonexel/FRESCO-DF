# FRESCO empirical report

Measured results from the supplied face videos. No target data were used for model selection.

Seeds = 10 (paper §6 recommends ≥10). Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].

Primary interpretation question: how much of the real paired image response is captured by an observed-quality partial response? A disagreement is an explanation-coverage gap; the algebraic decomposition does not establish causality.

## Measurements

| Head | Budget | k | Covariates | Held-out method | Video AUROC mean ± seed SD | Parameters | Head b256 ms |
|---|---|---:|---|---|---:|---:|---:|
|kan_fast|params|8|True|nan|0.5933 ± 0.1133|1028|0.7462|
|kan_fast|params|16|True|nan|0.4431 ± 0.0984|1525|0.7411|
|kan_fast|params|32|True|nan|0.5618 ± 0.0777|2677|0.7323|
|kan_fast|params|128|True|nan|0.6900 ± 0.0075|9589|0.7183|
|kan_spline|params|8|True|nan|0.5859 ± 0.1141|1063|2.2324|
|kan_spline|params|16|True|nan|0.4795 ± 0.1299|1521|2.3406|
|kan_spline|params|32|True|nan|0.5308 ± 0.0608|2673|2.2821|
|kan_spline|params|128|True|nan|0.7185 ± 0.0107|9585|2.3130|
|mlp_bspline|params|8|True|nan|0.4906 ± 0.1308|1009|1.2338|
|mlp_bspline|params|16|True|nan|0.4360 ± 0.0733|1569|1.2021|
|mlp_bspline|params|32|True|nan|0.5251 ± 0.0361|2737|1.2109|
|mlp_bspline|params|128|True|nan|0.7312 ± 0.0185|9649|1.2300|
|mlp_gelu|params|8|True|nan|0.3576 ± 0.0499|1009|0.2318|
|mlp_gelu|params|16|True|nan|0.5767 ± 0.0652|1585|0.2328|
|mlp_gelu|params|32|True|nan|0.6051 ± 0.0305|2737|0.2393|
|mlp_gelu|params|128|True|nan|0.7682 ± 0.0165|9649|0.2299|
|mlp_relu|params|8|True|nan|0.3290 ± 0.0517|1009|0.2442|
|mlp_relu|params|16|True|nan|0.5567 ± 0.0662|1585|0.2340|
|mlp_relu|params|32|True|nan|0.6115 ± 0.0402|2737|0.2437|
|mlp_relu|params|128|True|nan|0.7666 ± 0.0157|9649|0.2265|

## Hypothesis assessment

See [stats.json](stats.json) for paired conditional intervals and budget feasibility. TOST uses 90% intervals. Failure to demonstrate equivalence is inconclusive. Comparators are selected with FF++ validation only. Mean single-seed AUC contrasts are used; scores are not ensembled for inference.

| Comparison | Budget | k | Covariates | Held-out method | Verdict | Reason |
|---|---|---|---|---|---|---|
|kan_fast vs mlp_gelu|params|8|True|None|INCONCLUSIVE|Equivalence not demonstrated; this does not establish inequality|
|kan_spline vs mlp_gelu|params|8|True|None|INCONCLUSIVE|Requested budget match not achieved|
|kan_fast vs mlp_gelu|params|16|True|None|INCONCLUSIVE|Equivalence not demonstrated; this does not establish inequality|
|kan_spline vs mlp_gelu|params|16|True|None|INCONCLUSIVE|Equivalence not demonstrated; this does not establish inequality|
|kan_fast vs mlp_gelu|params|32|True|None|INCONCLUSIVE|Equivalence not demonstrated; this does not establish inequality|
|kan_spline vs mlp_gelu|params|32|True|None|INCONCLUSIVE|Equivalence not demonstrated; this does not establish inequality|
|kan_fast vs mlp_gelu|params|128|True|None|INCONCLUSIVE|Equivalence not demonstrated; this does not establish inequality|
|kan_spline vs mlp_gelu|params|128|True|None|INCONCLUSIVE|Equivalence not demonstrated; this does not establish inequality|

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
- [dimension_ablation](figures/dimension_ablation.svg) · [source data](figures/dimension_ablation.csv)
- [intervention_response](figures/intervention_response.svg) · [source data](figures/intervention_response.csv)
- [openset_auroc](figures/openset_auroc.svg) · [source data](figures/openset_auroc.csv)
- [pareto_auc_latency](figures/pareto_auc_latency.svg) · [source data](figures/pareto_auc_latency.csv)
- [response_curves_covariates](figures/response_curves_covariates.svg) · [source data](figures/response_curves_covariates.csv)
- [robustness_curves](figures/robustness_curves.svg) · [source data](figures/robustness_curves.csv)
- [stability](figures/stability.svg) · [source data](figures/stability.csv)
- [symbolic_fidelity](figures/symbolic_fidelity.svg) · [source data](figures/symbolic_fidelity.csv)

Completed runs: 200. Failure records: 0. Negative outcomes are retained.
