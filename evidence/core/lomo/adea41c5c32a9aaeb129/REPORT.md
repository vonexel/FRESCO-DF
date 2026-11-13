# FRESCO empirical report

Measured results from the supplied face videos. No target data were used for model selection.

Seeds = 10 (paper §6 recommends ≥10). Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].

Primary interpretation question: how much of the real paired image response is captured by an observed-quality partial response? A disagreement is an explanation-coverage gap; the algebraic decomposition does not establish causality.

## Measurements

| Head | Budget | k | Covariates | Held-out method | Video AUROC mean ± seed SD | Parameters | Head b256 ms |
|---|---|---:|---|---|---:|---:|---:|
|kan_fast|params|16|True|Deepfakes|0.4371 ± 0.0717|1525|0.7559|
|kan_fast|params|16|True|Face2Face|0.6272 ± 0.1415|1525|0.6941|
|kan_fast|params|16|True|FaceSwap|0.4265 ± 0.0947|1525|0.7169|
|kan_fast|params|16|True|NeuralTextures|0.4550 ± 0.1023|1525|0.7084|
|kan_spline|params|16|True|Deepfakes|0.5497 ± 0.1510|1521|2.3311|
|kan_spline|params|16|True|Face2Face|0.6182 ± 0.1035|1521|2.2125|
|kan_spline|params|16|True|FaceSwap|0.3797 ± 0.0697|1521|2.3130|
|kan_spline|params|16|True|NeuralTextures|0.4120 ± 0.0671|1521|2.2422|
|mlp_gelu|params|16|True|Deepfakes|0.5561 ± 0.0731|1585|0.2446|
|mlp_gelu|params|16|True|Face2Face|0.6193 ± 0.0475|1585|0.2354|
|mlp_gelu|params|16|True|FaceSwap|0.5217 ± 0.0467|1585|0.2299|
|mlp_gelu|params|16|True|NeuralTextures|0.4504 ± 0.0340|1585|0.2357|
|mlp_relu|params|16|True|Deepfakes|0.5628 ± 0.0706|1585|0.2300|
|mlp_relu|params|16|True|Face2Face|0.6082 ± 0.0652|1585|0.2366|
|mlp_relu|params|16|True|FaceSwap|0.5225 ± 0.0398|1585|0.2329|
|mlp_relu|params|16|True|NeuralTextures|0.4847 ± 0.0418|1585|0.2275|

## Hypothesis assessment

See [stats.json](stats.json) for paired conditional intervals and budget feasibility. TOST uses 90% intervals. Failure to demonstrate equivalence is inconclusive. Comparators are selected with FF++ validation only. Mean single-seed AUC contrasts are used; scores are not ensembled for inference.

| Comparison | Budget | k | Covariates | Held-out method | Verdict | Reason |
|---|---|---|---|---|---|---|
|kan_fast vs mlp_gelu|params|16|True|Deepfakes|INCONCLUSIVE|Equivalence not demonstrated; this does not establish inequality|
|kan_spline vs mlp_gelu|params|16|True|Deepfakes|INCONCLUSIVE|Equivalence not demonstrated; this does not establish inequality|
|kan_fast vs mlp_gelu|params|16|True|Face2Face|INCONCLUSIVE|Equivalence not demonstrated; this does not establish inequality|
|kan_spline vs mlp_gelu|params|16|True|Face2Face|INCONCLUSIVE|Equivalence not demonstrated; this does not establish inequality|
|kan_fast vs mlp_gelu|params|16|True|FaceSwap|INCONCLUSIVE|Equivalence not demonstrated; this does not establish inequality|
|kan_spline vs mlp_gelu|params|16|True|FaceSwap|INCONCLUSIVE|Equivalence not demonstrated; this does not establish inequality|
|kan_fast vs mlp_gelu|params|16|True|NeuralTextures|INCONCLUSIVE|Equivalence not demonstrated; this does not establish inequality|
|kan_spline vs mlp_gelu|params|16|True|NeuralTextures|INCONCLUSIVE|Equivalence not demonstrated; this does not establish inequality|

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
- [intervention_response](figures/intervention_response.svg) · [source data](figures/intervention_response.csv)
- [openset_auroc](figures/openset_auroc.svg) · [source data](figures/openset_auroc.csv)
- [pareto_auc_latency](figures/pareto_auc_latency.svg) · [source data](figures/pareto_auc_latency.csv)
- [response_curves_covariates](figures/response_curves_covariates.svg) · [source data](figures/response_curves_covariates.csv)
- [robustness_curves](figures/robustness_curves.svg) · [source data](figures/robustness_curves.csv)
- [stability](figures/stability.svg) · [source data](figures/stability.csv)
- [symbolic_fidelity](figures/symbolic_fidelity.svg) · [source data](figures/symbolic_fidelity.csv)

Completed runs: 160. Failure records: 0. Negative outcomes are retained.
