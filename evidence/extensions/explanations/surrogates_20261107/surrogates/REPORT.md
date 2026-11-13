# Quality-only response surrogates

All ten frozen teacher seeds per head were evaluated on 1800963 identical held-out frame-condition pairs. Zero, Ridge, additive spline GAM and additive EBM use the same eight quality features and source train/validation pairs. The EBM with two interactions is exploratory.

Lower G means a more faithful response prediction; zero-change has G=1 when the response denominator is defined. Each G averages video-level squared errors before the ratio and square root. Differences are mean paired-seed G differences; brackets are exploratory 95% crossed seed × test-cluster percentile intervals. The fixed-seed cluster intervals are also in metrics.json.

| Frozen teacher | Test set | Ridge G | GAM G | EBM G | GAM − Ridge [95% CI] | EBM − Ridge [95% CI] |
|---|---|---:|---:|---:|---:|---:|
| kan_spline | ffpp | 0.9400 | 0.9628 | 0.9440 | +0.0228 [+0.0019, +0.0505] | +0.0040 [-0.0067, +0.0136] |
| kan_spline | celebdfpp | 1.2688 | 1.4364 | 1.0860 | +0.1676 [+0.0766, +0.2599] | -0.1828 [-0.3164, -0.0608] |
| mlp_gelu | ffpp | 0.9724 | 0.9787 | 0.9806 | +0.0062 [-0.0072, +0.0225] | +0.0082 [+0.0018, +0.0141] |
| mlp_gelu | celebdfpp | 1.0299 | 1.1017 | 1.1038 | +0.0718 [+0.0295, +0.1152] | +0.0739 [+0.0435, +0.1031] |

On external Celeb-DF++, EBM − Ridge is -0.1828 [-0.3164, -0.0608] for KAN and +0.0739 [+0.0435, +0.1031] for MLP. EBM − zero is +0.0860 [+0.0274, +0.1402] for KAN and +0.1038 [+0.0724, +0.1346] for MLP. These results distinguish relative improvement over Ridge from fidelity against the no-change baseline.

metrics.csv records per-seed and per-condition MAE, RMSE, bias, sign exclusions and correlations; teacher_score_distribution.csv separates teacher scores by class. The SVG figures and their source tables are in figures/; fitted response models are not PD/ALE curves or detector AUROC comparisons.

## Limitations

All comparisons are exploratory after target inspection and unadjusted for multiple testing. Intervals condition on the observed source data, test clusters and generators; parent identities may remain dependent. Quality-only surrogates do not establish causal effects or a KAN-specific structural advantage.
