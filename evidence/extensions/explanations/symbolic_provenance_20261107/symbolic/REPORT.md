# KAN simplification: whole-model held-out evaluation

Ten original seeds evaluated; fully symbolified models: 0/10. Pruned copies and mixed symbolic/spline copies were compared with the original frozen KAN on identical FF++ and Celeb-DF++ inputs. Mixed copies retain rejected spline edges and are not fully symbolic.

The source-selected threshold is fixed before test; metrics, seed medians/interquartile ranges and exploratory paired AUROC, logit RMSE and normalized logit error intervals are in the result tables. The representative edge plot was fixed before this held-out evaluation. Retained-edge counts describe structure; stored mixed copies still include original spline weights. Head latency is measured in the current environment and excludes the encoder; concurrent work can affect it.

## Limitations

This is not a KAN-vs-MLP structural benchmark or a causal explanation. An edge fit alone is not whole-model fidelity. Source bounding boxes do not guarantee manifold support. Intervals are exploratory, unadjusted, and conditional on source data and test generators.
