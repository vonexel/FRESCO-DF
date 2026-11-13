import pytest
from fresco.extensions.surrogate_report import render_report


def test_surrogate_report_uses_completed_paired_results():
    zero = {"head_a_mean": .9, "difference": -.1, "crossed_seeds_clusters": {"ci95": [-.2, -.01]}}
    ridge = {"difference": -.2, "crossed_seeds_clusters": {"ci95": [-.3, -.05]}}
    reports = {f"{head}/{dataset}": {"ridge_minus_zero": zero, "gam_minus_zero": zero,
               "ebm_minus_zero": zero, "gam_minus_ridge": ridge, "ebm_minus_ridge": ridge}
               for head in ("kan_spline", "mlp_gelu") for dataset in ("ffpp", "celebdfpp")}
    value = {"status": "RUN", "seeds": list(range(10)), "pairs": 100, "contrasts": reports}
    text = render_report(value)
    assert "0.9000" in text and "-0.2000 [-0.3000, -0.0500]" in text
    value["seeds"] = [0]
    with pytest.raises(ValueError, match="ten-seed"):
        render_report(value)