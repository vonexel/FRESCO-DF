import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from fresco.statistics import auc_delong, metrics, paired_auc_bootstrap, fit_temperature, _auc_plan, _weighted_auc


def test_auc_and_calibration_ties():
    y=np.array([0,0,1,1])
    p=np.array([.1,.4,.35,.8])
    m=metrics(y,p)
    assert m["auc"] == .75
    lo,hi=auc_delong(y,p)
    assert lo <= .75 <= hi
    ties=metrics(y,np.full(4,.5))
    assert ties["auc"] == .5
    assert ties["ece"] == 0
    assert np.isclose(ties["brier"],.25)


def test_bootstrap_preserves_single_seed_estimand():
    y=np.tile([0,0,1,1],5)
    a=np.tile([[0,.1,.9,1],[.6,.55,.5,.45]],(1,5))
    b=np.full_like(a,.5)
    assert roc_auc_score(y,a.mean(0)) == 1
    result=paired_auc_bootstrap(y,a,b,groups=np.repeat(np.arange(5),4),resamples=50)
    assert abs(result["difference"]) < 1e-12
    assert result["seed_differences"] == [.5,-.5]
    assert result["equivalent_tost_percentile"]


def test_degenerate_clusters_are_inconclusive():
    r=paired_auc_bootstrap([0,1],[.2,.8],[.3,.7],groups=["one","one"],resamples=5)
    assert r["status"] == "INCONCLUSIVE"


def test_temperature_reduces_overconfidence():
    y=np.array([0,1,0,1])
    logits=np.array([-20,20,10,-10])
    t=fit_temperature(y,logits)
    assert t>1


def test_weighted_auc_exactly_matches_expanded_tied_video_sample():
    y=np.array([0,1,0,1,0,1])
    p=np.array([.1,.5,.5,.6,.2,.5])
    weights=np.array([2,3,0,1,4,2])
    expanded=np.repeat(np.arange(len(y)),weights)
    assert np.isclose(_weighted_auc(_auc_plan(y,p),weights),roc_auc_score(y[expanded],p[expanded]))


@pytest.mark.parametrize("labels,probabilities,expected", [
    ([0, 0, 0, 1, 1], [.1, .2, .9, .8, .7], 1/3),
    ([0, 0, 1, 1], [.5, .5, .5, .5], .5),
    ([0, 0, 1, 1], [.1, .2, .8, .9], 0),
    ([0, 0, 1, 1], [.9, .8, .2, .1], 1),
    ([0, 0, 1, 1], [.1, .6, .6, .9], .25),
])
def test_eer_interpolates_the_empirical_roc_crossing(labels, probabilities, expected):
    result = metrics(labels, probabilities)
    assert result["eer"] == pytest.approx(expected)
    assert result["eer_convention"] == (
        "linear empirical ROC interpolation at FPR=FNR; all thresholds retained"
    )

