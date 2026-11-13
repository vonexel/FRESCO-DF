from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, log_loss


def auc_delong(y, scores, alpha=0.05):
    y, scores = np.asarray(y), np.asarray(scores)
    pos, neg = np.sort(scores[y == 1]), np.sort(scores[y == 0])
    if min(len(pos), len(neg)) < 2:
        return [None, None]
    vp = (np.searchsorted(neg, pos, side="left") + np.searchsorted(neg, pos, side="right")) / (2 * len(neg))
    vn = 1 - (np.searchsorted(pos, neg, side="left") + np.searchsorted(pos, neg, side="right")) / (2 * len(pos))
    variance = np.var(vp, ddof=1) / len(pos) + np.var(vn, ddof=1) / len(neg)
    mean = vp.mean()
    delta = norm.ppf(1-alpha/2) * np.sqrt(max(variance, 0))
    return [float(max(0, mean-delta)), float(min(1, mean+delta))]


def reliability(y, p, bins=15):
    y, p = np.asarray(y), np.asarray(p)
    ids = np.minimum((p*bins).astype(int), bins-1)
    result = []
    for i in range(bins):
        mask = ids == i
        if mask.any():
            result.append({"bin": i, "count": int(mask.sum()), "confidence": float(p[mask].mean()),
                           "frequency": float(y[mask].mean())})
    return result


def metrics(y, p, threshold=0.5):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    if not len(y) or not np.isfinite(p).all():
        raise ValueError("Empty/nonfinite predictions")
    if len(np.unique(y)) < 2:
        return {"n": len(y), "auc": None, "reason": "single class; supply shared real controls"}
    fpr, tpr, _ = roc_curve(y, p, drop_intermediate=False)

    # the nearest threshold's mean(FPR, FNR) is not an equal-error rate on a stepwise ROC
    difference = fpr - (1-tpr)
    right = int(np.flatnonzero(difference >= 0)[0])
    left = right - 1
    weight = -difference[left] / (difference[right]-difference[left])
    eer = float(fpr[left] + weight * (fpr[right]-fpr[left]))
    rel = reliability(y, p)
    return {"n": len(y), "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum()),
            "auc": float(roc_auc_score(y, p)), "auc_ci_delong_video": auc_delong(y, p),
            "ap": float(average_precision_score(y, p)), "eer": eer,
            "eer_convention": "linear empirical ROC interpolation at FPR=FNR; all thresholds retained",
            "ece": sum(r["count"]*abs(r["confidence"]-r["frequency"]) for r in rel)/len(y),
            "brier": float(np.mean((p-y)**2)), "nll": float(log_loss(y, p, labels=[0, 1])),
            "fpr_at_validation_threshold": float((p[y == 0] >= threshold).mean()),
            "tpr_at_validation_threshold": float((p[y == 1] >= threshold).mean()),
            "threshold": float(threshold)}


def fit_temperature(y, logits):
    y, logits = np.asarray(y), np.asarray(logits)
    def objective(log_t):
        return np.mean(np.logaddexp(0, logits/np.exp(log_t))-y*logits/np.exp(log_t))
    opt = minimize_scalar(objective, bounds=(-3, 3), method="bounded")
    return float(np.exp(opt.x))


def validation_threshold(y, p, target_fpr=0.05):
    real = np.sort(np.asarray(p)[np.asarray(y) == 0])
    return float(np.nextafter(np.quantile(real, 1-target_fpr, method="higher"), np.inf))


def _auc_plan(y, scores):
    order=np.argsort(scores,kind="stable")
    sorted_scores=np.asarray(scores)[order]
    starts=np.r_[0,np.flatnonzero(np.diff(sorted_scores)!=0)+1]
    return order,starts,np.asarray(y)[order]


def _weighted_auc(plan, weights):
    # the repeated-video AUC
    order,starts,labels=plan
    w=weights[order]
    positive=np.add.reduceat(w*labels,starts)
    negative=np.add.reduceat(w*(1-labels),starts)
    denominator=positive.sum()*negative.sum()
    if denominator==0:
        return np.nan
    return np.sum(positive*(np.cumsum(negative)-negative*.5))/denominator


def paired_auc_bootstrap(y, a, b, groups=None, resamples=10000, seed=2026, margin=.01):
    """Paired percentile CIs, 90% for alpha=.05 TOST and 95% for differences.
    Inputs are aligned [seed,video] scores (or a single [video] vector).
    Statistic is MEAN of seed-specific AUC differences, never AUC of an ensemble.
    Groups resample source/identity clusters together. With few fixed seeds this
    estimates test-sampling uncertainty CONDITIONAL on those seeds; it does not
    estimate population-over-initializations uncertainty.
    """
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    if a.shape != b.shape or a.shape[1] != len(y):
        raise ValueError("Require paired scores [seed,video] with identical shape")
    groups = np.arange(len(y)).astype(str) if groups is None else np.asarray(groups).astype(str)
    unique,inverse = np.unique(groups,return_inverse=True)
    if len(unique) < 5:
        return {"status": "INCONCLUSIVE", "reason": "fewer than five independent clusters"}
    plans=[(_auc_plan(y,aa),_auc_plan(y,bb)) for aa,bb in zip(a,b)]
    rng, values = np.random.default_rng(seed), []
    for _ in range(resamples):
        counts = np.bincount(rng.integers(0,len(unique),len(unique)), minlength = len(unique))
        weights = counts[inverse]
        differences = [_weighted_auc(pa,weights)-_weighted_auc(pb,weights) for pa,pb in plans]
        if np.isfinite(differences).all():
            values.append(np.mean(differences))
    if len(values) < .9*resamples:
        return {"status": "INCONCLUSIVE", "reason": "degenerate bootstrap draws"}
    ci90, ci95 = np.quantile(values, [.05, .95]), np.quantile(values, [.025, .975])
    differences = [roc_auc_score(y, aa)-roc_auc_score(y, bb) for aa, bb in zip(a, b)]
    return {"difference": float(np.mean(differences)), "seed_differences": differences,
            "uncertainty": "test-cluster sampling conditional on fixed seeds",
            "ci90": ci90.tolist(), "ci95": ci95.tolist(), "margin": margin,
            "equivalent_tost_percentile": bool(ci90[0] > -margin and ci90[1] < margin),
            "superior_beyond_margin": bool(ci95[0] > margin), "clusters": len(unique),
            "resamples_valid": len(values), "unit": "source/identity cluster; paired videos"}