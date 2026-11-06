"""
Paired significance tests for two classifiers scored on the same specimens.

    paired_bootstrap_test
        Difference of two per-replicate metric vectors computed on shared
        resamples (``utils.bootstrap``). With the participant scheme it respects
        donor clustering, so this is the test to report.

    delong_test
        DeLong test for two correlated AUROCs. Closed form, so it resolves
        p-values below the bootstrap floor, but it assumes independent
        observations and has no way to express donor clustering (Tb: 140
        specimens from 67 donors): its standard errors are anti-conservative
        wherever a donor contributes more than one specimen. Use it as a
        cross-check on cohorts that are effectively one specimen per donor.

    bh_fdr
        Benjamini-Hochberg adjusted p-values across a family of tests.
"""

import numpy as np
from scipy import stats


def paired_bootstrap_test(dist_a, dist_b, point_a, point_b, alpha=0.05):
    """Paired percentile-bootstrap test of metric(a) - metric(b) = 0.

    dist_a, dist_b: length-B metric vectors evaluated on the same index set.
    point_a, point_b: full-sample metric values, used for the reported delta.

    The p-value is the two-sided CI-inversion value (p < alpha exactly when the
    percentile CI excludes 0), with the Davison-Hinkley (+1)/(B+1) correction so
    it is never exactly 0; its floor is 2/(B+1).
    """
    d = dist_a - dist_b
    b = d.size
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_lo = (np.count_nonzero(d <= 0) + 1) / (b + 1)
    p_hi = (np.count_nonzero(d >= 0) + 1) / (b + 1)
    return {
        "delta": point_a - point_b,
        "lo": float(lo),
        "hi": float(hi),
        "p": float(min(2 * min(p_lo, p_hi), 1.0)),
        "p_floor": 2 / (b + 1),
        "boot_sd": float(d.std(ddof=1)),
    }


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        prev = min(prev, p[i] * n / (n - rank + 1))
        adj[i] = prev
    return adj


def midrank(x):
    """Ranks of x with ties assigned their mean rank (1-based)."""
    order = np.argsort(x, kind="mergesort")
    z = x[order]
    n = z.size
    t = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and z[j + 1] == z[i]:
            j += 1
        t[i : j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    out = np.empty(n, dtype=float)
    out[order] = t
    return out


def delong_auc_cov(y, scores):
    """AUROCs and their covariance matrix for k classifiers on shared labels.

    Fast implementation via midranks (Sun & Xu 2014, IEEE SPL 21(11):1389-1393),
    algebraically identical to DeLong, DeLong & Clarke-Pearson (1988) but
    O(n log n) instead of O(n^2). Midranks make it exact under tied scores, which
    matters here: several methods emit heavily tied scores (Ostmeyer RA has 4
    distinct values across 91 samples).

    y: (n,) 0/1. scores: (k, n) positive-class scores.
    Returns (aucs (k,), S (k, k)) where S is the covariance of the AUC estimates.
    """
    y = np.asarray(y).astype(int)
    scores = np.atleast_2d(np.asarray(scores, dtype=float))
    pos = y == 1
    m, n = int(pos.sum()), int((~pos).sum())
    if m == 0 or n == 0:
        raise ValueError("need both classes present")
    k = scores.shape[0]

    v10 = np.empty((k, m))  # placement values of positives against negatives
    v01 = np.empty((k, n))  # placement values of negatives against positives
    aucs = np.empty(k)
    for r in range(k):
        x, z_neg = scores[r, pos], scores[r, ~pos]
        tx, ty = midrank(x), midrank(z_neg)
        tz = midrank(np.concatenate([x, z_neg]))
        aucs[r] = (tz[:m].sum() - m * (m + 1) / 2.0) / (m * n)
        v10[r] = (tz[:m] - tx) / n
        v01[r] = 1.0 - (tz[m:] - ty) / m

    s10 = np.atleast_2d(np.cov(v10, ddof=1))
    s01 = np.atleast_2d(np.cov(v01, ddof=1))
    return aucs, s10 / m + s01 / n


def delong_test(y, score_a, score_b, alpha=0.05):
    """Two-sided DeLong test of AUROC(a) - AUROC(b) = 0."""
    aucs, s = delong_auc_cov(y, np.vstack([score_a, score_b]))
    delta = float(aucs[0] - aucs[1])
    var = float(s[0, 0] + s[1, 1] - 2 * s[0, 1])
    if var <= 0:
        # identical or perfectly correlated placement values; no testable contrast
        return {
            "auc_a": float(aucs[0]),
            "auc_b": float(aucs[1]),
            "delta": delta,
            "se": 0.0,
            "z": np.nan,
            "p": 1.0,
            "lo": delta,
            "hi": delta,
        }
    se = np.sqrt(var)
    z = delta / se
    crit = stats.norm.ppf(1 - alpha / 2)
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "delta": delta,
        "se": float(se),
        "z": float(z),
        "p": float(2 * stats.norm.sf(abs(z))),
        "lo": float(delta - crit * se),
        "hi": float(delta + crit * se),
    }
