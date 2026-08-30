"""
src/evaluation/significance.py

EXISTING statistical techniques (paired bootstrap, McNemar's test), NEW
application. Architecture component: C10/F8 (Section 8, Section 15
"STATISTICAL TESTING").

Why this exists, quoting the spec directly: "Paired bootstrap over test
samples for TPR differences between our method and each baseline; report
confidence intervals, not just point estimates" and "McNemar's test for
paired binary decisions where appropriate." Single point-estimate
comparisons ("we got 0.71, they got 0.68") are not defensible in a viva
without an uncertainty estimate -- especially given Fomin's documented high
seed variance in this exact setting (Section 15).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def paired_bootstrap_ci(
    values_a: np.ndarray | list[float],
    values_b: np.ndarray | list[float],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap CI for the PAIRED difference mean(a) - mean(b), resampling
    (a_i, b_i) pairs together (not independently) -- appropriate when a and
    b are two detectors' scores on the SAME underlying test samples."""
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"paired arrays must be equal length, got {len(a)} vs {len(b)}")
    n = len(a)
    rng = np.random.default_rng(seed)

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = np.mean(a[idx]) - np.mean(b[idx])

    alpha = 1 - ci
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    point_estimate = float(np.mean(a) - np.mean(b))
    return {
        "point_estimate": point_estimate,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ci_level": ci,
        "n_boot": n_boot,
        "significant": bool(lo > 0 or hi < 0),  # CI excludes zero
    }


def mcnemar_test(correct_a: list[bool], correct_b: list[bool]) -> dict:
    """McNemar's test for two PAIRED binary classifiers (same test items).
    Tests whether the DISAGREEMENTS are asymmetric -- i.e. whether one
    detector is significantly more often right-when-the-other-is-wrong than
    the reverse, which is what actually matters for a paired comparison
    (unlike comparing raw accuracies independently)."""
    if len(correct_a) != len(correct_b):
        raise ValueError("paired arrays must be equal length")
    b = sum(1 for a_i, b_i in zip(correct_a, correct_b) if a_i and not b_i)  # a right, b wrong
    c = sum(1 for a_i, b_i in zip(correct_a, correct_b) if not a_i and b_i)  # a wrong, b right

    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0}

    # Exact binomial form (more reliable than chi-square for small b+c).
    result = stats.binomtest(min(b, c), n=b + c, p=0.5)
    return {"b": b, "c": c, "statistic": float(min(b, c)), "p_value": float(result.pvalue)}
