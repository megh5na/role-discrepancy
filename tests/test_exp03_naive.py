"""tests/test_exp03_naive.py -- E3-naive comparison helpers, offline
(synthetic scores, no model)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp03_baselines.run_exp03_naive import (  # noqa: E402
    auroc,
    auroc_diff_ci,
    paired_flag_test,
)


def _separable(rng, gap, n=400):
    benign = rng.normal(0.0, 1.0, n)
    injected = rng.normal(gap, 1.0, n)
    return benign, injected


def test_auroc_perfect_and_chance():
    assert auroc(np.zeros(50), np.ones(50)) == 1.0
    rng = np.random.default_rng(0)
    a = rng.normal(size=2000)
    b = rng.normal(size=2000)
    assert auroc(a, b) == pytest.approx(0.5, abs=0.05)


def test_stronger_detector_wins_the_paired_flag_test():
    rng = np.random.default_rng(1)
    ba, ia = _separable(rng, gap=3.0)   # strong
    bb, ib = _separable(rng, gap=0.5)   # weak
    t = paired_flag_test(ba, ia, bb, ib, fpr=0.05)
    assert t["tpr_a"] > t["tpr_b"]
    assert t["diff_a_minus_b"] > 0
    assert t["diff_ci95"][0] > 0          # CI excludes zero
    assert t["mcnemar_p"] < 0.001
    # direction of the discordant counts must match the sign of the difference
    assert t["a_only_flagged"] > t["b_only_flagged"]


def test_identical_detectors_are_not_significantly_different():
    rng = np.random.default_rng(2)
    b, i = _separable(rng, gap=1.5)
    t = paired_flag_test(b, i, b.copy(), i.copy(), fpr=0.05)
    assert t["diff_a_minus_b"] == 0.0
    assert t["mcnemar_p"] == 1.0


def test_flag_test_calibrates_each_detector_on_its_own_benign_scores():
    # detector b's scores are a shifted copy of a's; thresholds are per-detector,
    # so a pure monotone shift must not change TPR at a matched FPR
    rng = np.random.default_rng(3)
    b, i = _separable(rng, gap=2.0)
    t = paired_flag_test(b, i, b + 100.0, i + 100.0, fpr=0.05)
    assert t["tpr_a"] == pytest.approx(t["tpr_b"])


def test_auroc_diff_ci_sign_and_zero_case():
    rng = np.random.default_rng(4)
    ba, ia = _separable(rng, gap=3.0, n=200)
    bb, ib = _separable(rng, gap=0.3, n=200)
    d = auroc_diff_ci(ba, ia, bb, ib, n_boot=200)
    assert d["diff"] > 0 and d["ci95"][0] > 0
    same = auroc_diff_ci(ba, ia, ba, ia, n_boot=100)
    assert same["diff"] == 0.0 and same["ci95"] == [0.0, 0.0]
