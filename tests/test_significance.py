"""tests/test_significance.py -- C10/F8 significance testing."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.significance import mcnemar_test, paired_bootstrap_ci  # noqa: E402


def test_bootstrap_ci_detects_real_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(0.8, 0.05, 200)
    b = rng.normal(0.6, 0.05, 200)
    result = paired_bootstrap_ci(a, b, n_boot=1000, seed=1)
    assert result["point_estimate"] > 0.15
    assert result["significant"] is True
    assert result["ci_low"] > 0


def test_bootstrap_ci_no_difference_not_significant():
    rng = np.random.default_rng(0)
    a = rng.normal(0.7, 0.05, 200)
    b = a.copy()  # identical -- paired difference is exactly zero
    result = paired_bootstrap_ci(a, b, n_boot=1000, seed=1)
    assert abs(result["point_estimate"]) < 1e-9
    assert result["significant"] is False


def test_mcnemar_symmetric_disagreement_not_significant():
    # Equal numbers of "a right b wrong" and "a wrong b right" -- no real asymmetry.
    correct_a = [True, False, True, False, True, False, True, False] * 5
    correct_b = [False, True, False, True, False, True, False, True] * 5
    result = mcnemar_test(correct_a, correct_b)
    assert result["b"] == result["c"]
    assert result["p_value"] > 0.5


def test_mcnemar_asymmetric_disagreement_is_significant():
    rng = np.random.default_rng(0)
    n = 200
    # a right, b wrong in 80 cases; a wrong, b right in only 5 cases -- strong asymmetry.
    correct_a = [True] * 80 + [False] * 5 + [True] * 60 + [False] * 55
    correct_b = [False] * 80 + [True] * 5 + [True] * 60 + [False] * 55
    result = mcnemar_test(correct_a, correct_b)
    assert result["p_value"] < 0.01


def test_mcnemar_no_disagreements_returns_p_one():
    correct_a = [True, True, False, False]
    correct_b = [True, True, False, False]
    result = mcnemar_test(correct_a, correct_b)
    assert result["p_value"] == 1.0
