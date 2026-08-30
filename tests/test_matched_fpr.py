"""tests/test_matched_fpr.py -- C10 matched-FPR harness."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.matched_fpr import matched_fpr_comparison  # noqa: E402


def test_matched_fpr_achieves_target_on_benign():
    rng = np.random.default_rng(0)
    benign = rng.uniform(0, 1, 2000)
    injected = rng.uniform(0.3, 1.3, 2000)  # shifted higher -- a real detector

    results = matched_fpr_comparison({"detector_a": (benign, injected)}, target_fprs=[0.01, 0.05])
    for fpr in [0.01, 0.05]:
        achieved = results["detector_a"][fpr]["achieved_fpr"]
        assert abs(achieved - fpr) < 0.02


def test_better_detector_has_higher_tpr_at_same_fpr():
    rng = np.random.default_rng(1)
    benign = rng.uniform(0, 1, 2000)
    weak_injected = rng.uniform(0, 1.1, 2000)      # barely separated from benign
    strong_injected = rng.uniform(0.8, 2.0, 2000)  # well separated

    results = matched_fpr_comparison(
        {"weak": (benign, weak_injected), "strong": (benign, strong_injected)},
        target_fprs=[0.05],
    )
    assert results["strong"][0.05]["tpr"] > results["weak"][0.05]["tpr"]


def test_perfect_separation_gives_tpr_near_one():
    rng = np.random.default_rng(2)
    benign = rng.uniform(0, 0.3, 1000)
    injected = rng.uniform(0.7, 1.0, 1000)
    results = matched_fpr_comparison({"perfect": (benign, injected)}, target_fprs=[0.05])
    assert results["perfect"][0.05]["tpr"] > 0.95
