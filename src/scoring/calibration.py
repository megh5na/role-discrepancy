"""
src/scoring/calibration.py

EXISTING technique (empirical quantile thresholding), NEW application
(Section 6, "Threshold calibration on benign traffic at a target FPR").
Architecture component: C6 (Section 8).

Why a fixed threshold (e.g. 0.5) is indefensible here, quoting the spec
directly (Section 8 C6): "Fomin reports per-dataset optimal thresholds
ranging 0.01-0.73." Any single hardcoded cutoff would be arbitrary and
under-defended in a viva. Instead: pick the threshold empirically FROM
SCORES ON A HELD-OUT BENIGN SET, at whatever false-positive rate the
deployment wants to tolerate.
"""

from __future__ import annotations

import numpy as np


def calibrate_threshold(benign_scores: list[float] | np.ndarray, target_fpr: float) -> float:
    """Smallest threshold such that flagging score >= threshold on the given
    BENIGN scores produces false-positive rate <= target_fpr (empirical
    quantile at (1 - target_fpr))."""
    if not (0.0 < target_fpr < 1.0):
        raise ValueError(f"target_fpr must be in (0, 1), got {target_fpr}")
    scores = np.sort(np.asarray(benign_scores, dtype=float))
    if len(scores) == 0:
        raise ValueError("Cannot calibrate on an empty benign score set")
    idx = int(np.ceil((1.0 - target_fpr) * len(scores))) - 1
    idx = min(max(idx, 0), len(scores) - 1)
    return float(scores[idx])


def flag(score: float, threshold: float) -> bool:
    return score >= threshold


def empirical_fpr(benign_scores: list[float] | np.ndarray, threshold: float) -> float:
    """What FPR does this threshold actually achieve on this benign set --
    used to VERIFY calibration worked, and to report per-channel-pair FPRs
    (Experiment 6) by calling this per subgroup."""
    scores = np.asarray(benign_scores, dtype=float)
    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores >= threshold))
