"""tests/test_scoring.py -- C5 (discrepancy scorer) unit tests."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.schema import ROLE_ORDER, ROLE_TO_IDX, Role  # noqa: E402
from src.scoring import calibration, discrepancy  # noqa: E402
from src.scoring.asymmetry import (  # noqa: E402
    ASYMMETRIC_SEVERITY,
    SYMMETRIC_SEVERITY,
    combined_score,
    severity,
)


def _onehot(role: Role) -> np.ndarray:
    v = np.zeros(len(ROLE_ORDER))
    v[ROLE_TO_IDX[role]] = 1.0
    return v


def test_perfect_match_gives_zero_discrepancy():
    probs = _onehot(Role.USER)
    assert discrepancy.prob_deficit(probs, Role.USER) == 0.0
    assert discrepancy.neg_log_prob(probs, Role.USER) < 1e-6
    assert discrepancy.margin(probs, Role.USER) <= 0.0


def test_total_mismatch_gives_max_discrepancy():
    probs = _onehot(Role.TOOL)  # perceived TOOL with certainty
    d = discrepancy.prob_deficit(probs, Role.USER)  # declared USER
    assert d == 1.0
    m = discrepancy.margin(probs, Role.USER)
    assert m == 1.0  # tool is 100% more likely than declared user (0%)


def test_formulations_agree_on_ranking():
    # A near-uniform distribution should score lower discrepancy toward its
    # own peak than a distribution strongly peaked elsewhere.
    probs_close = np.array([0.1, 0.6, 0.1, 0.1, 0.1])  # peak at idx1
    probs_far = np.array([0.05, 0.05, 0.05, 0.8, 0.05])  # peak at idx3
    declared = ROLE_ORDER[1]
    for formulation in ["neg_log_prob", "prob_deficit", "margin"]:
        close_score = discrepancy.base_discrepancy(probs_close, declared, formulation)
        far_score = discrepancy.base_discrepancy(probs_far, declared, formulation)
        assert far_score > close_score, f"{formulation} did not rank mismatch higher"


def test_severity_matrix_covers_every_pair():
    for p in Role:
        for d in Role:
            assert (p, d) in ASYMMETRIC_SEVERITY
            assert (p, d) in SYMMETRIC_SEVERITY
            assert 0.0 <= ASYMMETRIC_SEVERITY[(p, d)] <= 1.0


def test_severity_diagonal_is_zero():
    for r in Role:
        assert severity(r, r, use_asymmetry=True) == 0.0
        assert severity(r, r, use_asymmetry=False) == 0.0


def test_asymmetry_core_injection_pattern_is_high_severity():
    # perceived=USER (command-like) sitting in a TOOL channel -- the classic
    # injection shape (Section 6) -- must be near-maximum severity.
    assert severity(Role.USER, Role.TOOL, use_asymmetry=True) == 1.0
    assert severity(Role.SYSTEM, Role.DOCUMENT, use_asymmetry=True) == 1.0


def test_asymmetry_benign_paste_pattern_is_low_severity():
    # perceived=DOCUMENT sitting in a USER channel -- pasting document text
    # into your own message -- must be near-floor severity (Section 6's own
    # worked example).
    assert severity(Role.DOCUMENT, Role.USER, use_asymmetry=True) < 0.2


def test_symmetric_severity_has_no_direction_preference():
    assert severity(Role.USER, Role.TOOL, use_asymmetry=False) == severity(
        Role.TOOL, Role.USER, use_asymmetry=False
    )


def test_combined_score_scales_with_severity():
    probs = np.array([0.05, 0.8, 0.05, 0.05, 0.05])  # confidently perceived USER
    declared = Role.TOOL  # high-severity direction
    result_asym = combined_score(probs, declared, use_asymmetry=True)
    result_sym = combined_score(probs, declared, use_asymmetry=False)
    assert result_asym["perceived_role"] == Role.USER
    assert result_asym["score"] > 0
    # Symmetric severity for a total mismatch (1.0) upper-bounds the asymmetric one here
    assert result_sym["score"] >= result_asym["score"] - 1e-9


def test_calibration_threshold_achieves_target_fpr_on_training_set():
    rng = np.random.default_rng(0)
    benign_scores = rng.uniform(0, 1, size=2000)
    for target_fpr in [0.01, 0.05, 0.1]:
        thr = calibration.calibrate_threshold(benign_scores, target_fpr)
        achieved = calibration.empirical_fpr(benign_scores, thr)
        # empirical FPR on the SAME set used to calibrate should be close to target
        assert abs(achieved - target_fpr) < 0.02, f"target={target_fpr} achieved={achieved}"


def test_calibration_flag_consistent_with_threshold():
    assert calibration.flag(0.9, 0.5) is True
    assert calibration.flag(0.1, 0.5) is False
    assert calibration.flag(0.5, 0.5) is True
