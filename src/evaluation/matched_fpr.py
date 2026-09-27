"""
src/evaluation/matched_fpr.py

NEW harness / EXISTING protocol (Section 8 C10, Section 4/14/15: "MANDATORY.
Comparing raw accuracies across detectors with different default thresholds
is meaningless and a panel may catch it").

Given, for each detector, a set of scores on a shared BENIGN set and a set
of scores on an INJECTED set, computes TPR at each detector's OWN threshold
calibrated to a target FPR on the shared benign set. This is what makes
Experiment 3's comparison (our discrepancy scorer vs. PromptGuard 2 vs.
ProtectAI v2 vs. ...) fair: every detector is evaluated at the SAME
false-positive budget, not at whatever its own default cutoff happens to be.
"""

from __future__ import annotations

import numpy as np

from src.scoring.calibration import calibrate_threshold, empirical_fpr


def matched_fpr_comparison( # this function takes a dictionary of detector scores (benign and injected) and computes the TPR for each detector at specified target FPRs. It returns a nested dictionary with the results.
    detector_scores: dict[str, tuple[np.ndarray, np.ndarray]],
    target_fprs: list[float] = [0.01, 0.05],
) -> dict:
    """
    detector_scores: {detector_name: (benign_scores, injected_scores)}
        Higher score = more likely flagged as injection, for every detector
        (callers must normalise sign/direction before calling this).

    Returns: {detector_name: {target_fpr: {"threshold":, "tpr":, "achieved_fpr":}}}
    """
    results: dict[str, dict] = {}
    for name, (benign, injected) in detector_scores.items():
        benign = np.asarray(benign, dtype=float)
        injected = np.asarray(injected, dtype=float)
        results[name] = {}
        for target_fpr in target_fprs:
            threshold = calibrate_threshold(benign, target_fpr) # this line calculates the threshold score for the detector such that the proportion of benign examples that exceed this threshold is at most the target false positive rate (FPR). This ensures that the detector is calibrated to flag injections while maintaining a controlled rate of false positives on benign data.
            tpr = float(np.mean(injected >= threshold)) if len(injected) else float("nan")
            achieved_fpr = empirical_fpr(benign, threshold)
            results[name][target_fpr] = {
                "threshold": threshold,
                "tpr": tpr,
                "achieved_fpr": achieved_fpr,
                "n_benign": len(benign),
                "n_injected": len(injected),
            }
    return results


def print_matched_fpr_table(results: dict, target_fprs: list[float] = [0.01, 0.05]) -> None:
    header = f"{'detector':<25}" + "".join(f"TPR@{fpr:.0%}".ljust(14) for fpr in target_fprs)
    print(header)
    for name, per_fpr in results.items():
        row = f"{name:<25}"
        for fpr in target_fprs:
            row += f"{per_fpr[fpr]['tpr']:.3f}".ljust(14)
        print(row)
