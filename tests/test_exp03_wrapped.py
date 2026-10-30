import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp03_baselines.run_exp03_wrapped import (  # noqa: E402
    NOVEL_TEMPLATES,
    bootstrap,
    build_conditions,
    groups_of,
    metrics_for,
    summarize,
    unique_in_order,
)

TEXTS = [f"Please do thing number {i} now." for i in range(12)]


def test_unique_in_order_keeps_first_occurrence_order():
    assert unique_in_order(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]


def test_conditions_cover_bare_known_and_novel():
    conds = build_conditions(TEXTS)
    assert "bare" in conds
    assert sum(c.startswith("known_") for c in conds) == 5
    assert sum(c.startswith("novel_") for c in conds) == len(NOVEL_TEMPLATES)
    assert all(len(v) == len(TEXTS) for v in conds.values())


def test_bare_is_unchanged_and_novel_templates_contain_the_span():
    conds = build_conditions(TEXTS)
    assert conds["bare"] == TEXTS
    for name in NOVEL_TEMPLATES:
        assert all(t in w for t, w in zip(TEXTS, conds[f"novel_{name}"]))


def test_wrapping_is_deterministic():
    assert build_conditions(TEXTS) == build_conditions(TEXTS)
    assert build_conditions(TEXTS, seed=11) != build_conditions(TEXTS, seed=12)


def test_novel_templates_are_not_the_training_templates():
    # the whole point of the novel set is that neither checkpoint trained on it
    from src.supervision import carriers

    trained = {t for ts in carriers._TEMPLATES_BY_CHANNEL.values() for t in ts}
    assert not (set(NOVEL_TEMPLATES.values()) & trained)


def test_groups_split_known_and_novel():
    g = groups_of(list(build_conditions(TEXTS)))
    assert len(g["known_avg"]) == 5 and len(g["novel_avg"]) == len(NOVEL_TEMPLATES)


def test_metrics_for_separates_fixed_and_recalibrated_threshold():
    rng = np.random.default_rng(0)
    bare_b = rng.normal(0, 1, 500)
    ben = bare_b + 5.0            # wrapping pushes every benign score up
    inj = rng.normal(8, 1, 100)   # injected also shifted
    m = metrics_for(bare_b, ben, inj, 0.05)
    # a threshold fixed on bare benign flags (almost) all shifted benign traffic
    assert m["A_fpr_on_wrapped_benign"] > 0.9
    # recalibrating on the shifted benign restores a sensible operating point
    assert m["B_auroc"] > 0.95


def _synthetic_scores(rng, gap_by_det, n_b=150, n_i=60):
    conds = list(build_conditions(["x"]))
    scores = {}
    for det, gap in gap_by_det.items():
        scores[det] = {}
        for c in conds:
            scores[det][c] = (rng.normal(0, 1, n_b), rng.normal(gap, 1, n_i))
    return scores, conds


def test_identical_detectors_have_zero_paired_difference():
    rng = np.random.default_rng(1)
    scores, conds = _synthetic_scores(rng, {"ours_naive": 2.0})
    scores["ours_transplant"] = scores["ours_naive"]
    point, draws = bootstrap(scores, conds, n_boot=30, seed=1)
    res = summarize(point, draws)
    for unit in ("bare", "known_avg", "novel_avg"):
        p = res["paired_transplant_minus_naive"][unit]["0.05"]["B_auroc"]
        assert p["point"] == 0.0 and p["ci95"] == [0.0, 0.0]


def test_stronger_detector_is_detected_as_better_in_paired_diff():
    rng = np.random.default_rng(2)
    scores, conds = _synthetic_scores(rng, {"ours_naive": 0.3, "ours_transplant": 3.0})
    point, draws = bootstrap(scores, conds, n_boot=60, seed=2)
    p = summarize(point, draws)["paired_transplant_minus_naive"]["known_avg"]["0.05"]["B_auroc"]
    assert p["point"] > 0.3 and p["ci_excludes_zero"]
    # every metric key present for the point and for the draws
    assert set(point) == set(draws)
