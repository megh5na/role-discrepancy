import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.ablation_severity.analyze_severity_ablation import VARIANTS, score_variant  # noqa: E402
from src.ingestion.schema import ROLE_ORDER, Role  # noqa: E402
from src.scoring.asymmetry import combined_score  # noqa: E402


def _p(**kw):
    """probability row in ROLE_ORDER from role=prob keywords (rest split evenly)."""
    v = np.array([kw.get(r.value, 0.0) for r in ROLE_ORDER], dtype=float)
    rest = 1.0 - v.sum()
    free = [i for i, r in enumerate(ROLE_ORDER) if r.value not in kw]
    v[free] = rest / len(free)
    return v


def test_default_variant_matches_the_project_combined_score():
    probs = np.stack([_p(user=0.7), _p(document=0.8), _p(tool=0.6), _p(system=0.5, user=0.3)])
    got = score_variant(probs, "prob_deficit|asym")
    want = [combined_score(p, Role.DOCUMENT, use_asymmetry=True, severity_mode="argmax")["score"] for p in probs]
    assert np.allclose(got, want)


def test_symmetric_matches_combined_score_with_symmetry_off():
    probs = np.stack([_p(user=0.7), _p(tool=0.6), _p(reasoning=0.55)])
    got = score_variant(probs, "prob_deficit|sym")
    want = [combined_score(p, Role.DOCUMENT, use_asymmetry=False)["score"] for p in probs]
    assert np.allclose(got, want)


def test_severity_none_equals_base_discrepancy():
    p = _p(user=0.7)
    assert score_variant(np.stack([p]), "prob_deficit|none")[0] == pytest.approx(1 - p[ROLE_ORDER.index(Role.DOCUMENT)])


def test_symmetric_zeroes_a_span_whose_argmax_is_the_declared_role_but_none_does_not():
    p = _p(document=0.45, user=0.40)  # argmax = document, but P(document) is only 0.45
    assert score_variant(np.stack([p]), "prob_deficit|sym")[0] == 0.0
    assert score_variant(np.stack([p]), "prob_deficit|none")[0] == pytest.approx(0.55)


def test_asymmetry_reweights_the_same_mismatch_by_channel_pair():
    # command-like perceived under DOCUMENT: severity 1.0; tool-like (no specific
    # rule, default 0.4): the same probability deficit scores lower
    user_like, tool_like = _p(user=0.7), _p(tool=0.7)
    a = score_variant(np.stack([user_like, tool_like]), "prob_deficit|asym")
    s = score_variant(np.stack([user_like, tool_like]), "prob_deficit|sym")
    assert s[0] == pytest.approx(s[1])           # symmetric cannot tell them apart
    assert a[0] > a[1]                           # asymmetric ranks the command-like one higher


def test_cmd_prob_ignores_declared_role_and_severity():
    p = _p(user=0.5, system=0.2)
    assert score_variant(np.stack([p]), "cmd_prob")[0] == pytest.approx(0.7)


def test_every_variant_runs_and_is_finite():
    probs = np.stack([_p(user=0.9), _p(document=0.9), _p(tool=0.5, reasoning=0.3)])
    for v in VARIANTS:
        out = score_variant(probs, v)
        assert out.shape == (3,) and np.all(np.isfinite(out))
