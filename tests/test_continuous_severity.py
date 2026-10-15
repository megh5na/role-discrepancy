"""tests/test_continuous_severity.py -- continuous severity variants (E6 follow-up), offline."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.ablation_severity.analyze_severity_ablation import score_variant  # noqa: E402
from experiments.exp06_overdefense.continuous_severity import cont_exp, cont_max, severity_matrix  # noqa: E402
from src.ingestion.schema import ROLE_ORDER, Role  # noqa: E402
from src.scoring.asymmetry import ASYMMETRIC_SEVERITY  # noqa: E402


def _onehot(role):
    v = np.zeros(len(ROLE_ORDER)); v[ROLE_ORDER.index(role)] = 1.0
    return v[None, :]


def test_matrix_matches_project_severity_and_has_zero_diagonal():
    M = severity_matrix()
    for i, p in enumerate(ROLE_ORDER):
        for j, d in enumerate(ROLE_ORDER):
            assert M[i, j] == ASYMMETRIC_SEVERITY[(p, d)]
    assert np.all(np.diag(M) == 0)


@pytest.mark.parametrize("fn", [cont_exp, cont_max])
def test_all_mass_on_declared_scores_zero(fn):
    for d in Role:
        assert fn(_onehot(d), d)[0] == 0.0


@pytest.mark.parametrize("fn", [cont_exp, cont_max])
def test_all_mass_on_one_other_role_equals_the_shipped_scorer(fn):
    for p in Role:
        for d in Role:
            if p == d:
                continue
            assert fn(_onehot(p), d)[0] == pytest.approx(score_variant(_onehot(p), "prob_deficit|asym", declared=d)[0])


def test_continuity_no_cliff_when_two_roles_are_close():
    # declared document; P(document) just above vs just below P(user): the shipped
    # scorer jumps from 0 to a large value, the continuous scores move smoothly
    above = np.array([[0.0, 0.40, 0.41, 0.0, 0.19]])   # ROLE_ORDER: system,user,document,tool,reasoning
    below = np.array([[0.0, 0.41, 0.40, 0.0, 0.19]])
    shipped = [score_variant(x, "prob_deficit|asym", declared=Role.DOCUMENT)[0] for x in (above, below)]
    assert shipped[0] == 0.0 and shipped[1] > 0.3          # the cliff
    for fn in (cont_exp, cont_max):
        a, b = fn(above, Role.DOCUMENT)[0], fn(below, Role.DOCUMENT)[0]
        assert abs(a - b) < 0.02


def test_cont_exp_hand_computed_example():
    # declared document: S[user]=1.0, S[reasoning]=0.7, S[tool]=0.4 (default), S[system]=1.0
    p = np.array([[0.1, 0.5, 0.2, 0.1, 0.1]])
    expected = 0.1 * 1.0 + 0.5 * 1.0 + 0.2 * 0.0 + 0.1 * 0.4 + 0.1 * 0.7
    assert cont_exp(p, Role.DOCUMENT)[0] == pytest.approx(expected)
    assert cont_max(p, Role.DOCUMENT)[0] == pytest.approx(0.5)


def test_vectorised_over_rows():
    p = np.stack([_onehot(Role.USER)[0], _onehot(Role.DOCUMENT)[0], _onehot(Role.TOOL)[0]])
    assert cont_exp(p, Role.DOCUMENT).tolist() == [1.0, 0.0, 0.4]
