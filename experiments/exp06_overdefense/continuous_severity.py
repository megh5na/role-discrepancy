"""
experiments/exp06_overdefense/continuous_severity.py

Continuous alternatives to the argmax severity lookup in src/scoring/asymmetry.py.
STATUS (2026-10-04): cont_exp was ADOPTED into src/scoring/asymmetry.py as
`expected_severity` / `combined_score(severity_mode="expected")` and is re-exported
here unchanged, so every analysis script and result file keeps working. cont_max
(sensitivity variant, rejected as worse) stays here only.

The shipped scorer is  (1 - P(declared)) x S[argmax role, declared]. The argmax
makes the multiplier jump (2.5x, or to 0 when the argmax equals the declared
role even if P(declared) is only ~0.45) when two roles are close, which is the
suspected source of seed fragility. These variants use the SAME severity matrix
S (unchanged, hand-set a priori) with the full probability vector:

  cont_exp   sum_r P(r) * S[r, declared]            (expected severity; S[d,d]=0)
  cont_max   max_r  P(r) * S[r, declared]            (largest weighted severity)

Properties (tested): if all probability mass sits on one role r != declared the
score equals S[r, declared], i.e. the same as the shipped scorer; all mass on
the declared role scores 0; both are continuous in the probabilities.
cont_exp is the pre-specified primary variant, cont_max a sensitivity variant;
both are reported, neither was chosen after looking at results.
"""
from __future__ import annotations

import numpy as np

from src.ingestion.schema import ROLE_ORDER, Role
from src.scoring.asymmetry import expected_severity, severity_matrix


_M = severity_matrix()


def cont_exp(probs: np.ndarray, declared: Role) -> np.ndarray:
    return expected_severity(probs, declared)


def cont_max(probs: np.ndarray, declared: Role) -> np.ndarray:
    return (probs * _M[:, ROLE_ORDER.index(declared)]).max(axis=1)


CONTINUOUS = {"cont_exp": cont_exp, "cont_max": cont_max}
