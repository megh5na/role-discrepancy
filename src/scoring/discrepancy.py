"""
src/scoring/discrepancy.py

*** OUR CONTRIBUTION (NEW) *** — Section 6 NEW-2, Section 8 C5, Section 17.

Converts (perceived-role distribution, declared role) into a scalar mismatch
score. Ye et al. propose COMPARING declared and perceived role as an open
question but specify no scoring function (Section 6: "Ye et al. propose
comparing declared and perceived role. No scoring function specified.") --
this file is that missing piece.

Three candidate formulations are implemented, all differing in what to do
with the FULL distribution vs. just the declared-role probability, so the
choice is an ablation (Section 14 ablations list: "discrepancy formulation
variants") rather than a single unmotivated pick:

  neg_log_prob    -log P(declared role).  Standard, information-theoretic:
                  "how surprised should the declared-role slot be, given
                  what the text actually reads as." Unbounded above.
  prob_deficit    1 - P(declared role). Bounded [0, 1], easy to threshold
                  and to combine multiplicatively with severity weights
                  (src/scoring/asymmetry.py) -- the DEFAULT formulation.
  margin          max_{r != declared} P(r) - P(declared role). Positive
                  exactly when some OTHER role is more likely than the
                  declared one -- the most literal reading of "discrepancy":
                  is there a role this text sounds MORE like than the one
                  it's declared as.

WHAT THIS FILE DOES NOT DO: apply channel-pair severity (that's
asymmetry.py) or pick a decision threshold (that's calibration.py). Kept
separate per Section 8's own component split, and because each is a
different design decision that should be independently ablatable.
"""

from __future__ import annotations

import math

import numpy as np

from src.ingestion.schema import ROLE_ORDER, ROLE_TO_IDX, Role

_EPS = 1e-9


def neg_log_prob(perceived_probs: np.ndarray, declared_role: Role) -> float:
    p = float(perceived_probs[ROLE_TO_IDX[declared_role]])
    return -math.log(max(p, _EPS))


def prob_deficit(perceived_probs: np.ndarray, declared_role: Role) -> float:
    p = float(perceived_probs[ROLE_TO_IDX[declared_role]])
    return 1.0 - p


def margin(perceived_probs: np.ndarray, declared_role: Role) -> float:
    declared_idx = ROLE_TO_IDX[declared_role]
    p_declared = float(perceived_probs[declared_idx])
    others = [float(perceived_probs[i]) for i in range(len(ROLE_ORDER)) if i != declared_idx]
    return max(others) - p_declared


_FORMULATIONS = {
    "neg_log_prob": neg_log_prob,
    "prob_deficit": prob_deficit,
    "margin": margin,
}

DEFAULT_FORMULATION = "prob_deficit"


def base_discrepancy(
    perceived_probs: np.ndarray, declared_role: Role, formulation: str = DEFAULT_FORMULATION
) -> float:
    if formulation not in _FORMULATIONS:
        raise ValueError(f"Unknown formulation {formulation!r}; choices: {list(_FORMULATIONS)}")
    return _FORMULATIONS[formulation](perceived_probs, declared_role)


def perceived_role_argmax(perceived_probs: np.ndarray) -> Role:
    return ROLE_ORDER[int(np.argmax(perceived_probs))]
