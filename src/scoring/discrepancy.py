from __future__ import annotations

import math

import numpy as np

from src.ingestion.schema import ROLE_ORDER, ROLE_TO_IDX, Role

_EPS = 1e-9


def neg_log_prob(perceived_probs: np.ndarray, declared_role: Role) -> float: # how surprised should the declared-role slot be, given what the text actually reads as.
    p = float(perceived_probs[ROLE_TO_IDX[declared_role]])
    return -math.log(max(p, _EPS))


def prob_deficit(perceived_probs: np.ndarray, declared_role: Role) -> float: # how much probability mass is missing from the declared role (1 - P(declared role)). Bounded [0, 1], easy to threshold and to combine multiplicatively with severity weights (src/scoring/asymmetry.py) -- the DEFAULT formulation.
    p = float(perceived_probs[ROLE_TO_IDX[declared_role]])
    return 1.0 - p


def margin(perceived_probs: np.ndarray, declared_role: Role) -> float: # is there a role this text sounds MORE like than the one it's declared as. Positive exactly when some OTHER role is more likely than the declared one.
    declared_idx = ROLE_TO_IDX[declared_role]
    p_declared = float(perceived_probs[declared_idx])
    others = [float(perceived_probs[i]) for i in range(len(ROLE_ORDER)) if i != declared_idx]
    return max(others) - p_declared


_FORMULATIONS = {
    "neg_log_prob": neg_log_prob,
    "prob_deficit": prob_deficit,
    "margin": margin,
}

DEFAULT_FORMULATION = "prob_deficit" # bounded - easy to threshold


def base_discrepancy( # this function takes the perceived role probabilities and the declared role, and computes a scalar discrepancy score based on the specified formulation (neg_log_prob, prob_deficit, or margin). It raises an error if an unknown formulation is provided.
    perceived_probs: np.ndarray, declared_role: Role, formulation: str = DEFAULT_FORMULATION
) -> float:
    if formulation not in _FORMULATIONS:
        raise ValueError(f"Unknown formulation {formulation!r}; choices: {list(_FORMULATIONS)}")
    return _FORMULATIONS[formulation](perceived_probs, declared_role)


def perceived_role_argmax(perceived_probs: np.ndarray) -> Role:
    return ROLE_ORDER[int(np.argmax(perceived_probs))]
