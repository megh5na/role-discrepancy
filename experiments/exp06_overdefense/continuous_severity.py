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
