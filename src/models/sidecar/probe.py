"""
src/models/sidecar/probe.py

EXISTING technique (logistic regression on frozen activations = "linear
probe", Section 2 term definition), NEW integration for our 5-role
vocabulary. Architecture component: C7.

Trains a multiclass logistic-regression probe on activations extracted at
a fixed layer (src/models/sidecar/hooks.py) from the wrapper construction
(wrapper_construction.py). The probe's predict_proba for the USER and
REASONING classes are exactly Ye et al.'s "Userness" and "CoTness" readouts
(Section 2: "Userness, CoTness) as probe output probabilities").
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from src.ingestion.schema import ROLE_ORDER, Role
from src.models.sidecar.hooks import extract_activations


def train_probe(activations: torch.Tensor, labels: list[Role], seed: int = 42) -> LogisticRegression:
    X = activations.cpu().numpy()
    y = [ROLE_ORDER.index(r) for r in labels]
    # NOTE: `multi_class="multinomial"` was removed in scikit-learn 1.9 (this
    # env's version) -- multinomial is now automatic for a multi-class `y`
    # with the default 'lbfgs' solver, so no argument is needed.
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X, y)
    return clf


def probe_readout(clf: LogisticRegression, activations: torch.Tensor) -> np.ndarray:
    """Returns (n, len(ROLE_ORDER)) probability matrix -- column order
    matches ROLE_ORDER, so readout[:, ROLE_TO_IDX[Role.USER]] is
    "Userness" and readout[:, ROLE_TO_IDX[Role.REASONING]] is "CoTness",
    matching Ye et al.'s named readouts exactly.

    Always shaped (n, len(ROLE_ORDER)) regardless of how many distinct
    classes the probe actually saw at fit time -- if a role was absent from
    training data, its column is all zeros rather than the array being
    narrower than ROLE_ORDER (which crashed on any missing class; caught by
    tests/test_sidecar_probe.py before it could bite the real 5-class run).
    """
    X = activations.cpu().numpy()
    proba = clf.predict_proba(X)  # (n, n_classes_seen), classes_ gives the mapping
    reindexed = np.zeros((X.shape[0], len(ROLE_ORDER)), dtype=proba.dtype)
    for col, cls in enumerate(clf.classes_):
        reindexed[:, cls] = proba[:, col]
    return reindexed


def readout_for_role(readout: np.ndarray, role: Role) -> np.ndarray:
    return readout[:, ROLE_ORDER.index(role)]
