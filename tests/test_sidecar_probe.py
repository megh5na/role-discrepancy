"""tests/test_sidecar_probe.py -- offline test for the sidecar's linear
probe (catches sklearn-version regressions like the removed `multi_class`
kwarg, found running validate.py for the first time)."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.schema import ROLE_ORDER, Role  # noqa: E402
from src.models.sidecar.probe import probe_readout, readout_for_role, train_probe  # noqa: E402


def test_train_probe_multiclass_no_kwarg_error():
    """Regression test: LogisticRegression(multi_class=...) was removed in
    scikit-learn 1.9 (this env). train_probe must not pass it."""
    torch.manual_seed(0)
    n_per_class = 20
    hidden = 8
    acts = []
    labels = []
    for i, role in enumerate(ROLE_ORDER):
        # Well-separated synthetic clusters per class.
        cluster = torch.randn(n_per_class, hidden) + i * 10.0
        acts.append(cluster)
        labels.extend([role] * n_per_class)
    acts = torch.cat(acts, dim=0)

    clf = train_probe(acts, labels, seed=0)
    readout = probe_readout(clf, acts)
    assert readout.shape == (len(labels), len(ROLE_ORDER))
    # Well-separated clusters -- probe should recover them near-perfectly.
    preds = readout.argmax(axis=1)
    true_idx = [ROLE_ORDER.index(r) for r in labels]
    acc = sum(1 for p, t in zip(preds, true_idx) if p == t) / len(true_idx)
    assert acc > 0.95


def test_readout_for_role_selects_correct_column():
    torch.manual_seed(1)
    acts = torch.randn(10, 4)
    labels = [Role.USER, Role.DOCUMENT] * 5
    clf = train_probe(acts, labels, seed=1)
    readout = probe_readout(clf, acts)
    user_col = readout_for_role(readout, Role.USER)
    assert user_col.shape == (10,)
    assert (user_col >= 0).all() and (user_col <= 1).all()
