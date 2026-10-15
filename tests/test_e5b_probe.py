"""tests/test_e5b_probe.py -- E5b driver helpers (no model)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp05_oracle_gap.run_e5b_probe import BAND, GUARD, auroc, classify, split  # noqa: E402


def test_interpretation_bands_match_the_preregistration():
    assert BAND == 0.05 and GUARD == 0.60
    assert classify(0.05).startswith("oracle ahead") and classify(0.20).startswith("oracle ahead")
    assert classify(0.049).startswith("no oracle advantage") and classify(-0.049).startswith("no oracle advantage") and classify(0.0).startswith("no oracle advantage")
    assert classify(-0.05) == "encoder ahead" and classify(-0.3) == "encoder ahead"


def test_split_recipe_is_the_encoders_recipe():
    ex = list(range(6317))
    tr, va = split(ex)
    assert (len(tr), len(va)) == (5686, 631)                      # same 5,686 / 631 as both encoder checkpoints
    assert sorted(tr + va) == ex and split(ex) == (tr, va)         # a partition, and deterministic


def test_auroc_direction_and_chance():
    assert auroc(np.array([0.9, 0.8]), np.array([0.1, 0.2])) == 1.0
    assert auroc(np.array([0.1, 0.2]), np.array([0.9, 0.8])) == 0.0
    assert auroc(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == 0.5
