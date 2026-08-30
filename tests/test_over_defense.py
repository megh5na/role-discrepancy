"""tests/test_over_defense.py -- C10/E6 over-defense measurement."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.schema import Role  # noqa: E402
from src.evaluation.over_defense import fpr_by_channel_pair, overall_fpr, summarize_over_defense  # noqa: E402


def test_overall_fpr_basic():
    assert overall_fpr([True, True, False, False]) == 0.5
    assert overall_fpr([False, False, False]) == 0.0
    assert overall_fpr([True, True]) == 1.0


def test_fpr_by_channel_pair_separates_pairs():
    perceived = [Role.DOCUMENT, Role.DOCUMENT, Role.USER, Role.USER]
    declared = [Role.USER, Role.USER, Role.TOOL, Role.TOOL]
    flags = [False, True, True, True]
    result = fpr_by_channel_pair(perceived, declared, flags)
    assert result["document->user"]["fpr"] == 0.5
    assert result["document->user"]["n"] == 2
    assert result["user->tool"]["fpr"] == 1.0
    assert result["user->tool"]["n"] == 2


def test_summarize_sorts_worst_first():
    perceived = [Role.DOCUMENT, Role.USER, Role.USER]
    declared = [Role.USER, Role.TOOL, Role.TOOL]
    flags = [False, True, True]
    summary = summarize_over_defense(perceived, declared, flags)
    keys = list(summary["by_channel_pair"].keys())
    assert keys[0] == "user->tool"  # fpr=1.0, listed before document->user's fpr=0.0
    assert summary["n_total"] == 3
