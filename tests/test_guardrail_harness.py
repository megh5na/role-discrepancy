"""tests/test_guardrail_harness.py -- offline tests for the label-resolution
logic in src/baselines/guardrails/harness.py (no model download needed)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines.guardrails.harness import (  # noqa: E402
    KNOWN_GUARDRAILS,
    resolve_injection_index,
)


def test_known_guardrails_registry_well_formed():
    assert len(KNOWN_GUARDRAILS) >= 2
    for key, spec in KNOWN_GUARDRAILS.items():
        assert spec.hf_model_id
        assert spec.name


def test_resolve_by_explicit_hint():
    id2label = {0: "SAFE", 1: "INJECTION"}
    assert resolve_injection_index(id2label, "INJECTION") == 1


def test_resolve_by_positive_marker_keyword():
    id2label = {0: "BENIGN", 1: "MALICIOUS"}
    assert resolve_injection_index(id2label, None) == 1

    id2label2 = {0: "LEGIT", 1: "UNSAFE"}
    assert resolve_injection_index(id2label2, None) == 1


def test_resolve_falls_back_to_index_one_for_plain_binary():
    id2label = {0: "LABEL_0", 1: "LABEL_1"}
    assert resolve_injection_index(id2label, None) == 1


def test_resolve_raises_on_missing_hint():
    id2label = {0: "SAFE", 1: "INJECTION"}
    try:
        resolve_injection_index(id2label, "NOT_A_REAL_LABEL")
        assert False, "expected ValueError"
    except ValueError:
        pass
