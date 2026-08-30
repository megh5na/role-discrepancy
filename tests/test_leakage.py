"""
tests/test_leakage.py

*** FAILS if any test span appears in training *** (Section 17 repository
conventions; Section 27 rule 10). This is the single test named explicitly,
by filename, in the spec's own repository structure.

Checks the ACTUAL files written by src/supervision/build_supervision_sets.py
under data/processed/. Skips (does not silently pass) if those files haven't
been built yet in this environment, mirroring test_schema.py's pattern --
but note the skip is loud (reason printed), not a silent pass, and CI/local
runs that have built the data will always exercise the real check.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


def _load_jsonl_field(path: Path, field: str) -> set[str]:
    import json

    ids = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.add(json.loads(line)[field])
    return ids


def test_no_swap_test_span_appears_in_naive_control():
    naive_path = _PROCESSED / "naive_control.jsonl"
    swap_path = _PROCESSED / "swap_test_pairs.jsonl"
    if not (naive_path.exists() and swap_path.exists()):
        pytest.skip("data/processed/ not built yet -- run src.supervision.build_supervision_sets")

    naive_origins = _load_jsonl_field(naive_path, "origin_record_id")
    swap_origins = _load_jsonl_field(swap_path, "origin_record_id")
    overlap = naive_origins & swap_origins
    assert not overlap, f"LEAKAGE: {len(overlap)} swap-test spans found in naive_control training data"


def test_no_swap_test_span_appears_in_transplant_train():
    transplant_path = _PROCESSED / "transplant_train.jsonl"
    swap_path = _PROCESSED / "swap_test_pairs.jsonl"
    if not (transplant_path.exists() and swap_path.exists()):
        pytest.skip("data/processed/ not built yet -- run src.supervision.build_supervision_sets")

    transplant_origins = _load_jsonl_field(transplant_path, "origin_record_id")
    swap_origins = _load_jsonl_field(swap_path, "origin_record_id")
    overlap = transplant_origins & swap_origins
    assert not overlap, f"LEAKAGE: {len(overlap)} swap-test spans found in transplant_train training data"
