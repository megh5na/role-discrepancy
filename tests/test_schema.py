"""tests/test_schema.py -- Record schema round-trip and basic validity."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.schema import DataCategory, Record, Role  # noqa: E402


def test_record_roundtrip():
    r = Record(
        record_id=Record.make_id("dummy", "hello world", 0),
        text="hello world",
        declared_role=Role.USER,
        source_dataset="dummy",
        category=DataCategory.REGISTER_SOURCE,
    )
    d = r.to_dict()
    r2 = Record.from_dict(d)
    assert r2.text == r.text
    assert r2.declared_role == Role.USER
    assert r2.category == DataCategory.REGISTER_SOURCE
    assert r2.record_id == r.record_id


def test_make_id_deterministic():
    id1 = Record.make_id("src", "same text", 5)
    id2 = Record.make_id("src", "same text", 5)
    id3 = Record.make_id("src", "different text", 5)
    assert id1 == id2
    assert id1 != id3


def test_role_vocab_is_five_roles():
    assert {r.value for r in Role} == {
        "system",
        "user",
        "document",
        "tool",
        "reasoning",
    }


def test_clean_spans_file_has_valid_schema():
    """Integration-ish check: every record in the built clean-spans file
    round-trips and has a non-empty text field. Skips if the file hasn't
    been built yet (keeps this test runnable before the data pipeline)."""
    path = Path(__file__).resolve().parents[1] / "data/interim/clean_spans_register.jsonl"
    if not path.exists():
        import pytest

        pytest.skip("clean_spans_register.jsonl not built yet")

    from src.ingestion.io_utils import read_records

    n = 0
    for r in read_records(path):
        assert r.text.strip() != ""
        assert isinstance(r.declared_role, Role)
        assert r.category == DataCategory.REGISTER_SOURCE
        n += 1
    assert n > 0
