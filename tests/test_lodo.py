"""tests/test_lodo.py -- C10 LODO harness."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.schema import DataCategory, Record, Role  # noqa: E402
from src.evaluation.lodo import lodo_fold_names, lodo_splits  # noqa: E402


def _make_records() -> list[Record]:
    records = []
    for ds_name in ["dsA", "dsB", "dsC"]:
        for i in range(5):
            records.append(
                Record(
                    record_id=f"{ds_name}-{i}",
                    text=f"text {i}",
                    declared_role=Role.USER,
                    source_dataset=ds_name,
                    category=DataCategory.REGISTER_SOURCE,
                )
            )
    return records


def test_lodo_produces_one_fold_per_dataset():
    records = _make_records()
    folds = list(lodo_splits(records))
    assert {f[0] for f in folds} == {"dsA", "dsB", "dsC"}
    assert len(folds) == 3


def test_lodo_test_fold_disjoint_from_train_fold():
    records = _make_records()
    for held_out, train, test in lodo_splits(records):
        train_ids = {r.record_id for r in train}
        test_ids = {r.record_id for r in test}
        assert train_ids.isdisjoint(test_ids)
        assert all(r.source_dataset == held_out for r in test)
        assert all(r.source_dataset != held_out for r in train)


def test_lodo_covers_all_records_across_folds():
    records = _make_records()
    for held_out, train, test in lodo_splits(records):
        assert len(train) + len(test) == len(records)


def test_lodo_fold_names_sorted_and_unique():
    records = _make_records()
    assert lodo_fold_names(records) == ["dsA", "dsB", "dsC"]
