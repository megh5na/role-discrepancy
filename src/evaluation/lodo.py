"""
src/evaluation/lodo.py

NEW harness / EXISTING protocol (Fomin, arXiv 2602.14161 — Section 4.5,
6/E6, 10). Architecture component: C10.

Leave-one-dataset-out cross-validation: rotate each SOURCE DATASET out as
the held-out test fold, train on the rest. This is "the honest evaluation
protocol" (Section 10) specifically because random splits let a detector
partially succeed by recognising which dataset a sample came from rather
than recognising the actual phenomenon (Section 4.5: a classifier predicting
WHICH of 18 datasets a sample came from reaches 96.6% accuracy — sources are
trivially separable). Grouping by `source_dataset`, never by record_id or a
random split, is what prevents that shortcut from leaking into our own
reported numbers.

Generic over any Record list (works for register-source role-classification
LODO -- Experiment 2 -- and for injection-corpus detection LODO -- Experiment
4 -- with the same function).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterator

from src.ingestion.schema import Record


def lodo_splits(records: list[Record]) -> Iterator[tuple[str, list[Record], list[Record]]]:
    """Yields (held_out_dataset_name, train_records, test_records) once per
    distinct source_dataset present in `records`."""
    by_dataset: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        by_dataset[r.source_dataset].append(r)

    all_datasets = list(by_dataset.keys())
    for held_out in all_datasets:
        test_records = by_dataset[held_out]
        train_records = [r for name in all_datasets if name != held_out for r in by_dataset[name]]
        yield held_out, train_records, test_records


def lodo_fold_names(records: list[Record]) -> list[str]:
    return sorted({r.source_dataset for r in records})
