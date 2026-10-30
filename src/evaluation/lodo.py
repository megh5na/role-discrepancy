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
