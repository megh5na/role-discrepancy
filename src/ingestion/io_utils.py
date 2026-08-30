"""
src/ingestion/io_utils.py

EXISTING — plain jsonl read/write helpers. No research content.
Used by every loader in src/ingestion/ to persist Records to data/interim/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from src.ingestion.schema import Record


def write_records(records: Iterable[Record], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_records(path: str | Path) -> Iterator[Record]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield Record.from_dict(json.loads(line))
