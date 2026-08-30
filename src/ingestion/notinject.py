"""
src/ingestion/notinject.py

EXISTING dataset (leolee99/NotInject, ACL 2025 InjecGuard/PIGuard), NEW
normalisation. Category A, Section 10: "339 benign samples containing
injection trigger words. Purpose: Over-defense probe. Split: Test only."

The HF hosting splits this into three sub-collections (NotInject_one/two/
three, by trigger-word count) totalling 339 -- matches the spec's stated
size exactly, confirming this is the right dataset.

Every record here is BENIGN (is_injected=False) by construction -- that's
the point: these are ordinary user questions that happen to contain words
like "ignore," used to measure whether a detector over-fires on lexical
triggers alone (Section 4.5's over-defense finding: guard models fall to
~60% accuracy on this kind of text).
"""

from __future__ import annotations

from datasets import load_dataset

from src.ingestion.schema import DataCategory, Record, Role

_SPLITS = ["NotInject_one", "NotInject_two", "NotInject_three"]


def load_notinject() -> list[Record]:
    records = []
    for split in _SPLITS:
        ds = load_dataset("leolee99/NotInject", split=split)
        for i, row in enumerate(ds):
            text = (row.get("prompt") or "").strip()
            if not text:
                continue
            records.append(
                Record(
                    record_id=Record.make_id(f"notinject-{split}", text, i),
                    text=text,
                    declared_role=Role.USER,
                    source_dataset="notinject",
                    category=DataCategory.INJECTION,
                    is_injected=False,
                    meta={"category": row.get("category"), "word_list": row.get("word_list"), "split": split},
                )
            )
    return records


if __name__ == "__main__":
    from src.ingestion.io_utils import write_records

    recs = load_notinject()
    n = write_records(recs, "data/interim/notinject.jsonl")
    print(f"Wrote {n} NotInject records to data/interim/notinject.jsonl")
