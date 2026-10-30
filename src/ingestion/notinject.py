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
