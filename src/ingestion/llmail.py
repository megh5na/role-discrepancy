from __future__ import annotations

from datasets import load_dataset

from src.ingestion.schema import DataCategory, Record, Role


def load_llmail(n: int = 3000, seed: int = 21) -> list[Record]:
    ds = load_dataset("microsoft/llmail-inject-challenge", split="Phase1")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    records = []
    for i, row in enumerate(ds):
        body = (row.get("body") or "").strip()
        if not body:
            continue
        records.append(
            Record(
                record_id=Record.make_id("llmail-inject", body, i),
                text=body,
                declared_role=Role.DOCUMENT,  # email body = retrieved/read content
                source_dataset="llmail-inject",
                category=DataCategory.INJECTION,
                is_injected=True,
                injection_type=row.get("scenario"),
                meta={"subject": row.get("subject"), "objectives": row.get("objectives")},
            )
        )
    return records


if __name__ == "__main__":
    from collections import Counter

    from src.ingestion.io_utils import write_records

    recs = load_llmail(n=3000)
    n = write_records(recs, "data/interim/llmail.jsonl")
    print(f"Wrote {n} LLMail-Inject records to data/interim/llmail.jsonl")
    print("scenario distribution:", Counter(r.injection_type for r in recs))
