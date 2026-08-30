"""
src/ingestion/llmail.py

EXISTING dataset (microsoft/llmail-inject-challenge), NEW normalisation.
Category A, Section 10: "LLMail-Inject... ~9,998 email-body injections
across difficulty levels... 100% malicious; single delivery vector. Split:
Test only; one LODO fold."

The hosted version is much larger than the spec's stated ~9,998 (370,724
rows in Phase1 alone) -- this is the raw competition-submission log (every
red-team attempt against Microsoft's LLMail-Inject challenge, not a curated
benchmark subset), so we SAMPLE down to a manageable size rather than using
it all. `scenario` (e.g. "level1a") is the difficulty/attack-family field,
used the same way `attack_category` is for BIPIA (Experiment 4, held-out
family splits).

HONEST NOTE ON DATA QUALITY: many submission bodies are unnatural
adversarial text (e.g. repeated filler tokens like "yes" interleaved with
the payload -- real red-teamers probing model robustness, not naturalistic
prose). This is genuine attack diversity, not a cleaning bug -- kept as-is
rather than filtered to "nicer-looking" text, since filtering would bias
toward the injection styles we already expect to catch.
"""

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
