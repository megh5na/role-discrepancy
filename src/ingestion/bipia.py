"""
src/ingestion/bipia.py

EXISTING dataset, NEW normalisation logic (Section 8 C1). Category A
(injection corpora) — Section 10: "TEST ONLY, NEVER TRAINING."

Source: geodesic-research/bipia on HuggingFace (41,250 rows), a hosted copy
of the BIPIA benchmark (Section 4.1/10 of the spec: task contexts -- email,
code, table -- with configurable injection position). Verified the
`attack_str` field is always a literal substring of `context` (spot-checked
in the build session).

Produces TWO kinds of Record from each row:
  - the injected span itself (`attack_str`), is_injected=True -- for
    measuring TPR (Experiment 3/4)
  - a benign span from the SAME context with the attack removed, split into
    sentences, is_injected=False -- for measuring FPR on the same
    distribution the attacks were embedded in (needed for the matched-FPR
    comparison, Section 4/14/15: "MANDATORY... comparing raw accuracies...
    is meaningless")

CHANNEL MAPPING (a judgement call, stated explicitly per Section 8 C1's own
warning: "Channel annotation is judgement-heavy; corpora disagree about
structure more than the papers suggest"): BIPIA's three task types (email,
table, code) are all instances of the SAME structural situation -- an
assistant reading externally-retrieved content it did not generate. All
three are mapped to declared_role=DOCUMENT, not split across DOCUMENT/TOOL,
because BIPIA's own framing (Section 4.1, Greshake et al.'s threat model)
treats them uniformly as "retrieved content," and table/code content here
is prose-embedded (SUBJECT:/EMAIL_FROM:/CONTENT: style, or a QA passage),
not raw structured API/JSON output the way our TOOL register actually reads.
Recorded here so a reviewer can see and challenge the call.

attack_category (4 coarse families: task_irrelevant, task_relevant,
targeted, code) is stored as `injection_type` -- this is what
held-out-attack-family splitting (Experiment 4) rotates over.
"""

from __future__ import annotations

import pysbd
from datasets import load_dataset

from src.ingestion.schema import DataCategory, Record, Role

_segmenter = pysbd.Segmenter(language="en", clean=False)


def load_bipia(n: int | None = None, seed: int = 21) -> list[Record]:
    ds = load_dataset("geodesic-research/bipia", split="train")
    if n is not None:
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    records: list[Record] = []
    for i, row in enumerate(ds):
        context = row["context"]
        attack = row["attack_str"]

        # 1. The injected span itself.
        if attack and attack in context:
            records.append(
                Record(
                    record_id=Record.make_id("bipia-attack", attack, i),
                    text=attack,
                    declared_role=Role.DOCUMENT,
                    source_dataset="bipia",
                    category=DataCategory.INJECTION,
                    is_injected=True,
                    injection_type=row.get("attack_category"),
                    meta={
                        "attack_name": row.get("attack_name"),
                        "task_name": row.get("task_name"),
                        "position": row.get("position"),
                    },
                )
            )
            # 2. Benign remainder of the same context, sentence-segmented,
            #    for FPR measurement on the same underlying distribution.
            benign_remainder = context.replace(attack, " ")
            for j, sent in enumerate(_segmenter.segment(benign_remainder)):
                sent = sent.strip()
                if len(sent.split()) < 4:
                    continue
                records.append(
                    Record(
                        record_id=Record.make_id("bipia-benign", sent, i * 100 + j),
                        text=sent,
                        declared_role=Role.DOCUMENT,
                        source_dataset="bipia",
                        category=DataCategory.INJECTION,
                        is_injected=False,
                        meta={"task_name": row.get("task_name")},
                    )
                )
    return records


if __name__ == "__main__":
    from collections import Counter

    from src.ingestion.io_utils import write_records

    recs = load_bipia(n=1500)
    n = write_records(recs, "data/interim/bipia.jsonl")
    print(f"Wrote {n} BIPIA records to data/interim/bipia.jsonl")
    counts = Counter((r.is_injected, r.injection_type) for r in recs)
    for k, v in sorted(counts.items(), key=lambda kv: str(kv[0])):
        print(f"  is_injected={k[0]} injection_type={k[1]}: {v}")
