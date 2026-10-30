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
