"""
src/preprocessing/build_clean_spans.py

NEW pipeline glue (Section 8, C2) wiring together strip -> segment -> dedup.
Runs the full C2 stage: raw/interim records -> clean, span-level, delimiter-
free, deduplicated records ready for C3 (supervision construction).

Usage:
    python -m src.preprocessing.build_clean_spans \
        --in data/interim/register_sources.jsonl \
        --out data/interim/clean_spans_register.jsonl
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace

from src.ingestion.io_utils import read_records, write_records
from src.preprocessing.dedup import dedup_records
from src.preprocessing.delimiters import contains_delimiter_artifact, strip_delimiters
from src.preprocessing.segmentation import segment_records


def clean_and_segment(records: list, granularity: str = "sentence") -> list:
    # 1. Strip delimiters BEFORE segmentation, so pysbd never has to reason
    #    about chat-template tokens as if they were sentence punctuation.
    stripped = []
    for r in records:
        clean_text = strip_delimiters(r.text)
        if not clean_text:
            continue
        stripped.append(replace(r, text=clean_text))

    # 2. Segment into spans (sentence- or message-level per config).
    spans = segment_records(stripped, granularity=granularity)

    # 3. Hard safety check: re-verify after segmentation. Segmentation
    #    shouldn't introduce artifacts, but this is the security-critical
    #    boundary (Section 8, C2) -- verify, don't assume.
    for s in spans:
        if contains_delimiter_artifact(s.text):
            raise RuntimeError(
                f"Delimiter artifact survived cleaning+segmentation: {s.text!r}"
            )

    # 4. Dedup WITHIN each declared_role group (see dedup.py docstring for
    #    why scoping matters).
    by_role: dict[str, list] = {}
    for s in spans:
        by_role.setdefault(s.declared_role.value, []).append(s)

    deduped = []
    for role, group in by_role.items():
        deduped.extend(dedup_records(group))

    return deduped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--granularity", default="sentence", choices=["sentence", "message"])
    args = ap.parse_args()

    records = list(read_records(args.in_path))
    print(f"Loaded {len(records)} raw records from {args.in_path}")

    clean = clean_and_segment(records, granularity=args.granularity)
    print(f"After strip+segment+dedup: {len(clean)} clean spans")

    n = write_records(clean, args.out_path)
    print(f"Wrote {n} clean spans to {args.out_path}")

    counts = Counter((r.declared_role.value, r.source_dataset) for r in clean)
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
