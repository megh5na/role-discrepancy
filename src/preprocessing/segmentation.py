"""
src/preprocessing/segmentation.py

NEW (standard technique -- pysbd sentence boundary detection -- applied to
project-specific requirements). Section 8, C2.

Splits cleaned record text into spans for the encoder to see. Granularity is
a config switch, not two code paths: sentence-level is
the current default (localises injections better, per Section 10's own
tradeoff note), message-level (the whole record as one span) is available
for the granularity ablation (Section 14) without new code.

Also applies length filtering: spans that are too short carry little
register signal (a 2-token span could be almost anything); spans that are
too long dilute it and stop being a "span" in the sense the discrepancy
scorer expects. Bounds follow the spec's stated typical range (Section 11:
"10-200 tokens typical") using whitespace-token count as a cheap proxy for
model tokens (avoids importing a tokenizer this early in the pipeline).
"""

from __future__ import annotations

from dataclasses import replace

import pysbd

from src.ingestion.schema import Record

_MIN_TOKENS = 4
_MAX_TOKENS = 200

_segmenter_cache: dict[str, pysbd.Segmenter] = {}


def _segmenter() -> pysbd.Segmenter:
    if "en" not in _segmenter_cache:
        _segmenter_cache["en"] = pysbd.Segmenter(language="en", clean=False)
    return _segmenter_cache["en"]


def _n_tokens(text: str) -> int:
    return len(text.split())


def segment_record(record: Record, granularity: str = "sentence") -> list[Record]:
    """Split one Record's `text` into one or more span-level Records.

    granularity: "sentence" (default) or "message" (no splitting; the whole
    cleaned record is a single span, subject to the same length filter).

    New Records reuse the parent's declared_role/source_dataset/category/
    is_injected/injection_type but get fresh record_ids (span identity, not
    parent identity) and carry `meta["parent_record_id"]` for traceability.
    """
    text = record.text
    if granularity == "message":
        pieces = [text]
    elif granularity == "sentence":
        pieces = _segmenter().segment(text)
    else:
        raise ValueError(f"Unknown granularity: {granularity!r}")

    out: list[Record] = []
    for i, piece in enumerate(pieces):
        piece = piece.strip()
        n_tok = _n_tokens(piece)
        if n_tok < _MIN_TOKENS or n_tok > _MAX_TOKENS:
            continue
        new_id = Record.make_id(
            f"{record.source_dataset}-span", piece, i
        ) + f"-{record.record_id[-8:]}"
        span = replace(
            record,
            record_id=new_id,
            text=piece,
            meta={**record.meta, "parent_record_id": record.record_id},
        )
        out.append(span)
    return out


def segment_records(
    records: list[Record], granularity: str = "sentence"
) -> list[Record]:
    out: list[Record] = []
    for r in records:
        out.extend(segment_record(r, granularity=granularity))
    return out
