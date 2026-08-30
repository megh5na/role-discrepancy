"""
src/ingestion/register_corpora.py

EXISTING datasets, NEW normalisation logic (Section 8, C1: "Existing datasets
/ NEW normalisation logic").
Architecture component: C1 (Data Ingestion) -> Category B, Section 10.

Loads the four REGISTER SOURCE corpora (Category B — TRAINING data for the
perceived-role encoder, C4). These are the corpora whose text naturally
"sounds like" one of the four non-system roles:

    USER      -> databricks/databricks-dolly-15k, field 'instruction'
                 (imperative/interrogative, second-person-directed register)
    DOCUMENT  -> wikimedia/wikipedia (20231101.en), field 'text'
                 (declarative, third-person expository register)
    TOOL      -> glaiveai/glaive-function-calling-v2, the "FUNCTION RESPONSE:"
                 segments of the 'chat' field
                 (structured JSON, machine-to-machine register)
    REASONING -> openai/gsm8k (main), field 'answer'
                 (first-person deliberative, step-by-step register)

WHY THESE SOURCES, SPECIFICALLY: each is chosen because its natural register
is unambiguous and well documented as belonging to that communicative
situation, and because none of them require us to fabricate text (Section 10:
"Prefer human-sourced; LLM-generated items must be marked" — none of these
four are LLM-generated).

IMPORTANT — what this file does NOT do: it does not assign spans to
"channels" for training. `declared_role` here is the ORIGIN REGISTER — a
statement about how the text naturally reads, independent of any channel
it might later be placed in. Channel placement / transplantation is
src/supervision/transplant.py's job (C3), not this file's. Conflating the
two here would defeat the whole point of the supervision construction.

Sample sizes are capped (`n` per source) for tractable CPU/MPS iteration on
a single dev laptop — see docs/decisions.md. This is an interim-scale
pipeline for Review 3, not the terminal dataset size.
"""

from __future__ import annotations

import random
import re
from typing import Iterator

from datasets import load_dataset

from src.ingestion.schema import DataCategory, Record, Role

_SEED = 13


def load_user_register(n: int = 1500) -> list[Record]:
    """Dolly-15k instructions -> USER register."""
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    idxs = list(range(len(ds)))
    random.Random(_SEED).shuffle(idxs)
    records = []
    for i in idxs:
        text = ds[i]["instruction"].strip()
        if not text:
            continue
        records.append(
            Record(
                record_id=Record.make_id("dolly-15k", text, i),
                text=text,
                declared_role=Role.USER,
                source_dataset="dolly-15k",
                category=DataCategory.REGISTER_SOURCE,
            )
        )
        if len(records) >= n:
            break
    return records


def load_document_register(n: int = 1500) -> list[Record]:
    """Wikipedia article openings -> DOCUMENT register.

    Streamed (not fully downloaded): wikimedia/wikipedia is tens of GB;
    we only need the first sentence or two of `n` articles.
    """
    ds = load_dataset(
        "wikimedia/wikipedia", "20231101.en", split="train", streaming=True
    )
    records = []
    for i, row in enumerate(ds):
        text = row["text"].strip()
        if not text:
            continue
        # Keep a document-length excerpt; C2 will sentence-segment it.
        excerpt = text[:800]
        records.append(
            Record(
                record_id=Record.make_id("wikipedia-en", excerpt, i),
                text=excerpt,
                declared_role=Role.DOCUMENT,
                source_dataset="wikipedia-en",
                category=DataCategory.REGISTER_SOURCE,
                meta={"title": row.get("title", "")},
            )
        )
        if len(records) >= n:
            break
    return records


_FUNC_RESPONSE_RE = re.compile(
    r"FUNCTION RESPONSE:\s*(\{.*?\})\s*(?:\n\n\nASSISTANT|\Z)", re.DOTALL
)


def _extract_function_responses(chat_text: str) -> Iterator[str]:
    for m in _FUNC_RESPONSE_RE.finditer(chat_text):
        blob = m.group(1).strip()
        if blob:
            yield blob


def load_tool_register(n: int = 1500) -> list[Record]:
    """glaive-function-calling-v2 FUNCTION RESPONSE blobs -> TOOL register."""
    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    idxs = list(range(len(ds)))
    random.Random(_SEED).shuffle(idxs)
    records = []
    for i in idxs:
        chat = ds[i].get("chat") or ""
        for blob in _extract_function_responses(chat):
            records.append(
                Record(
                    record_id=Record.make_id(
                        "glaive-function-calling-v2", blob, len(records)
                    ),
                    text=blob,
                    declared_role=Role.TOOL,
                    source_dataset="glaive-function-calling-v2",
                    category=DataCategory.REGISTER_SOURCE,
                )
            )
            if len(records) >= n:
                return records
    return records


_CALC_ANNOTATION_RE = re.compile(r"<<[^>]*>>")


def load_reasoning_register(n: int = 1500) -> list[Record]:
    """gsm8k step-by-step answers -> REASONING register.

    Strips the '<<48/2=24>>' calculator annotations (a GSM8K-specific
    artifact, not part of natural reasoning register) and the trailing
    '#### <final answer>' marker (a dataset delimiter, same class of thing
    C2's delimiter stripping removes generally -- stripped here at the
    source since it's dataset-specific rather than a general chat-template
    artifact).
    """
    ds = load_dataset("openai/gsm8k", "main", split="train")
    idxs = list(range(len(ds)))
    random.Random(_SEED).shuffle(idxs)
    records = []
    for i in idxs:
        raw = ds[i]["answer"]
        text = _CALC_ANNOTATION_RE.sub("", raw)
        text = text.split("####")[0].strip()
        if not text:
            continue
        records.append(
            Record(
                record_id=Record.make_id("gsm8k-reasoning", text, i),
                text=text,
                declared_role=Role.REASONING,
                source_dataset="gsm8k-reasoning",
                category=DataCategory.REGISTER_SOURCE,
            )
        )
        if len(records) >= n:
            break
    return records


def load_all_register_sources(n_per_source: int = 1500) -> list[Record]:
    records: list[Record] = []
    records += load_user_register(n_per_source)
    records += load_document_register(n_per_source)
    records += load_tool_register(n_per_source)
    records += load_reasoning_register(n_per_source)
    return records


if __name__ == "__main__":
    from src.ingestion.io_utils import write_records

    recs = load_all_register_sources(n_per_source=1500)
    out_path = "data/interim/register_sources.jsonl"
    n = write_records(recs, out_path)
    print(f"Wrote {n} register-source records to {out_path}")
    from collections import Counter

    counts = Counter((r.declared_role.value, r.source_dataset) for r in recs)
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
