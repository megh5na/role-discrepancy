"""
src/ingestion/register_corpora.py

EXISTING datasets, NEW normalisation logic (Section 8, C1: "Existing datasets
/ NEW normalisation logic").
Architecture component: C1 (Data Ingestion) -> Category B, Section 10.

Loads the REGISTER SOURCE corpora (Category B — TRAINING data for the
perceived-role encoder, C4). Each register is sourced from >=1 real, public
corpus whose text naturally "sounds like" that role:

    USER      -> databricks/databricks-dolly-15k ('instruction')
                 + Open-Orca/OpenOrca ('question')
                 (imperative/interrogative, second-person-directed register)
    DOCUMENT  -> wikimedia/wikipedia 20231101.en ('text')
                 + allenai/c4 'en' ('text')
                 (declarative, third-person expository register)
    TOOL      -> glaiveai/glaive-function-calling-v2, "FUNCTION RESPONSE:"
                 segments of 'chat'
                 (structured JSON, machine-to-machine register)
                 *** SINGLE-SOURCED — see LIMITATION note below ***
    REASONING -> openai/gsm8k 'main' ('answer')
                 + microsoft/orca-math-word-problems-200k ('answer')
                 (first-person deliberative, step-by-step register)
    SYSTEM    -> fka/awesome-chatgpt-prompts ('prompt')
                 (persona/identity-defining directive register)
                 *** SINGLE-SOURCED — see LIMITATION note below ***
                 *** NOT IN THE SPEC'S SECTION 10 LIST — see note below ***

NOTE ON SYSTEM REGISTER: Section 10 of the spec lists register-source corpora
for USER/DOCUMENT/TOOL/REASONING only; it names no source for SYSTEM. But
Section 8's own example encoder output is a 5-way distribution including
`system` ("{system: 0.02, user: 0.91, ...}"), and Role (schema.py) is a
5-value enum matching Ye et al.'s vocabulary. This is a genuine gap in the
document, not a design choice to silently resolve either way (Section 27
rule 1: "If code and this document disagree, flag it rather than silently
following either"). Resolution: added a SYSTEM source
(fka/awesome-chatgpt-prompts -- persona/identity-instruction prompts, e.g.
"I want you to act as a linux terminal...", genuinely distinct in register
from USER task requests) so the encoder's output space matches Section 8's
stated 5-class example. Flagged here.

WHY MULTIPLE CORPORA PER REGISTER (balance condition, Section 6 NEW-1 /
Section 10): if every "user"-labelled example comes from Dolly and nothing
else, the model can hit high accuracy by learning Dolly's specific house
style rather than the user register in general — the corpus-identity
shortcut named explicitly in Section 10's cross-cutting risks. Two
independent corpora per register (three of four registers here) makes
corpus identity a much weaker predictor of the label than register is,
which is the whole point of the balance condition in C3.

LIMITATION (documented honestly, not hidden — Section 27 rule 7):
TOOL register is currently single-sourced (glaive-function-calling-v2 only).
Two rehosts of the same underlying conversations were evaluated as candidate
second sources (hypervariance/function-calling-sharegpt,
Locutusque/function-calling-chatml) and rejected: inspection showed identical
underlying examples (same "get_exchange_rate", same "generate_password"
conversations) under different formatting, i.e. NOT an independent corpus --
using it would have been fake diversity, which the spec explicitly warns
against more than it warns against having less data (Section 27 rule 7:
"Never fabricate results"; the same principle applies to fabricating
corpus diversity). Flagged as an open item:
find one genuinely independent real tool-output corpus.

IMPORTANT — what this file does NOT do: it does not assign spans to
"channels" for training. `declared_role` here is the ORIGIN REGISTER — how
the text naturally reads, independent of any channel it might later be
placed in. Channel placement / transplantation is src/supervision/transplant.py's
job (C3), not this file's.

Sample sizes are capped (`n` per source) for tractable CPU/MPS iteration on
a single dev laptop. Interim scale, not
the terminal dataset size.
"""

from __future__ import annotations

import random
import re
from typing import Iterator

from datasets import load_dataset

from src.ingestion.schema import DataCategory, Record, Role

_SEED = 13


def _shuffled_indices(n_total: int, seed: int = _SEED) -> list[int]:
    idxs = list(range(n_total))
    random.Random(seed).shuffle(idxs)
    return idxs


# --- USER register --------------------------------------------------------


def load_dolly(n: int = 1500) -> list[Record]:
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    records = []
    for i in _shuffled_indices(len(ds)):
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


def load_openorca(n: int = 1500) -> list[Record]:
    ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
    records = []
    for i, row in enumerate(ds):
        text = (row.get("question") or "").strip()
        if not text:
            continue
        records.append(
            Record(
                record_id=Record.make_id("openorca", text, i),
                text=text,
                declared_role=Role.USER,
                source_dataset="openorca",
                category=DataCategory.REGISTER_SOURCE,
            )
        )
        if len(records) >= n:
            break
    return records


def load_user_register(n_per_source: int = 1500) -> list[Record]:
    return load_dolly(n_per_source) + load_openorca(n_per_source)


# --- DOCUMENT register -----------------------------------------------------


def load_wikipedia(n: int = 1500) -> list[Record]:
    ds = load_dataset(
        "wikimedia/wikipedia", "20231101.en", split="train", streaming=True
    )
    records = []
    for i, row in enumerate(ds):
        text = row["text"].strip()
        if not text:
            continue
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


def load_c4(n: int = 1500) -> list[Record]:
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    records = []
    for i, row in enumerate(ds):
        text = (row.get("text") or "").strip()
        if not text:
            continue
        excerpt = text[:800]
        records.append(
            Record(
                record_id=Record.make_id("c4-en", excerpt, i),
                text=excerpt,
                declared_role=Role.DOCUMENT,
                source_dataset="c4-en",
                category=DataCategory.REGISTER_SOURCE,
            )
        )
        if len(records) >= n:
            break
    return records


def load_document_register(n_per_source: int = 1500) -> list[Record]:
    return load_wikipedia(n_per_source) + load_c4(n_per_source)


# --- TOOL register (single-sourced -- see module docstring LIMITATION) ----

_FUNC_RESPONSE_RE = re.compile(
    r"FUNCTION RESPONSE:\s*(\{.*?\})\s*(?:\n\n\nASSISTANT|\Z)", re.DOTALL
)


def _extract_function_responses(chat_text: str) -> Iterator[str]:
    for m in _FUNC_RESPONSE_RE.finditer(chat_text):
        blob = m.group(1).strip()
        if blob:
            yield blob


def load_glaive_tool(n: int = 6000) -> list[Record]:
    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    records = []
    for i in _shuffled_indices(len(ds)):
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


def load_tool_register(n_per_source: int = 6000) -> list[Record]:
    return load_glaive_tool(n_per_source)


# --- REASONING register ----------------------------------------------------

_CALC_ANNOTATION_RE = re.compile(r"<<[^>]*>>")


def load_gsm8k(n: int = 1500) -> list[Record]:
    ds = load_dataset("openai/gsm8k", "main", split="train")
    records = []
    for i in _shuffled_indices(len(ds)):
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


def load_orca_math(n: int = 1500) -> list[Record]:
    ds = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
    records = []
    for i in _shuffled_indices(len(ds)):
        text = (ds[i].get("answer") or "").strip()
        if not text:
            continue
        records.append(
            Record(
                record_id=Record.make_id("orca-math", text, i),
                text=text,
                declared_role=Role.REASONING,
                source_dataset="orca-math",
                category=DataCategory.REGISTER_SOURCE,
            )
        )
        if len(records) >= n:
            break
    return records


def load_reasoning_register(n_per_source: int = 1500) -> list[Record]:
    return load_gsm8k(n_per_source) + load_orca_math(n_per_source)


# --- SYSTEM register (single-sourced -- see module docstring note) --------


def load_system_prompts(n: int = 2100) -> list[Record]:
    ds = load_dataset("fka/awesome-chatgpt-prompts", split="train")
    records = []
    for i in _shuffled_indices(len(ds)):
        text = (ds[i].get("prompt") or "").strip()
        if not text:
            continue
        records.append(
            Record(
                record_id=Record.make_id("awesome-chatgpt-prompts", text, i),
                text=text,
                declared_role=Role.SYSTEM,
                source_dataset="awesome-chatgpt-prompts",
                category=DataCategory.REGISTER_SOURCE,
            )
        )
        if len(records) >= n:
            break
    return records


def load_system_register(n_per_source: int = 2100) -> list[Record]:
    return load_system_prompts(n_per_source)


def load_all_register_sources(
    n_user: int = 1500,
    n_document: int = 1500,
    n_tool: int = 6000,
    n_reasoning: int = 1500,
    n_system: int = 2100,
) -> list[Record]:
    records: list[Record] = []
    records += load_user_register(n_user)
    records += load_document_register(n_document)
    records += load_tool_register(n_tool)
    records += load_reasoning_register(n_reasoning)
    records += load_system_register(n_system)
    return records


if __name__ == "__main__":
    from collections import Counter

    from src.ingestion.io_utils import write_records

    recs = load_all_register_sources()
    out_path = "data/interim/register_sources.jsonl"
    n = write_records(recs, out_path)
    print(f"Wrote {n} register-source records to {out_path}")

    counts = Counter((r.declared_role.value, r.source_dataset) for r in recs)
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
