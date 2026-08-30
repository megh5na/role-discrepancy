"""
src/ingestion/schema.py

EXISTING (standard dataclass/enum patterns) — no research novelty here.
Architecture component: C1, Data Ingestion (Section 8/17).

Defines the single unified record format every raw corpus gets normalised
into. Every downstream stage (preprocessing, supervision construction,
training, evaluation) assumes this schema and only this schema.

Why this file exists (from the spec, Section 8, C1):
Four injection corpora, ~6 register-source corpora, and a benign role-mixed
corpus we construct ourselves all disagree about structure — different
column names, different nesting, different ideas of what a "turn" is.
Normalising them once here means every later stage (delimiter stripping,
transplantation, LODO splitting, the leakage test) is written once instead
of once per source.

Read this file first (per Section 17 repository conventions).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Role(str, Enum):
    """
    The channel / role vocabulary.

    RESOLVED [TO BE DECIDED, spec Section 0]: 5-role vocabulary, matching
    Ye et al. 2026 (arXiv 2603.12277) exactly.
    Keeping our label space identical to theirs is what makes the sidecar
    reproduction (C7) and the oracle-gap comparison (Experiment 5) a fair,
    like-for-like measurement instead of one that first needs a label
    remapping (which would itself be a source of error).
    """

    SYSTEM = "system"
    USER = "user"
    DOCUMENT = "document"
    TOOL = "tool"
    REASONING = "reasoning"


class DataCategory(str, Enum):
    """
    The four data categories, Section 10. DO NOT CONFLATE THEM — they play
    structurally different roles in the pipeline, and mixing them (most
    dangerously: training on category A) silently destroys the
    attack-agnostic claim that is this project's central point
    (Section 27, Absolute Prohibition B).
    """

    INJECTION = "A_injection_eval_only"        # BIPIA, InjecAgent, LLMail, NotInject — TEST ONLY, NEVER TRAIN
    REGISTER_SOURCE = "B_register_source"      # Dolly, OpenOrca, C4/wiki/Enron, tool/JSON, reasoning traces — TRAIN
    BENIGN_ROLE_MIXED = "C_benign_role_mixed"  # constructed by us — calibration + over-defense eval (E6)
    SIDECAR_PROBE = "D_sidecar_probe_data"     # C4/Dolma neutral text for Ye et al.'s wrapper construction — sidecar (C7) only


@dataclass
class Record:
    """
    One unified record.

    Fields
    ------
    record_id
        Deterministic id — hash of (source_dataset, index, text) — so
        re-running ingestion doesn't change record identity, which matters
        for the leakage test (tests/test_leakage.py) comparing ids across
        splits.
    text
        Raw span text, exactly as extracted. May still contain delimiter or
        chat-template artifacts — stripping those is C2's job, not C1's.
    declared_role
        For category B (register source): the ORIGIN REGISTER of the text —
        the channel this text naturally sounds like, because of which corpus
        it's from (Dolly -> USER, C4/openwebtext -> DOCUMENT, ...). This is
        the ground-truth label C3 (supervision construction) preserves under
        transplantation, regardless of which channel a transplanted copy is
        later embedded in.
        For category A (injection, eval only): the channel the application
        actually declared for the span in the attacked request.
        For category C (benign role-mixed): the channel the text is embedded
        in, in our constructed scenario.
    source_dataset
        Name of the originating corpus, e.g. "dolly-15k", "openwebtext-10k".
        THIS is the unit LODO splits are made on — never split by record_id
        or randomly for the honest evaluation protocol (Section 10; Section
        27 rule 10).
    category
        One of the DataCategory values above.
    is_injected
        Only meaningful for category A/C. True if this span is (part of) a
        planted injection payload. None elsewhere.
    injection_type
        Free-text attack-family label, when known (category A only) — used
        for the held-out-attack-family split (E4).
    meta
        Free-form dict for source-specific extras worth keeping.
    """

    record_id: str
    text: str
    declared_role: Role
    source_dataset: str
    category: DataCategory
    is_injected: Optional[bool] = None
    injection_type: Optional[str] = None
    meta: dict = field(default_factory=dict)

    @staticmethod
    def make_id(source_dataset: str, text: str, index: int) -> str:
        """Deterministic id: identical input always hashes to the same id."""
        h = hashlib.sha256(
            f"{source_dataset}::{index}::{text}".encode("utf-8")
        ).hexdigest()
        return f"{source_dataset}-{h[:16]}"

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "text": self.text,
            "declared_role": self.declared_role.value,
            "source_dataset": self.source_dataset,
            "category": self.category.value,
            "is_injected": self.is_injected,
            "injection_type": self.injection_type,
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: dict) -> "Record":
        return Record(
            record_id=d["record_id"],
            text=d["text"],
            declared_role=Role(d["declared_role"]),
            source_dataset=d["source_dataset"],
            category=DataCategory(d["category"]),
            is_injected=d.get("is_injected"),
            injection_type=d.get("injection_type"),
            meta=d.get("meta") or {},
        )
