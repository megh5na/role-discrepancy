"""
src/supervision/types.py

*** OUR CONTRIBUTION (NEW) *** — shared data types for C3.

TrainingExample is what the encoder (C4) actually trains on: `text` is the
only thing the model ever sees; every other field is bookkeeping used to
build/balance/audit the construction and is NEVER passed to the model
(Section 27, Absolute Prohibition A).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from src.ingestion.schema import Role


@dataclass
class TrainingExample:
    example_id: str
    text: str
    label: Role                 # training target = ORIGIN register
    position_channel: Role      # which carrier (or native/bare) the text is dressed as -- bookkeeping ONLY, never model input
    is_transplanted: bool       # position_channel != label's natural origin
    source_dataset: str
    origin_record_id: str
    meta: dict = field(default_factory=dict)

    @staticmethod
    def make_id(origin_record_id: str, position_channel: Role, salt: str = "") -> str:
        h = hashlib.sha256(
            f"{origin_record_id}::{position_channel.value}::{salt}".encode("utf-8")
        ).hexdigest()
        return f"ex-{h[:16]}"

    def to_dict(self) -> dict:
        return {
            "example_id": self.example_id,
            "text": self.text,
            "label": self.label.value,
            "position_channel": self.position_channel.value,
            "is_transplanted": self.is_transplanted,
            "source_dataset": self.source_dataset,
            "origin_record_id": self.origin_record_id,
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: dict) -> "TrainingExample":
        return TrainingExample(
            example_id=d["example_id"],
            text=d["text"],
            label=Role(d["label"]),
            position_channel=Role(d["position_channel"]),
            is_transplanted=d["is_transplanted"],
            source_dataset=d["source_dataset"],
            origin_record_id=d["origin_record_id"],
            meta=d.get("meta") or {},
        )


@dataclass
class SwapPairItem:
    """One held-out swap-test item: the SAME underlying span, rendered two
    ways -- natively (bare, as it would naturally appear) and foreign
    (wrapped in a different channel's carrier). NEVER used in training
    (Section 8, C3: "Swap-pair reservation... NEVER in training").

    The swap-consistency metric (Experiment 1) checks whether the trained
    encoder predicts the SAME label for both renderings.
    """

    pair_id: str
    label: Role                 # ground-truth origin register (shared by both renderings)
    native_text: str            # bare / native-channel rendering
    foreign_text: str           # foreign-channel-carrier rendering
    foreign_channel: Role
    source_dataset: str
    origin_record_id: str

    def to_dict(self) -> dict:
        return {
            "pair_id": self.pair_id,
            "label": self.label.value,
            "native_text": self.native_text,
            "foreign_text": self.foreign_text,
            "foreign_channel": self.foreign_channel.value,
            "source_dataset": self.source_dataset,
            "origin_record_id": self.origin_record_id,
        }

    @staticmethod
    def from_dict(d: dict) -> "SwapPairItem":
        return SwapPairItem(
            pair_id=d["pair_id"],
            label=Role(d["label"]),
            native_text=d["native_text"],
            foreign_text=d["foreign_text"],
            foreign_channel=Role(d["foreign_channel"]),
            source_dataset=d["source_dataset"],
            origin_record_id=d["origin_record_id"],
        )
