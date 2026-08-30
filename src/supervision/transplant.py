"""
src/supervision/transplant.py

*** OUR CONTRIBUTION (NEW) *** — THE PRIMARY CONTRIBUTION (Section 6 NEW-1,
Section 13 E1, Section 17: "src/supervision/transplant.py -- cross-channel
relocation with style-following labels").

WHAT THIS BUILDS AND WHY (restated concretely, in code terms):

Given a span of text whose ORIGIN REGISTER is known (because it came from a
register-source corpus -- Category B, e.g. a Dolly instruction is
known-USER-register), this module produces training examples in one of two
ways, controlled by `transplant_rate`:

  - NATIVE (probability 1 - transplant_rate): the span is presented bare,
    exactly as it naturally occurs. label = position_channel = origin
    register. This is what "naive channel labelling" (Section 6, the
    baseline we're contrasting against) looks like for every example, all
    the time -- so transplant_rate=0.0 reduces exactly to the naive
    baseline, which is what makes the transplantation-rate sweep ablation
    (Section 14) a real ablation and not two unrelated code paths.

  - TRANSPLANTED (probability transplant_rate): the span's content is
    embedded in a FOREIGN channel's carrier (src/supervision/carriers.py)
    -- e.g. a Dolly instruction (USER-register) gets wrapped in JSON braces
    (TOOL-typical formatting). label = origin register (USER) still;
    position_channel = the foreign carrier (TOOL). This is the example that
    forces a model trying to fit BOTH conditions to actually read register,
    because "IS wrapped in JSON" no longer reliably predicts "IS tool
    register."

Swap pairs are reserved BEFORE either set is built and are never included in
either (Section 8 C3(c): "NEVER in training" -- enforced structurally here,
not just documented, by removing the swap pool from the span list before
`build_naive_control_set` / `build_transplant_set` ever see it).
"""

from __future__ import annotations

import random
from collections import defaultdict

from src.ingestion.schema import Record, Role
from src.supervision import carriers
from src.supervision.balance import balance_by_count, balance_by_length
from src.supervision.types import SwapPairItem, TrainingExample

DEFAULT_SWAP_PAIRS_PER_LABEL = 120


def group_by_label(spans: list[Record]) -> dict[Role, list[Record]]:
    out: dict[Role, list[Record]] = defaultdict(list)
    for s in spans:
        out[s.declared_role].append(s)
    return out


def reserve_swap_pool(
    spans_by_label: dict[Role, list[Record]],
    n_per_label: int = DEFAULT_SWAP_PAIRS_PER_LABEL,
    seed: int = 7,
) -> tuple[dict[Role, list[Record]], dict[Role, list[Record]]]:
    """Split each label's spans into (remaining_for_training, swap_pool).

    The swap pool is removed from `remaining_for_training` -- structurally,
    not just by convention -- so `build_naive_control_set` /
    `build_transplant_set`, which only ever see `remaining_for_training`,
    cannot accidentally include a swap-test span. This is what
    tests/test_leakage.py checks holds for the files actually written to
    disk.
    """
    rng = random.Random(seed)
    remaining: dict[Role, list[Record]] = {}
    swap_pool: dict[Role, list[Record]] = {}
    for label, spans in spans_by_label.items():
        shuffled = spans[:]
        rng.shuffle(shuffled)
        k = min(n_per_label, max(0, len(shuffled) // 10))  # never take more than 10% of a class
        swap_pool[label] = shuffled[:k]
        remaining[label] = shuffled[k:]
    return remaining, swap_pool


def build_naive_control_set(spans: list[Record]) -> list[TrainingExample]:
    """Baseline (b) from Section 6 NEW-1: bare natural text, labelled by
    origin corpus, with NO transplantation at all. Every example's
    position_channel equals its label by construction -- this is precisely
    the confound (Section 5) we expect to make a model follow POSITION
    rather than REGISTER when they're forced to disagree (the swap test).
    """
    examples = []
    for s in spans:
        examples.append(
            TrainingExample(
                example_id=TrainingExample.make_id(s.record_id, s.declared_role, "naive"),
                text=s.text,
                label=s.declared_role,
                position_channel=s.declared_role,
                is_transplanted=False,
                source_dataset=s.source_dataset,
                origin_record_id=s.record_id,
            )
        )
    return examples


def build_transplant_set(
    spans: list[Record], transplant_rate: float, seed: int = 42
) -> list[TrainingExample]:
    """THE construction. See module docstring for the mechanism."""
    rng = random.Random(seed)
    examples = []
    for s in spans:
        origin = s.declared_role
        if rng.random() < transplant_rate:
            foreign_channels = [c for c in carriers.available_channels() if c != origin]
            target = rng.choice(foreign_channels)
            text = carriers.embed_in_channel(s.text, target, rng)
            examples.append(
                TrainingExample(
                    example_id=TrainingExample.make_id(s.record_id, target, "transplant"),
                    text=text,
                    label=origin,
                    position_channel=target,
                    is_transplanted=True,
                    source_dataset=s.source_dataset,
                    origin_record_id=s.record_id,
                )
            )
        else:
            examples.append(
                TrainingExample(
                    example_id=TrainingExample.make_id(s.record_id, origin, "native"),
                    text=s.text,
                    label=origin,
                    position_channel=origin,
                    is_transplanted=False,
                    source_dataset=s.source_dataset,
                    origin_record_id=s.record_id,
                )
            )
    return examples


def build_swap_test_set(
    swap_pool_by_label: dict[Role, list[Record]], seed: int = 7
) -> list[SwapPairItem]:
    """For each held-out span, produce ONE native (bare) rendering and ONE
    foreign-carrier rendering of the SAME text, for the swap-consistency
    metric (Experiment 1): a well-trained encoder should predict the SAME
    label for both, since the actual content -- and therefore its register
    -- hasn't changed.
    """
    rng = random.Random(seed)
    items = []
    for label, spans in swap_pool_by_label.items():
        for s in spans:
            foreign_channels = [c for c in carriers.available_channels() if c != label]
            foreign_channel = rng.choice(foreign_channels)
            foreign_text = carriers.embed_in_channel(s.text, foreign_channel, rng)
            items.append(
                SwapPairItem(
                    pair_id=f"swap-{s.record_id}",
                    label=label,
                    native_text=s.text,
                    foreign_text=foreign_text,
                    foreign_channel=foreign_channel,
                    source_dataset=s.source_dataset,
                    origin_record_id=s.record_id,
                )
            )
    return items


def build_all_supervision_sets(
    spans: list[Record],
    transplant_rate: float = 0.5,
    n_swap_pairs_per_label: int = DEFAULT_SWAP_PAIRS_PER_LABEL,
    seed: int = 42,
) -> dict[str, list]:
    """Top-level entry point (C3). Returns:
        {
          "naive_control": list[TrainingExample],
          "transplant_train": list[TrainingExample],
          "swap_test_pairs": list[SwapPairItem],
        }

    Pipeline: group by label -> reserve swap pool (removed from training
    pool structurally) -> balance by count -> balance by length ->
    build naive control AND transplant sets from the SAME balanced pool
    (so any difference between the two conditions is due to the
    construction, not to different underlying spans).
    """
    by_label = group_by_label(spans)
    remaining, swap_pool = reserve_swap_pool(by_label, n_swap_pairs_per_label, seed=seed)

    balanced = balance_by_count(remaining, seed=seed)
    balanced = balance_by_length(balanced, seed=seed)

    pool = [s for spans_ in balanced.values() for s in spans_]

    naive = build_naive_control_set(pool)
    transplant = build_transplant_set(pool, transplant_rate=transplant_rate, seed=seed)
    swap_pairs = build_swap_test_set(swap_pool, seed=seed)

    return {
        "naive_control": naive,
        "transplant_train": transplant,
        "swap_test_pairs": swap_pairs,
    }
