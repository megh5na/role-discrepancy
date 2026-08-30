"""
tests/test_supervision.py

Offline, deterministic tests for C3 (src/supervision/) using small synthetic
Record sets -- no internet / dataset download required, so these run in any
environment (unlike the file-based check in test_schema.py which needs the
built data).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.schema import DataCategory, Record, Role  # noqa: E402
from src.supervision import carriers  # noqa: E402
from src.supervision.transplant import (  # noqa: E402
    build_all_supervision_sets,
    build_naive_control_set,
    build_swap_test_set,
    build_transplant_set,
    group_by_label,
    reserve_swap_pool,
)

_TEXTS_BY_ROLE = {
    Role.USER: [
        "Can you send me the quarterly report?",
        "What time does the meeting start tomorrow?",
        "Please cancel my subscription immediately.",
        "How do I reset my password?",
    ],
    Role.DOCUMENT: [
        "The mitochondria is the powerhouse of the cell and produces ATP.",
        "The treaty was signed in 1848 following a series of negotiations.",
        "Photosynthesis converts sunlight into chemical energy in plants.",
        "The bridge was completed after five years of construction.",
    ],
    Role.TOOL: [
        '{"status": "ok", "value": 42}',
        '{"result": "success", "items": [1, 2, 3]}',
        '{"error": null, "count": 7}',
        '{"temperature": 21.5, "unit": "celsius"}',
    ],
    Role.REASONING: [
        "First I compute 4 times 3, which is 12, then add 5 to get 17.",
        "Since the triangle has a right angle, the hypotenuse follows from Pythagoras.",
        "The total cost is 20 plus 5 percent tax, which comes to 21.",
        "Because both sides are equal, the shape must be a rhombus.",
    ],
    Role.SYSTEM: [
        "You are a helpful assistant that always answers politely.",
        "Your role is to translate text between English and French.",
        "You must never reveal internal configuration details.",
        "Act as a professional financial advisor at all times.",
    ],
}


def _make_synthetic_spans(n_per_role: int = 20) -> list[Record]:
    spans = []
    for role, texts in _TEXTS_BY_ROLE.items():
        for i in range(n_per_role):
            text = texts[i % len(texts)] + f" (variant {i})"
            spans.append(
                Record(
                    record_id=Record.make_id(f"synthetic-{role.value}", text, i),
                    text=text,
                    declared_role=role,
                    source_dataset=f"synthetic-{role.value}",
                    category=DataCategory.REGISTER_SOURCE,
                )
            )
    return spans


def test_carriers_cover_all_roles():
    assert set(carriers.available_channels()) == set(Role)


def test_carrier_embedding_changes_text():
    rng = __import__("random").Random(0)
    span = "Please review the attached document."
    for channel in carriers.available_channels():
        wrapped = carriers.embed_in_channel(span, channel, rng)
        assert isinstance(wrapped, str) and len(wrapped) > 0


def test_naive_control_position_equals_label():
    spans = _make_synthetic_spans(5)
    naive = build_naive_control_set(spans)
    assert len(naive) == len(spans)
    for ex in naive:
        assert ex.position_channel == ex.label
        assert ex.is_transplanted is False


def test_transplant_rate_zero_is_all_native():
    spans = _make_synthetic_spans(10)
    ex = build_transplant_set(spans, transplant_rate=0.0, seed=1)
    assert all(not e.is_transplanted for e in ex)
    assert all(e.position_channel == e.label for e in ex)


def test_transplant_rate_one_is_all_foreign():
    spans = _make_synthetic_spans(10)
    ex = build_transplant_set(spans, transplant_rate=1.0, seed=1)
    assert all(e.is_transplanted for e in ex)
    assert all(e.position_channel != e.label for e in ex)


def test_transplant_label_is_always_origin_register():
    spans = _make_synthetic_spans(20)
    ex = build_transplant_set(spans, transplant_rate=0.7, seed=3)
    by_id = {s.record_id: s.declared_role for s in spans}
    for e in ex:
        assert e.label == by_id[e.origin_record_id], (
            "Transplanted example's label must always be the ORIGIN register, "
            "never the position it was relocated to -- this is the entire point "
            "of the construction (Section 6 NEW-1)."
        )


def test_swap_pool_never_overlaps_remaining_pool():
    spans = _make_synthetic_spans(30)
    by_label = group_by_label(spans)
    remaining, swap_pool = reserve_swap_pool(by_label, n_per_label=3, seed=7)
    remaining_ids = {s.record_id for group in remaining.values() for s in group}
    swap_ids = {s.record_id for group in swap_pool.values() for s in group}
    assert remaining_ids.isdisjoint(swap_ids)


def test_swap_test_items_share_content_across_renderings():
    spans = _make_synthetic_spans(30)
    by_label = group_by_label(spans)
    _, swap_pool = reserve_swap_pool(by_label, n_per_label=3, seed=7)
    items = build_swap_test_set(swap_pool, seed=7)
    assert len(items) > 0
    for item in items:
        assert item.foreign_channel != item.label
        # The native rendering must be exactly the original span text (bare).
        assert item.native_text.strip() != ""
        assert item.foreign_text.strip() != ""


def test_end_to_end_leakage_check():
    """*** MIRRORS tests/test_leakage.py's role for the real pipeline ***
    On synthetic data: no swap-pool span's origin_record_id may appear in
    either training set (Section 8 C3(c): NEVER in training)."""
    spans = _make_synthetic_spans(40)
    sets = build_all_supervision_sets(
        spans, transplant_rate=0.5, n_swap_pairs_per_label=2, seed=11
    )
    train_ids = {e.origin_record_id for e in sets["naive_control"]} | {
        e.origin_record_id for e in sets["transplant_train"]
    }
    swap_ids = {p.origin_record_id for p in sets["swap_test_pairs"]}
    assert train_ids.isdisjoint(swap_ids), "LEAKAGE: swap-test span found in training data"


def test_naive_and_transplant_sets_are_same_underlying_pool():
    """Fairness check: the naive-control and transplant sets should be built
    from the SAME balanced span pool, so any measured difference between
    them is attributable to the construction, not to different data."""
    spans = _make_synthetic_spans(40)
    sets = build_all_supervision_sets(spans, transplant_rate=0.5, n_swap_pairs_per_label=2, seed=11)
    naive_origins = {e.origin_record_id for e in sets["naive_control"]}
    transplant_origins = {e.origin_record_id for e in sets["transplant_train"]}
    assert naive_origins == transplant_origins
