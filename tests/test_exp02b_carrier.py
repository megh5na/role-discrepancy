"""tests/test_exp02b_carrier.py -- E2b evaluation logic, offline.

Uses fake predict functions, so no model is loaded or trained. The training
path is the same code E2 already ran; what is new in E2b, and tested here, is
how the held-out evaluation set is built and scored."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp02_supervision_compare.run_exp02b_carrier import (  # noqa: E402
    build_eval_set,
    evaluate_bare_and_wrapped,
    summarize,
)
from src.ingestion.schema import DataCategory, Record, Role  # noqa: E402


def _records(n=30, role=Role.DOCUMENT):
    return [
        Record(
            record_id=f"r{i}", text=f"This is held out sentence number {i}.",
            declared_role=role, source_dataset="dsX",
            category=DataCategory.REGISTER_SOURCE,
        )
        for i in range(n)
    ]


def test_eval_set_is_deterministic_for_a_seed():
    recs = _records()
    a = build_eval_set(recs, 10, eval_seed=7)
    b = build_eval_set(recs, 10, eval_seed=7)
    assert [r.record_id for r in a[0]] == [r.record_id for r in b[0]]
    assert a[1] == b[1] and a[2] == b[2]


def test_eval_set_respects_cap_and_small_pools():
    recs = _records(30)
    assert len(build_eval_set(recs, 10, 7)[0]) == 10
    assert len(build_eval_set(recs, 500, 7)[0]) == 30  # cap larger than pool


def test_wrap_channel_is_never_the_true_origin():
    recs = _records(50, role=Role.TOOL)
    _, wrapped, channels = build_eval_set(recs, 50, 7)
    assert Role.TOOL.value not in channels
    assert len(wrapped) == len(channels) == 50


def test_wrapped_text_contains_the_original_span():
    recs = _records(20)
    sample, wrapped, _ = build_eval_set(recs, 20, 7)
    # a TOOL carrier swaps double quotes for single quotes; these spans have none
    for rec, w in zip(sample, wrapped):
        assert rec.text in w


def test_evaluate_scores_bare_and_wrapped_separately():
    recs = _records(10)
    sample, wrapped, _ = build_eval_set(recs, 10, 7)
    bare_texts = {r.text for r in sample}

    def fake_predict(texts):
        # right on bare text, always wrong on anything wrapped
        return [Role.DOCUMENT if t in bare_texts else Role.TOOL for t in texts]

    m = evaluate_bare_and_wrapped(fake_predict, sample, wrapped)
    # carrier templates that leave the span unchanged would count as "bare";
    # the USER carrier "{span}" does exactly that, so count them explicitly
    unchanged = sum(1 for r, w in zip(sample, wrapped) if r.text == w)
    assert m["bare_accuracy"] == 1.0
    assert m["wrapped_accuracy"] == pytest.approx(unchanged / 10)
    assert m["bare_wrapped_agreement"] == pytest.approx(unchanged / 10)
    assert len(m["bare_correct"]) == len(m["wrapped_correct"]) == 10
    assert m["true_label"] == "document"


def test_summarize_averages_seeds_within_fold_before_folds():
    def row(fold, seed, bare, wrapped, agree):
        return {"model": "m", "condition": "naive", "held_out_corpus": fold, "seed": seed,
                "bare_accuracy": bare, "wrapped_accuracy": wrapped,
                "bare_wrapped_agreement": agree}

    rows = [
        row("A", 1, 0.2, 0.0, 0.5), row("A", 2, 0.4, 0.2, 0.5),  # fold A mean bare 0.3
        row("B", 1, 0.9, 0.5, 1.0),                              # fold B has one seed
    ]
    s = summarize(rows)["m|naive"]
    # (0.3 + 0.9) / 2, not the mean of the three raw rows
    assert s["bare_accuracy"] == pytest.approx(0.6)
    assert s["wrapped_accuracy"] == pytest.approx((0.1 + 0.5) / 2)
    assert s["n_folds"] == 2
