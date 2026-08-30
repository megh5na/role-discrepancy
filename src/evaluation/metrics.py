"""
src/evaluation/metrics.py

NEW harness / EXISTING metric definitions (Section 8 C10, Section 15).
Generic metric functions that work over any `predict_fn: list[str] -> list[Role]`
so the SAME code evaluates the encoder (C4) and the TF-IDF baseline (C8a)
identically -- important for a fair E1 comparison (Section 14 E1: "Baseline:
Same encoder trained with naive channel labels; TF-IDF+LR on both").
"""

from __future__ import annotations

from src.ingestion.schema import Role
from src.supervision.types import SwapPairItem


def swap_consistency(predict_fn, swap_pairs: list[SwapPairItem]) -> dict:
    """THE gate metric (Experiment 1, Section 14/15).

    swap_consistency: fraction of pairs where the model predicts the SAME
    label for the native and foreign-carrier renderings of the same span.
    This is the primary validity check -- "near chance -> the model learned
    position, and NOTHING downstream is interpretable" (Section 15).

    Also reports native_acc / foreign_acc (against the true origin-register
    label) as supplementary diagnostics: a model could be "consistent" by
    being confidently wrong on both renderings, so consistency alone doesn't
    tell the whole story -- these disambiguate that case.
    """
    if not swap_pairs:
        return {"swap_consistency": float("nan"), "native_acc": float("nan"), "foreign_acc": float("nan"), "n": 0}

    native_texts = [p.native_text for p in swap_pairs]
    foreign_texts = [p.foreign_text for p in swap_pairs]
    labels = [p.label for p in swap_pairs]

    native_preds = predict_fn(native_texts)
    foreign_preds = predict_fn(foreign_texts)

    n = len(swap_pairs)
    consistent = sum(1 for a, b in zip(native_preds, foreign_preds) if a == b)
    native_correct = sum(1 for a, l in zip(native_preds, labels) if a == l)
    foreign_correct = sum(1 for b, l in zip(foreign_preds, labels) if b == l)

    return {
        "swap_consistency": consistent / n,
        "native_acc": native_correct / n,
        "foreign_acc": foreign_correct / n,
        "n": n,
    }


def role_accuracy(predict_fn, texts: list[str], labels: list[Role]) -> float:
    preds = predict_fn(texts)
    correct = sum(1 for p, l in zip(preds, labels) if p == l)
    return correct / len(labels) if labels else float("nan")


def confusion_counts(predict_fn, texts: list[str], labels: list[Role]) -> dict:
    """label -> predicted -> count, for error analysis (Section 14)."""
    from collections import defaultdict

    preds = predict_fn(texts)
    table: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for p, l in zip(preds, labels):
        table[l.value][p.value] += 1
    return {k: dict(v) for k, v in table.items()}
