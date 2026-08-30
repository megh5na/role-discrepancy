"""
src/evaluation/over_defense.py

NEW measurement (Section 6 NEW-3, Experiment 6 — "FIRST measurement of the
benign role-mismatch rate on legitimate traffic... A novel measurement
regardless of its value"). Architecture component: C10.

Computes the false-positive / benign-mismatch rate, OVERALL and BROKEN DOWN
BY CHANNEL PAIR (perceived_role, declared_role) -- the breakdown is what
turns "our FPR is X%" into something actionable: Section 6 NEW-2's whole
argument for asymmetric severity is that mismatch concentrates in specific,
mostly-benign channel-pair directions (e.g. perceived=DOCUMENT under
declared=USER, someone pasting an email) rather than uniformly.

This module is detector-agnostic: it takes a set of already-computed
(perceived_role, declared_role, flagged) triples and aggregates them. The
discrepancy scorer (src/scoring/) or a guardrail baseline both produce
inputs in this shape.
"""

from __future__ import annotations

from collections import defaultdict

from src.ingestion.schema import Role


def overall_fpr(flags: list[bool]) -> float:
    if not flags:
        return float("nan")
    return sum(flags) / len(flags)


def fpr_by_channel_pair(
    perceived_roles: list[Role], declared_roles: list[Role], flags: list[bool]
) -> dict[str, dict]:
    """Returns {"perceived->declared": {"fpr": ..., "n": ...}} for every
    channel pair actually observed in the data -- not every possible pair,
    so a sparse pair with n=2 doesn't get reported alongside n=500 pairs
    without a visible sample-size caveat (the "n" field IS that caveat)."""
    buckets: dict[tuple[Role, Role], list[bool]] = defaultdict(list)
    for p, d, f in zip(perceived_roles, declared_roles, flags):
        buckets[(p, d)].append(f)

    out = {}
    for (p, d), bucket_flags in buckets.items():
        key = f"{p.value}->{d.value}"
        out[key] = {
            "fpr": sum(bucket_flags) / len(bucket_flags),
            "n": len(bucket_flags),
        }
    return out


def summarize_over_defense(
    perceived_roles: list[Role], declared_roles: list[Role], flags: list[bool]
) -> dict:
    """Top-level Experiment 6 output: overall FPR + per-channel-pair
    breakdown, sorted by FPR descending so the worst offenders are first --
    directly usable for the failure-browser / error-analysis writeup
    (Section 14: "Error analysis... Source of E6's qualitative
    characterisation")."""
    overall = overall_fpr(flags)
    by_pair = fpr_by_channel_pair(perceived_roles, declared_roles, flags)
    sorted_pairs = dict(sorted(by_pair.items(), key=lambda kv: kv[1]["fpr"], reverse=True))
    return {"overall_fpr": overall, "n_total": len(flags), "by_channel_pair": sorted_pairs}
