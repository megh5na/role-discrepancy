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
