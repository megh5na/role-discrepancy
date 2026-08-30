"""
src/supervision/build_supervision_sets.py

*** OUR CONTRIBUTION (NEW) *** — pipeline entry point wiring transplant.py +
balance.py together, and running the balance diagnostics before anything
gets trained on the result.

Usage:
    python -m src.supervision.build_supervision_sets \
        --in data/interim/clean_spans_register.jsonl \
        --out-dir data/processed \
        --transplant-rate 0.5
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.ingestion.io_utils import read_records
from src.supervision.balance import corpus_prediction_probe, ensure_position_coverage
from src.supervision.transplant import build_all_supervision_sets, group_by_label


def write_jsonl(items, path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out-dir", dest="out_dir", required=True)
    ap.add_argument("--transplant-rate", type=float, default=0.5)
    ap.add_argument("--n-swap-per-label", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    spans = list(read_records(args.in_path))
    print(f"Loaded {len(spans)} clean spans from {args.in_path}")

    by_label = group_by_label(spans)
    print("Per-label span counts (before balancing):")
    for label, group in sorted(by_label.items(), key=lambda kv: kv[0].value):
        corpora = Counter(s.source_dataset for s in group)
        print(f"  {label.value}: {len(group)}  corpora={dict(corpora)}")

    print("\nRunning corpus-prediction probe (diagnostic, not gated -- see balance.py docstring)...")
    probe_results = corpus_prediction_probe(by_label, seed=args.seed)
    for label, acc in sorted(probe_results.items()):
        print(f"  P(source_dataset | text) accuracy for label={label}: {acc:.3f}")
    if not probe_results:
        print("  (no label has >=2 source corpora at this point -- probe skipped)")

    sets = build_all_supervision_sets(
        spans,
        transplant_rate=args.transplant_rate,
        n_swap_pairs_per_label=args.n_swap_per_label,
        seed=args.seed,
    )

    out_dir = Path(args.out_dir)
    n_naive = write_jsonl(sets["naive_control"], out_dir / "naive_control.jsonl")
    n_transplant = write_jsonl(sets["transplant_train"], out_dir / "transplant_train.jsonl")
    n_swap = write_jsonl(sets["swap_test_pairs"], out_dir / "swap_test_pairs.jsonl")

    print(f"\nWrote {n_naive} naive-control examples -> {out_dir / 'naive_control.jsonl'}")
    print(f"Wrote {n_transplant} transplant-train examples -> {out_dir / 'transplant_train.jsonl'}")
    print(f"Wrote {n_swap} swap-test pairs -> {out_dir / 'swap_test_pairs.jsonl'}")

    coverage = ensure_position_coverage(sets["transplant_train"])
    print("\nPosition coverage in transplant_train (label -> position_channel -> count):")
    for label, positions in sorted(coverage.items(), key=lambda kv: kv[0].value):
        nonzero = {p.value: c for p, c in positions.items() if c > 0}
        print(f"  {label.value}: {nonzero}")

    # Structural leakage check: no origin_record_id should appear in BOTH a
    # training set and the swap pool (Section 8 C3(c): NEVER in training).
    train_ids = {ex.origin_record_id for ex in sets["naive_control"]} | {
        ex.origin_record_id for ex in sets["transplant_train"]
    }
    swap_ids = {p.origin_record_id for p in sets["swap_test_pairs"]}
    overlap = train_ids & swap_ids
    print(f"\nLeakage check: swap-pool / training overlap = {len(overlap)} (must be 0)")
    assert len(overlap) == 0, "LEAKAGE: a swap-test span appears in a training set!"


if __name__ == "__main__":
    main()
