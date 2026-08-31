"""
scripts/build_dataset.py

Single entry point for C1 + C2: raw -> ingested -> clean spans, in one
command (previously two separate
commands). Idempotent-ish: re-running re-downloads/re-derives everything;
this is a laptop-scale project without a need for incremental caching logic
beyond what `datasets` already does internally.

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --skip-injection-corpora  # faster iteration on C3+ only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f"FAILED: {' '.join(cmd)} (exit code {result.returncode})")
        sys.exit(result.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-injection-corpora", action="store_true",
                     help="Skip BIPIA/NotInject/LLMail (Category A, slower, eval-only)")
    args = ap.parse_args()

    py = sys.executable

    print("=" * 70)
    print("STEP 1/2 -- C1: Ingestion")
    print("=" * 70)
    run([py, "-m", "src.ingestion.register_corpora"])
    if not args.skip_injection_corpora:
        run([py, "-m", "src.ingestion.bipia"])
        run([py, "-m", "src.ingestion.notinject"])
        run([py, "-m", "src.ingestion.llmail"])

    print("\n" + "=" * 70)
    print("STEP 2/2 -- C2: Preprocessing (strip/segment/dedup)")
    print("=" * 70)
    run([
        py, "-m", "src.preprocessing.build_clean_spans",
        "--in", "data/interim/register_sources.jsonl",
        "--out", "data/interim/clean_spans_register.jsonl",
    ])

    print("\nDone. Next: python -m src.supervision.build_supervision_sets "
          "--in data/interim/clean_spans_register.jsonl --out-dir data/processed")


if __name__ == "__main__":
    main()
