"""
experiments/exp04_generalization/build_e4_sets.py

Builds the E4 inputs. See docs/decisions.md "E4 PRE-REGISTRATION" for the design
and the confirmation criterion fixed before any LLMail scoring.

 * LLMail short set: unique bodies with 5 to 45 words and ASCII fraction >= 0.9
   (English-oriented encoders, 64-token window), delimiters stripped, level group
   parsed from the `scenario` field. LLMail has no benign emails.
 * BIPIA family map: stripped unique injected string -> attack_category, aligned
   with the unique-in-order lists the earlier runs used.
Outputs: data/interim/e4_llmail_short.jsonl (gitignored), e4_sets_meta.json.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from experiments.exp03_baselines.run_exp03_naive import load_spans
from experiments.exp03_baselines.run_exp03_wrapped import unique_in_order
from src.ingestion.io_utils import read_records
from src.preprocessing.delimiters import strip_delimiters

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MIN_W, MAX_W, MIN_ASCII = 5, 45, 0.9


def ascii_frac(t: str) -> float:
    return sum(ord(c) < 128 for c in t) / max(1, len(t))


def main():
    rows = list(read_records(REPO / "data" / "interim" / "llmail.jsonl"))
    seen, kept = set(), []
    for r in rows:
        t = strip_delimiters(r.text)
        if not (MIN_W <= len(t.split()) <= MAX_W) or ascii_frac(t) < MIN_ASCII or t in seen:
            continue
        seen.add(t)
        m = re.match(r"level(\d)", r.injection_type or "")
        kept.append({"text": t, "scenario": r.injection_type, "level": m.group(1) if m else "other"})
    out = REPO / "data" / "interim" / "e4_llmail_short.jsonl"
    out.write_text("\n".join(json.dumps(k) for k in kept) + "\n")

    # BIPIA family map aligned with the unique-in-order injected list used everywhere
    _, injected = load_spans()
    order = unique_in_order(injected)
    fam = {}
    for r in read_records(REPO / "data" / "interim" / "bipia.jsonl"):
        if r.is_injected:
            fam.setdefault(strip_delimiters(r.text), r.injection_type)
    bipia_family = [fam[t] for t in order]
    meta = {"llmail_n": len(kept), "llmail_levels": {lv: sum(k["level"] == lv for k in kept) for lv in "1234"},
            "llmail_total_rows": len(rows), "criteria": {"min_words": MIN_W, "max_words": MAX_W, "min_ascii": MIN_ASCII},
            "bipia_family": bipia_family,
            "bipia_family_counts": {f: bipia_family.count(f) for f in sorted(set(bipia_family))}}
    (HERE / "e4_sets_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"LLMail short set: {len(kept)} unique emails of {len(rows)} rows; by level {meta['llmail_levels']}")
    print(f"BIPIA unique injected {len(order)}; per family {meta['bipia_family_counts']}")


if __name__ == "__main__":
    main()
