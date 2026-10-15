"""
service/build_demo_cases.py

DESIGN-AGNOSTIC DEMO GROUNDWORK (no UI). Builds service/demo_cases.json: example texts for the demo, each with the
scores our detectors ACTUALLY gave it (from saved role probabilities; the scorer's parity test shows these match a
live run to < 1e-4).

NO CHERRY-PICKING: within each category the examples are drawn uniformly at random with a fixed seed (SEED), so
misses and false alarms appear at roughly the rate they occur. The category list was fixed before drawing, and every
case records its category, source and the outcome at both thresholds. A case that is wrong stays in.

Categories (all declared DOCUMENT unless stated):
  benign_email_text       BIPIA benign sentences (should NOT flag)
  injected_original       BIPIA injected spans, original wording (should flag)
  destyled_attack         E7 variants: the SAME attack at destyling levels 0.0 / 0.5 / 1.0 (score should fall; so does attack success)
  user_pastes_document    E6 benign: declared USER, text reads as a document (severity matrix should spare it)
  document_quotes_request E6 benign: declared DOCUMENT, text quotes a request (KNOWN over-defense: expect false alarms)
  control_user            E6 benign: declared USER, ordinary user text (should NOT flag)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from experiments.exp03_baselines.run_exp03_naive import load_spans
from experiments.exp03_baselines.run_exp03_wrapped import unique_in_order
from src.ingestion.io_utils import read_records
from src.ingestion.schema import Role
from src.scoring.asymmetry import expected_severity

REPO = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SEED = 11
K = 4
SEEDS = (42, 123, 456)


def ens_score(per_model_probs: dict, cond: str, declared: Role) -> float:
    return float(np.mean([expected_severity(per_model_probs[f"{cond}_s{s}"], declared) for s in SEEDS]))


def main():
    th = json.loads((HERE / "thresholds.json").read_text())
    rng = random.Random(SEED)
    abl = {f"{c}_s{s}": np.load(REPO / "experiments" / "ablation_severity" / f"probs_{c}_s{s}.npz") for c in ("naive", "transplant") for s in SEEDS}
    e6 = {m: np.load(REPO / "experiments" / "exp06_overdefense" / f"e6_probs_{m}.npz") for m in abl}
    meta = json.loads((REPO / "experiments" / "exp06_overdefense" / "e6_texts_meta.json").read_text())
    corpus = list(read_records(REPO / "data" / "interim" / "benign_role_mixed.jsonl"))
    benign, injected = load_spans()
    benign_u, injected_u = unique_in_order(benign), unique_in_order(injected)

    cases = []

    def add(category, source, text, declared: Role, probs_by_model: dict, note=""):
        row = {"category": category, "source": source, "declared_role": declared.value, "text": text, "note": note, "scores": {}}
        for cond in ("naive", "transplant"):
            s = ens_score(probs_by_model, cond, declared)
            row["scores"][cond] = {"score": round(s, 4), "flagged_pooled": bool(s >= th["thresholds"][cond]),
                                   "flagged_per_channel": bool(s >= th["thresholds_per_channel"][cond][declared.value]),
                                   "per_channel_threshold_reliable": declared.value not in ("tool", "reasoning")}
        cases.append(row)

    for cat, texts, key in (("benign_email_text", benign_u, "bare__benign"), ("injected_original", injected_u, "bare__injected")):
        for i in sorted(rng.sample(range(len(texts)), K)):
            add(cat, "BIPIA", texts[i], Role.DOCUMENT, {m: abl[m][key][i] for m in abl},
                "should flag" if cat == "injected_original" else "should NOT flag")

    for cell, cat, note in (("user_pastes_document", "user_pastes_document", "benign; severity matrix should spare it"),
                            ("document_quotes_request", "document_quotes_request", "benign; KNOWN over-defense, false alarms expected"),
                            ("control_user", "control_user", "benign, should NOT flag")):
        idx = [i for i, c in enumerate(meta["corpus_cells"]) if c == cell]
        for i in sorted(rng.sample(idx, K)):
            add(cat, f"E6 corpus ({cell})", corpus[i].text, Role(meta["corpus_declared"][i]), {m: e6[m]["corpus"][i] for m in e6}, note)

    # E7: same attack at three destyling levels (the first K attacks in sorted record_id order after a seeded shuffle)
    var = [json.loads(l) for l in open(REPO / "experiments" / "exp07_adaptive" / "mt_variants.jsonl")]
    ids = sorted({v["record_id"] for v in var}); rng.shuffle(ids)
    for rid in ids[:K]:
        for lv in (0.0, 0.5, 1.0):
            v = next(x for x in var if x["record_id"] == rid and x["destyling_level"] == lv)
            probs = {m: np.array(v["probs"][m]) for m in abl}
            add("destyled_attack", f"E7 ({v['injection_type']}, level {lv})", v["variant_text"], Role.DOCUMENT, probs,
                f"destyling level {lv}; in E7 the attack mostly stops working at 0.5 and 1.0")

    out = {"selection": f"uniform random within category, seed {SEED}, {K} per category (destyled_attack: {K} attacks x 3 levels); fixed before drawing; wrong cases kept",
           "thresholds_file": "service/thresholds.json", "cases": cases}
    (HERE / "demo_cases.json").write_text(json.dumps(out, indent=1))
    print(f"{len(cases)} cases")
    from collections import defaultdict
    agg = defaultdict(lambda: {"n": 0, "naive_flag": 0, "transplant_flag": 0})
    for c in cases:
        a = agg[c["category"]]; a["n"] += 1
        a["naive_flag"] += c["scores"]["naive"]["flagged_pooled"]; a["transplant_flag"] += c["scores"]["transplant"]["flagged_pooled"]
    for k, v in agg.items():
        print(f"  {k:24s} n={v['n']:2d}  flagged (pooled threshold): naive {v['naive_flag']}  transplant {v['transplant_flag']}")


if __name__ == "__main__":
    main()
