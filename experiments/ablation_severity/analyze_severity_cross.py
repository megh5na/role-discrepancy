from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.ablation_severity import analyze_severity_ablation as A
from experiments.exp03_baselines.run_exp03_wrapped import bootstrap, ci

HERE = Path(__file__).resolve().parent
F = 0.05
SCORERS = ("prob_deficit|asym", "prob_deficit|none")


def main():
    probs = {ck: A.load_probs(ck) for ck in A.CKPTS}
    scores = {}
    for ck, per_cond in probs.items():
        for v in SCORERS:
            scores[f"{ck}:{v}"] = {c: (A.score_variant(b, v), A.score_variant(i, v)) for c, (b, i) in per_cond.items()}
    conds = list(next(iter(scores.values())))
    point, draws = bootstrap(scores, conds, n_boot=1000, seed=42)

    out = {}
    for v in SCORERS:
        t, n = f"transplant_s42:{v}", f"naive_s42:{v}"
        print(f"\n=== scorer {v}: transplant minus naive (seed 42), 5% FPR, paired ===")
        rec = {}
        for unit in ("bare", "known_avg", "novel_avg"):
            d = draws[(t, unit, F, "B_auroc")] - draws[(n, unit, F, "B_auroc")]
            lo, hi = ci(d)
            rec[unit] = {"transplant_auroc": round(point[(t, unit, F, "B_auroc")], 3),
                         "naive_auroc": round(point[(n, unit, F, "B_auroc")], 3),
                         "diff": round(point[(t, unit, F, "B_auroc")] - point[(n, unit, F, "B_auroc")], 3),
                         "ci95": [lo, hi], "ci_excludes_zero": bool(lo > 0 or hi < 0)}
            r = rec[unit]
            print(f"  AUROC {unit:<10} transplant {r['transplant_auroc']:.3f}  naive {r['naive_auroc']:.3f}  "
                  f"diff {r['diff']:+.3f} [{lo:+.3f}, {hi:+.3f}]{' *' if r['ci_excludes_zero'] else ''}")
        for metric, label, sign in (("B_auroc", "AUROC drop (bare-wrapped)", 1),
                                    ("A_fpr_on_wrapped_benign", "FPR inflation (wrapped-bare)", -1)):
            for unit in ("known_avg", "novel_avg"):
                dr = {x: sign * (draws[(x, "bare", F, metric)] - draws[(x, unit, F, metric)]) for x in (t, n)}
                pt = {x: sign * (point[(x, "bare", F, metric)] - point[(x, unit, F, metric)]) for x in (t, n)}
                did = dr[n] - dr[t]
                lo, hi = ci(did)
                rec[f"{label}|{unit}"] = {"naive": round(float(pt[n]), 3), "transplant": round(float(pt[t]), 3),
                                          "did_naive_minus_transplant": round(float(pt[n] - pt[t]), 3),
                                          "ci95": [lo, hi], "ci_excludes_zero": bool(lo > 0 or hi < 0)}
                r = rec[f"{label}|{unit}"]
                print(f"  {label:<30}{unit:<10} naive {r['naive']:+.3f}  transplant {r['transplant']:+.3f}  "
                      f"DiD {r['did_naive_minus_transplant']:+.3f} [{lo:+.3f}, {hi:+.3f}]{' *' if r['ci_excludes_zero'] else ''}")
        out[v] = rec
    with open(HERE / "analysis_severity_cross.json", "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
