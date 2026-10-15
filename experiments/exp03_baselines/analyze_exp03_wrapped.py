"""
experiments/exp03_baselines/analyze_exp03_wrapped.py

Follow-up analysis of E3-wrapped from the saved per-span scores
(scores_exp03_wrapped.npz); needs no models.

The question E3-wrapped exists to answer is whether the NAIVE detector
degrades more than the TRANSPLANT detector when attacks are wrapped. That is
a difference-in-differences, so it needs the bootstrap draws (run_exp03_
wrapped.py only saved summaries):

    drop_d(metric)  = metric_d(bare) - metric_d(wrapped unit)
    DiD(metric)     = drop_naive - drop_transplant        (> 0: naive degrades more)

Also reports false-alarm inflation (benign FPR under the bare-calibrated
threshold minus its bare value), where larger means more false alarms.
Same resampling scheme and seed as run_exp03_wrapped.py (unique strings,
threshold re-derived inside every resample, one resample shared by both
detectors).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.exp03_baselines.run_exp03_wrapped import (
    bootstrap,
    ci,
)

HERE = Path(__file__).resolve().parent
DETS = ("ours_transplant", "ours_naive")
UNITS = ("known_avg", "novel_avg")
F = 0.05


def load_scores():
    npz = np.load(HERE / "scores_exp03_wrapped.npz")
    scores: dict = {}
    for key in npz.files:
        det, cond, kind = key.split("__")
        scores.setdefault(det, {}).setdefault(cond, [None, None])[0 if kind == "benign" else 1] = npz[key]
    cond_names = list(scores[DETS[0]])
    return {d: {c: tuple(v) for c, v in per.items()} for d, per in scores.items()}, cond_names


def main():
    scores, cond_names = load_scores()
    point, draws = bootstrap(scores, cond_names, n_boot=1000, seed=42)

    def val(det, unit, metric, src):
        return src[(det, unit, F, metric)]

    out = {}
    print(f"{'metric':<26}{'unit':<11}{'naive drop':<24}{'transplant drop':<24}{'DiD (naive - transplant)'}")
    for metric, label, sign in (
        ("B_auroc", "AUROC (recalibrated)", +1),
        ("B_tpr", "TPR @5% FPR (recalibrated)", +1),
        ("A_fpr_on_wrapped_benign", "benign FPR inflation (fixed thr)", -1),
    ):
        for unit in UNITS:
            rec = {}
            for src_name, src in (("point", point), ("draws", draws)):
                for d in DETS:
                    # drop = bare - wrapped for quality metrics; for FPR, inflation = wrapped - bare
                    v = sign * (val(d, "bare", metric, src) - val(d, unit, metric, src))
                    rec[(src_name, d)] = v
            did_pt = rec[("point", "ours_naive")] - rec[("point", "ours_transplant")]
            did_draws = rec[("draws", "ours_naive")] - rec[("draws", "ours_transplant")]
            lo, hi = ci(did_draws)
            out.setdefault(metric, {})[unit] = {
                "naive_drop": {"point": round(float(rec[("point", "ours_naive")]), 3),
                               "ci95": ci(rec[("draws", "ours_naive")])},
                "transplant_drop": {"point": round(float(rec[("point", "ours_transplant")]), 3),
                                    "ci95": ci(rec[("draws", "ours_transplant")])},
                "did_naive_minus_transplant": {"point": round(float(did_pt), 3), "ci95": [lo, hi],
                                               "ci_excludes_zero": bool(lo > 0 or hi < 0)},
            }
            r = out[metric][unit]
            print(f"{label:<26}{unit:<11}"
                  f"{r['naive_drop']['point']:+.3f} {str(r['naive_drop']['ci95']):<16}"
                  f"{r['transplant_drop']['point']:+.3f} {str(r['transplant_drop']['ci95']):<16}"
                  f"{r['did_naive_minus_transplant']['point']:+.3f} {r['did_naive_minus_transplant']['ci95']}"
                  f"{'  *' if r['did_naive_minus_transplant']['ci_excludes_zero'] else ''}")
    print("\n(* = 95% CI excludes zero.  For quality metrics a positive drop is a loss; "
          "for FPR inflation a positive value is more false alarms.)")
    with open(HERE / "analysis_exp03_wrapped.json", "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
