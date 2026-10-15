"""
experiments/exp03_baselines/analyze_exp03_naive_dedup.py

Deduplicated re-analysis of E3 / E3-naive.

WHY: the 800 benign / 800 injected BIPIA spans E3 uses are NOT 1600
independent samples. After delimiter stripping there are only 120 unique
injected strings (one repeats 16 times) and 549 unique benign strings, with
no string shared between the classes. Duplicates get identical scores, so
span-level McNemar / bootstrap tests (run_exp03_naive.py) count the same
attack many times and are overconfident. Here each unique string counts once,
and the bootstrap resamples unique strings (benign and injected), re-deriving
the matched-FPR threshold inside every resample, so threshold uncertainty is
included. The SAME resampled strings are used for both detectors in a
comparison, so differences are paired.

Reads scores_exp03_naive.npz (saved by run_exp03_naive.py); needs no models.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.exp03_baselines.run_exp03_naive import auroc, load_spans
from src.scoring.calibration import calibrate_threshold

HERE = Path(__file__).resolve().parent
FPRS = [0.01, 0.05]
N_BOOT = 2000


def unique_scores(texts, scores):
    """One score per unique string (mean over duplicates), in first-seen order."""
    order, acc = [], {}
    for t, s in zip(texts, scores):
        if t not in acc:
            order.append(t)
            acc[t] = []
        acc[t].append(s)
    return order, np.array([np.mean(acc[t]) for t in order]), max(
        float(np.ptp(acc[t])) for t in order)


def tpr_at_fpr(benign, injected, fpr):
    return float(np.mean(injected >= calibrate_threshold(benign, fpr)))


def main():
    benign_txt, injected_txt = load_spans()
    npz = np.load(HERE / "scores_exp03_naive.npz")
    names = sorted({k.split("__")[0] for k in npz.files})

    det = {}
    max_dup_spread = 0.0
    for n in names:
        _, b, sb = unique_scores(benign_txt, npz[f"{n}__benign"])
        _, i, si = unique_scores(injected_txt, npz[f"{n}__injected"])
        det[n] = (b, i)
        max_dup_spread = max(max_dup_spread, sb, si)
    n_b, n_i = len(next(iter(det.values()))[0]), len(next(iter(det.values()))[1])
    print(f"unique benign={n_b}  unique injected={n_i}  "
          f"(max score spread across duplicates of one string: {max_dup_spread:.2e})")

    rng = np.random.default_rng(42)
    boots = {n: {f: [] for f in FPRS} for n in names}
    auroc_boots = {n: [] for n in names}
    for _ in range(N_BOOT):
        bi = rng.integers(0, n_b, n_b)
        ii = rng.integers(0, n_i, n_i)
        for n, (b, i) in det.items():
            for f in FPRS:
                boots[n][f].append(tpr_at_fpr(b[bi], i[ii], f))
            auroc_boots[n].append(auroc(b[bi], i[ii]))

    def ci(x):
        lo, hi = np.quantile(x, [0.025, 0.975])
        return [round(float(lo), 3), round(float(hi), 3)]

    out = {"n_unique_benign": n_b, "n_unique_injected": n_i, "detectors": {}}
    print(f"\n{'detector':<36}{'TPR@1%':<22}{'TPR@5%':<22}{'AUROC'}")
    for n, (b, i) in det.items():
        row = {"auroc": auroc(b, i), "auroc_ci95": ci(auroc_boots[n])}
        line = f"{n:<36}"
        for f in FPRS:
            t = tpr_at_fpr(b, i, f)
            row[f"tpr_{f}"] = t
            row[f"tpr_{f}_ci95"] = ci(boots[n][f])
            line += f"{t:.3f} {ci(boots[n][f])!s:<14}"
        line += f"{row['auroc']:.3f} {row['auroc_ci95']}"
        out["detectors"][n] = row
        print(line)

    for ref in ("ours_transplant", "ours_naive"):
        print(f"\nPaired differences, {ref} minus other (same resampled strings):")
        out["paired_vs_" + ref] = {}
        for n in names:
            if n == ref:
                continue
            d = {}
            for f in FPRS:
                diff = np.array(boots[ref][f]) - np.array(boots[n][f])
                pt = tpr_at_fpr(*det[ref], f) - tpr_at_fpr(*det[n], f)
                d[f"tpr_{f}"] = {"diff": round(pt, 3), "ci95": ci(diff)}
            ad = np.array(auroc_boots[ref]) - np.array(auroc_boots[n])
            d["auroc"] = {"diff": round(auroc(*det[ref]) - auroc(*det[n]), 3), "ci95": ci(ad)}
            out["paired_vs_" + ref][n] = d
            print(f"  vs {n:<34} TPR@1% {d['tpr_0.01']['diff']:+.3f} {d['tpr_0.01']['ci95']}   "
                  f"TPR@5% {d['tpr_0.05']['diff']:+.3f} {d['tpr_0.05']['ci95']}   "
                  f"AUROC {d['auroc']['diff']:+.3f} {d['auroc']['ci95']}")

    with open(HERE / "analysis_exp03_naive_dedup.json", "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
