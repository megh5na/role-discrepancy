"""
experiments/exp03_baselines/analyze_exp03_seeds_perm.py

Seed-level inference for the E3 seed study. analyze_exp03_seeds.py gives
intervals over STRING sampling only; the unit of replication for the
naive-vs-transplant question is the trained model (3 seeds per condition).

Exact permutation test: each model contributes one number per quantity; the
statistic is mean(naive) - mean(transplant). The null reshuffles which 3 of
the 6 models are called "naive" (C(6,3) = 20 labelings). With 3 vs 3 models the
smallest attainable p-value is 1/20 = 0.05 one-sided and 0.10 two-sided, so
this test can never show strong significance at this seed count; it can only
say whether the observed split is the most extreme one possible.

Uses point estimates per model (5% FPR) from the saved scores; no models needed.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from experiments.exp03_baselines.analyze_exp03_seeds import load_models
from experiments.exp03_baselines.run_exp03_wrapped import one_pass

HERE = Path(__file__).resolve().parent
F = 0.05


def per_model_quantities(models):
    cond = list(next(iter(models.values())))
    nb = len(next(iter(models.values()))["bare"][0])
    ni = len(next(iter(models.values()))["bare"][1])
    pt = one_pass(models, cond, np.arange(nb), np.arange(ni))
    q = {}
    for m in models:
        g = lambda u, k: pt[(m, u, F, k)]  # noqa: E731
        q[m] = {
            "bare_auroc": g("bare", "B_auroc"),
            "known_wrapped_auroc": g("known_avg", "B_auroc"),
            "novel_wrapped_auroc": g("novel_avg", "B_auroc"),
            "auroc_drop_known": g("bare", "B_auroc") - g("known_avg", "B_auroc"),
            "auroc_drop_novel": g("bare", "B_auroc") - g("novel_avg", "B_auroc"),
            "tpr_drop_known": g("bare", "B_tpr") - g("known_avg", "B_tpr"),
            "tpr_drop_novel": g("bare", "B_tpr") - g("novel_avg", "B_tpr"),
            "fpr_inflation_known": g("known_avg", "A_fpr_on_wrapped_benign") - g("bare", "A_fpr_on_wrapped_benign"),
            "fpr_inflation_novel": g("novel_avg", "A_fpr_on_wrapped_benign") - g("bare", "A_fpr_on_wrapped_benign"),
        }
    return q


def permutation_test(values: dict[str, float], is_naive: dict[str, bool]):
    names = sorted(values)
    obs = np.mean([values[m] for m in names if is_naive[m]]) - np.mean([values[m] for m in names if not is_naive[m]])
    stats = []
    for naive_set in itertools.combinations(names, len(names) // 2):
        s = set(naive_set)
        stats.append(np.mean([values[m] for m in names if m in s])
                     - np.mean([values[m] for m in names if m not in s]))
    stats = np.array(stats)
    return {
        "observed_naive_minus_transplant": round(float(obs), 3),
        "n_labelings": len(stats),
        "p_one_sided_naive_greater": round(float(np.mean(stats >= obs - 1e-12)), 3),
        "p_two_sided": round(float(np.mean(np.abs(stats) >= abs(obs) - 1e-12)), 3),
    }


def main():
    models = load_models()
    is_naive = {m: m.startswith("naive") for m in models}
    q = per_model_quantities(models)
    names = list(next(iter(q.values())))
    out = {"per_model": {m: {k: round(float(v), 3) for k, v in d.items()} for m, d in q.items()}, "permutation": {}}
    print(f"{'quantity':<24}{'naive (s42,s123,s456)':<30}{'transplant (s42,s123,s456)':<30}"
          f"{'naive-transplant':<18}{'p 1-sided':<11}{'p 2-sided'}")
    for k in names:
        vals = {m: q[m][k] for m in q}
        r = permutation_test(vals, is_naive)
        out["permutation"][k] = r
        fmt = lambda c: " ".join(f"{vals[f'{c}_s{s}']:+.3f}" for s in (42, 123, 456))  # noqa: E731
        print(f"{k:<24}{fmt('naive'):<30}{fmt('transplant'):<30}"
              f"{r['observed_naive_minus_transplant']:+.3f}{'':<9}{r['p_one_sided_naive_greater']:<11}{r['p_two_sided']}")
    with open(HERE / "analysis_exp03_seeds_perm.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nSmallest attainable p with 3 vs 3 models: 0.05 one-sided, 0.10 two-sided.")


if __name__ == "__main__":
    main()
