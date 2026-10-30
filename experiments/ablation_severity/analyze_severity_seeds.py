from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.ablation_severity.analyze_severity_ablation import load_probs, score_variant
from experiments.exp03_baselines.analyze_exp03_ensemble import (
    GUARDRAILS,
    bare_bootstrap,
    guardrail_bare_scores,
    make_ensemble,
    make_rank_ensemble,
)
from experiments.exp03_baselines.analyze_exp03_seeds_perm import permutation_test
from experiments.exp03_baselines.run_exp03_wrapped import bootstrap, ci, one_pass

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1] / "experiments" / "exp03_baselines"
SEEDS = (42, 123, 456)
MODELS = [f"{c}_s{s}" for c in ("naive", "transplant") for s in SEEDS]
SCORERS = ("prob_deficit|asym", "prob_deficit|none")
F = 0.05
UNITS = ("bare", "known_avg", "novel_avg")


def saved_scores_for(model: str):
    cond, seed = model.split("_s")
    if seed == "42":
        z = np.load(BASE / "scores_exp03_wrapped.npz")
        det = "ours_naive" if cond == "naive" else "ours_transplant"
        return lambda key: z[f"{det}__{key}"]
    z = np.load(BASE / f"scores_exp03_wrapped_seed{seed}_{cond}.npz")
    return lambda key: z[key]


def quantities(point: dict, model: str):
    g = lambda u, k: point[(model, u, F, k)]  # noqa: E731
    return {
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


def main():
    out: dict = {}
    probs = {m: load_probs(m) for m in MODELS}
    conds = list(probs[MODELS[0]])

    # ---- A. validation against everything previously saved
    worst = 0.0
    for m in MODELS:
        get = saved_scores_for(m)
        for c, (b, i) in probs[m].items():
            for kind, arr in (("benign", b), ("injected", i)):
                worst = max(worst, float(np.max(np.abs(score_variant(arr, SCORERS[0]) - get(f"{c}__{kind}")))))
    out["validation_max_abs_diff"] = worst
    print(f"A. validation: max |asym score from saved probabilities - previously saved score| over 6 models = {worst:.2e}")
    assert worst < 1e-4

    scores = {f"{m}:{v}": {c: (score_variant(b, v), score_variant(i, v)) for c, (b, i) in probs[m].items()}
              for m in MODELS for v in SCORERS}
    nb, ni = len(probs[MODELS[0]]["bare"][0]), len(probs[MODELS[0]]["bare"][1])
    point = one_pass(scores, conds, np.arange(nb), np.arange(ni))

    # ---- B/C. per-model numbers, seed spread, seed-level permutation tests
    out["per_model"], out["permutation"] = {}, {}
    for v in SCORERS:
        q = {m: quantities(point, f"{m}:{v}") for m in MODELS}
        out["per_model"][v] = {m: {k: round(float(x), 3) for k, x in d.items()} for m, d in q.items()}
        print(f"\n=== scorer {v} ===")
        print(f"{'model':<17}{'bare AUROC':<12}{'known-wrapped':<15}{'novel-wrapped':<15}{'benign FPR known/novel (bare thr)'}")
        for m in MODELS:
            d = q[m]
            print(f"{m:<17}{d['bare_auroc']:<12.3f}{d['known_wrapped_auroc']:<15.3f}{d['novel_wrapped_auroc']:<15.3f}"
                  f"{d['fpr_inflation_known'] + point[(f'{m}:{v}', 'bare', F, 'A_fpr_on_wrapped_benign')]:.3f} / "
                  f"{d['fpr_inflation_novel'] + point[(f'{m}:{v}', 'bare', F, 'A_fpr_on_wrapped_benign')]:.3f}")
        print("  seed spread of bare AUROC (mean +/- sd, [min, max]):")
        for cond in ("naive", "transplant"):
            vals = np.array([q[f"{cond}_s{s}"]["bare_auroc"] for s in SEEDS])
            print(f"    {cond:<11}{vals.mean():.3f} +/- {vals.std(ddof=1):.3f}  [{vals.min():.3f}, {vals.max():.3f}]")
            out.setdefault("seed_spread_bare_auroc", {}).setdefault(v, {})[cond] = {
                "mean": round(float(vals.mean()), 3), "sd": round(float(vals.std(ddof=1)), 3),
                "min": round(float(vals.min()), 3), "max": round(float(vals.max()), 3)}
        is_naive = {m: m.startswith("naive") for m in MODELS}
        print("  seed-level exact permutation test, naive minus transplant (3 vs 3):")
        print(f"  {'quantity':<24}{'naive s42/s123/s456':<26}{'transplant s42/s123/s456':<27}{'diff':<9}{'p1':<7}{'p2'}")
        for k in next(iter(q.values())):
            vals = {m: q[m][k] for m in MODELS}
            r = permutation_test(vals, is_naive)
            out["permutation"].setdefault(v, {})[k] = r
            fmt = lambda c: " ".join(f"{vals[f'{c}_s{s}']:+.2f}" for s in SEEDS)  # noqa: E731
            print(f"  {k:<24}{fmt('naive'):<26}{fmt('transplant'):<27}"
                  f"{r['observed_naive_minus_transplant']:+.3f}  {r['p_one_sided_naive_greater']:<7}{r['p_two_sided']}")

    # ---- D. per-model effect of the severity matrix (paired string bootstrap)
    print("\nD. what the severity matrix does per model: AUROC(none) - AUROC(asym), paired string bootstrap")
    bpoint, bdraws = bootstrap(scores, conds, n_boot=500, seed=42)
    out["severity_effect_per_model"] = {}
    for m in MODELS:
        row = {}
        for u in UNITS:
            d = bdraws[(f"{m}:{SCORERS[1]}", u, F, "B_auroc")] - bdraws[(f"{m}:{SCORERS[0]}", u, F, "B_auroc")]
            lo, hi = ci(d)
            row[u] = {"diff": round(float(bpoint[(f"{m}:{SCORERS[1]}", u, F, "B_auroc")]
                                          - bpoint[(f"{m}:{SCORERS[0]}", u, F, "B_auroc")]), 3), "ci95": [lo, hi]}
        out["severity_effect_per_model"][m] = row
        print(f"  {m:<17}" + "   ".join(f"{u}: {row[u]['diff']:+.3f} {row[u]['ci95']}" for u in UNITS))

    # ---- E. ensembles under the severity-free scorer
    none = {m: scores[f"{m}:{SCORERS[1]}"] for m in MODELS}
    naive_m, trans_m = [f"naive_s{s}" for s in SEEDS], [f"transplant_s{s}" for s in SEEDS]
    out["ensembles"] = {}
    for label, maker in (("score_avg", make_ensemble), ("rank_avg", make_rank_ensemble)):
        ens = {"ens_naive": maker(none, naive_m), "ens_transplant": maker(none, trans_m)}
        pt, dr = bootstrap(ens, conds, n_boot=1000, seed=42)
        print(f"\nE. ENSEMBLES, severity-free scorer, {label} (point; paired transplant - naive [95% CI over strings])")
        rec = {}
        for u in UNITS:
            a_n, a_t = pt[("ens_naive", u, F, "B_auroc")], pt[("ens_transplant", u, F, "B_auroc")]
            d = dr[("ens_transplant", u, F, "B_auroc")] - dr[("ens_naive", u, F, "B_auroc")]
            lo, hi = ci(d)
            rec[u] = {"naive": round(a_n, 3), "transplant": round(a_t, 3), "diff": round(a_t - a_n, 3), "ci95": [lo, hi]}
            print(f"   AUROC {u:<10} naive {a_n:.3f}  transplant {a_t:.3f}  diff {a_t - a_n:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                  f"{' *' if lo > 0 or hi < 0 else ''}")
        for metric, name, sign in (("B_auroc", "AUROC drop", 1), ("B_tpr", "TPR drop", 1),
                                   ("A_fpr_on_wrapped_benign", "FPR inflation", -1)):
            for u in ("known_avg", "novel_avg"):
                drp = {e: sign * (dr[(e, "bare", F, metric)] - dr[(e, u, F, metric)]) for e in ens}
                ptd = {e: sign * (pt[(e, "bare", F, metric)] - pt[(e, u, F, metric)]) for e in ens}
                did = drp["ens_naive"] - drp["ens_transplant"]
                lo, hi = ci(did)
                rec[f"{name}|{u}"] = {"naive": round(float(ptd["ens_naive"]), 3), "transplant": round(float(ptd["ens_transplant"]), 3),
                                      "did_naive_minus_transplant": round(float(ptd["ens_naive"] - ptd["ens_transplant"]), 3),
                                      "ci95": [lo, hi]}
                print(f"   {name:<14}{u:<10} naive {ptd['ens_naive']:+.3f}  transplant {ptd['ens_transplant']:+.3f}  "
                      f"DiD {ptd['ens_naive'] - ptd['ens_transplant']:+.3f} [{lo:+.3f}, {hi:+.3f}]{' *' if lo > 0 or hi < 0 else ''}")
        # bare detection vs guardrails
        bare = {n: ens[n]["bare"] for n in ens}
        bare.update(guardrail_bare_scores())
        gp, gd = bare_bootstrap(bare, n_boot=2000)
        print("   bare vs guardrails (unique strings):")
        grec = {}
        for n in bare:
            grec[n] = {k: round(gp[(n, k)], 3) for k in ("auroc", "tpr_0.01", "tpr_0.05")}
            print(f"     {n:<36}AUROC {gp[(n, 'auroc')]:.3f} {ci(gd[(n, 'auroc')])}  TPR@5% {gp[(n, 'tpr_0.05')]:.3f} {ci(gd[(n, 'tpr_0.05')])}")
        for e in ens:
            for g in GUARDRAILS:
                dd = gd[(e, "auroc")] - gd[(g, "auroc")]
                lo, hi = ci(dd)
                grec[f"{e}_minus_{g}_auroc"] = {"diff": round(gp[(e, "auroc")] - gp[(g, "auroc")], 3), "ci95": [lo, hi]}
                print(f"     {e} minus {g}: AUROC {gp[(e, 'auroc')] - gp[(g, 'auroc')]:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                      f"{' *' if lo > 0 or hi < 0 else ''}")
        out["ensembles"][label] = {"conditions": rec, "bare_vs_guardrails": grec}

    with open(HERE / "analysis_severity_seeds.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\n(* = 95% CI over unique strings excludes zero. String intervals exclude training-seed "
          "uncertainty; the permutation p-values are the seed-level evidence and cannot be below 0.05 / 0.10.)")


if __name__ == "__main__":
    main()
