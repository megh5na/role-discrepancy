"""
experiments/exp06_overdefense/analyze_continuous_severity.py

Does a continuous (probability-weighted) severity keep the shipped matrix's
over-defense benefit while removing its seed fragility? Scorers, all from saved
role probabilities of the six models (+ seed ensembles):
  asym      shipped: (1-P(declared)) x S[argmax, declared]
  none      base discrepancy alone
  cont_exp  sum_r P(r) S[r, declared]      (pre-specified primary)
  cont_max  max_r P(r) S[r, declared]      (sensitivity)
Questions:
  1 pooled separability (BIPIA injected vs 3 benign pools): AUROC, TPR@5% FPR
  2 deployment-like operating point: mixed-cell / NotInject FPR, BIPIA TPR
  3 seed spread (sd across seeds) per condition and pool
  4 carrier-wrapped BIPIA (bare / known / novel AUROC) and seed spread
  5 paired differences cont_exp - asym, record-level bootstrap (no seed uncertainty)
  6 seed-level permutation, naive vs transplant (3 vs 3)
Same caveats as analyze_e6.py (corpus benign by construction; record-level CIs).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.ablation_severity.analyze_severity_ablation import load_probs, score_variant
from experiments.exp03_baselines.analyze_exp03_seeds_perm import permutation_test
from experiments.exp03_baselines.run_exp03_wrapped import one_pass
from experiments.exp06_overdefense.analyze_e6 import MODELS, SEEDS, ci, load, metrics, score_decl
from experiments.exp06_overdefense.analyze_e6_pooled import SAME_CHANNEL_CELLS, evaluate
from experiments.exp06_overdefense.continuous_severity import CONTINUOUS
from src.ingestion.schema import Role

HERE = Path(__file__).resolve().parent
SCORERS = ["asym", "none", "cont_exp", "cont_max"]
UNITS = MODELS + ["ens_naive", "ens_transplant"]
POOLS = ["bipia_benign", "same_channel", "all_benign"]


def score(probs, declared_arr, scorer):
    if scorer in CONTINUOUS:
        out = np.empty(len(probs))
        for ch in set(declared_arr.tolist()):
            m = declared_arr == ch
            out[m] = CONTINUOUS[scorer](probs[m], Role(ch))
        return out
    return score_decl(probs, declared_arr, {"asym": "prob_deficit|asym", "none": "prob_deficit|none"}[scorer])


def main():
    meta, decl, control, cells, raw = load()
    sc = {}
    for m in MODELS:
        r = raw[m]
        for s in SCORERS:
            sc[(m, s)] = {"corpus": score(r["corpus"], decl, s),
                          "notinject": score(r["notinject"], np.array(["user"] * len(r["notinject"])), s),
                          "bipia_ben": score(r["bipia_ben"], np.array(["document"] * len(r["bipia_ben"])), s),
                          "bipia_inj": score(r["bipia_inj"], np.array(["document"] * len(r["bipia_inj"])), s)}
    for cond in ("naive", "transplant"):
        for s in SCORERS:
            mem = [f"{cond}_s{x}" for x in SEEDS]
            sc[(f"ens_{cond}", s)] = {k: np.mean([sc[(m, s)][k] for m in mem], axis=0) for k in sc[(mem[0], s)]}

    same = np.isin(cells, list(SAME_CHANNEL_CELLS))
    pools = {"bipia_benign": lambda d: d["bipia_ben"],
             "same_channel": lambda d: np.r_[d["bipia_ben"], d["corpus"][same]],
             "all_benign": lambda d: np.r_[d["bipia_ben"], d["corpus"], d["notinject"]]}
    ctrl_ids, mixed_ids = np.where(control)[0], np.where(~control)[0]
    d0 = sc[(UNITS[0], "asym")]
    n_i, n_b, n_n = len(d0["bipia_inj"]), len(d0["bipia_ben"]), len(d0["notinject"])
    psz = {p: len(f(d0)) for p, f in pools.items()}
    keys = [(u, s) for u in UNITS for s in SCORERS]

    sep = {(u, s, p): evaluate(sc[(u, s)]["bipia_inj"], f(sc[(u, s)])) for u, s in keys for p, f in pools.items()}
    ident = (np.arange(len(ctrl_ids)), np.arange(len(mixed_ids)), np.arange(n_i), np.arange(n_b), np.arange(n_n))
    opp = {k: metrics(sc[k], decl, ctrl_ids, mixed_ids, *ident) for k in keys}

    rng = np.random.default_rng(42)
    dsep = {(u, s, p): [] for u, s in keys for p in pools}
    dopp = {k: {} for k in keys}
    for _ in range(300):
        ii = rng.integers(0, n_i, n_i)
        nn = {p: rng.integers(0, n, n) for p, n in psz.items()}
        idx = (rng.integers(0, len(ctrl_ids), len(ctrl_ids)), rng.integers(0, len(mixed_ids), len(mixed_ids)),
               ii, rng.integers(0, n_b, n_b), rng.integers(0, n_n, n_n))
        for u, s in keys:
            for p, f in pools.items():
                dsep[(u, s, p)].append(evaluate(sc[(u, s)]["bipia_inj"][ii], f(sc[(u, s)])[nn[p]])[0])
            for k, v in metrics(sc[(u, s)], decl, ctrl_ids, mixed_ids, *idx).items():
                dopp[(u, s)].setdefault(k, []).append(v)
    dsep = {k: np.array(v) for k, v in dsep.items()}
    dopp = {k: {a: np.array(b) for a, b in v.items()} for k, v in dopp.items()}

    out = {}
    # 1. AUROC tables
    for p in POOLS:
        print(f"\n=== 1. AUROC, BIPIA injected vs {p} (TPR@5% FPR in brackets) ===")
        print(f"{'unit':<17}" + "".join(f"{s:<20}" for s in SCORERS))
        for u in UNITS:
            print(f"{u:<17}" + "".join(f"{sep[(u, s, p)][0]:.3f} ({sep[(u, s, p)][1]:.2f})".ljust(20) for s in SCORERS))
    out["auroc"] = {f"{u}|{s}|{p}": [round(sep[(u, s, p)][0], 3), round(sep[(u, s, p)][1], 3)]
                    for u, s in keys for p in POOLS}

    # 3. seed spread
    print("\n=== 3. seed-to-seed sd of AUROC (mean +/- sd over 3 seeds) ===")
    print(f"{'pool / condition':<28}" + "".join(f"{s:<20}" for s in SCORERS))
    out["seed_spread"] = {}
    for p in POOLS:
        for cond in ("naive", "transplant"):
            cells_ = []
            for s in SCORERS:
                v = np.array([sep[(f"{cond}_s{x}", s, p)][0] for x in SEEDS])
                cells_.append(f"{v.mean():.3f} +/- {v.std(ddof=1):.3f}")
                out["seed_spread"][f"{p}|{cond}|{s}"] = [round(float(v.mean()), 3), round(float(v.std(ddof=1)), 3)]
            print(f"{p + ' / ' + cond:<28}" + "".join(c.ljust(20) for c in cells_))

    # 2. operating point
    print("\n=== 2. deployment-like operating point (5% of pooled control cells flagged) ===")
    print(f"{'unit':<17}{'scorer':<10}{'mixed FPR':<12}{'NotInject FPR':<15}{'BIPIA benign FPR':<18}{'BIPIA TPR'}")
    for u in ["ens_naive", "ens_transplant"] + MODELS:
        for s in SCORERS:
            o = opp[(u, s)]
            print(f"{u:<17}{s:<10}{o['mixed_fpr']:<12.3f}{o['notinject_fpr']:<15.3f}{o['bipia_benign_fpr']:<18.3f}{o['bipia_tpr']:.3f}")
    out["operating_point"] = {f"{u}|{s}": {k: round(float(opp[(u, s)][k]), 3) for k in
                                           ("mixed_fpr", "notinject_fpr", "bipia_benign_fpr", "bipia_tpr")} for u, s in keys}

    # 5. paired differences vs asym
    print("\n=== 5. paired: scorer minus asym, record-level 95% CI (AUROC on each pool) ===")
    out["paired_vs_asym"] = {}
    for s in ("cont_exp", "cont_max", "none"):
        print(f"  {s} minus asym")
        for u in UNITS:
            row = {}
            cells_ = []
            for p in POOLS:
                d = dsep[(u, s, p)] - dsep[(u, "asym", p)]
                row[p] = {"diff": round(sep[(u, s, p)][0] - sep[(u, "asym", p)][0], 3), "ci95": ci(d)}
                cells_.append(f"{p}: {row[p]['diff']:+.3f} {row[p]['ci95']}")
            out["paired_vs_asym"][f"{s}|{u}"] = row
            print(f"    {u:<17}" + "   ".join(cells_))
    print("\n  operating point, cont_exp minus asym (record-level CIs): mixed FPR / NotInject FPR / BIPIA TPR")
    for u in UNITS:
        cells_ = []
        for met in ("mixed_fpr", "notinject_fpr", "bipia_tpr"):
            d = dopp[(u, "cont_exp")][met] - dopp[(u, "asym")][met]
            cells_.append(f"{met} {opp[(u, 'cont_exp')][met] - opp[(u, 'asym')][met]:+.3f} {ci(d)}")
        print(f"    {u:<17}" + "   ".join(cells_))

    # 6. seed-level permutation, all_benign AUROC and mixed FPR
    print("\n=== 6. naive minus transplant, seed-level exact permutation (3 vs 3; min p 0.05 / 0.10) ===")
    out["permutation"] = {}
    is_naive = {m: m.startswith("naive") for m in MODELS}
    for s in SCORERS:
        for label, getter in (("all_benign AUROC", lambda m, s=s: sep[(m, s, "all_benign")][0]),
                              ("mixed FPR", lambda m, s=s: opp[(m, s)]["mixed_fpr"])):
            vals = {m: getter(m) for m in MODELS}
            r = permutation_test(vals, is_naive)
            out["permutation"][f"{s}|{label}"] = r
            print(f"  {s:<9}{label:<18}naive {[round(vals[f'naive_s{x}'], 2) for x in SEEDS]}  "
                  f"transplant {[round(vals[f'transplant_s{x}'], 2) for x in SEEDS]}  "
                  f"diff {r['observed_naive_minus_transplant']:+.3f}  p2={r['p_two_sided']}")

    # 4. carrier-wrapped BIPIA
    print("\n=== 4. carrier-wrapped BIPIA (12 conditions), AUROC recalibrated, 5% FPR: bare / known / novel ===")
    wp = {m: load_probs(m) for m in MODELS}
    conds = list(wp[MODELS[0]])
    wsc = {}
    for m in MODELS:
        for s in SCORERS:
            wsc[f"{m}:{s}"] = {c: (score(b, np.array(["document"] * len(b)), s), score(i, np.array(["document"] * len(i)), s))
                               for c, (b, i) in wp[m].items()}
    nb, ni = len(wp[MODELS[0]]["bare"][0]), len(wp[MODELS[0]]["bare"][1])
    wpt = one_pass(wsc, conds, np.arange(nb), np.arange(ni))
    out["wrapped"] = {}
    print(f"{'condition / scorer':<26}{'bare mean+/-sd':<18}{'known mean+/-sd':<18}{'novel mean+/-sd':<18}{'AUROC drop known / novel'}")
    for cond in ("naive", "transplant"):
        for s in SCORERS:
            row = {}
            for unit in ("bare", "known_avg", "novel_avg"):
                v = np.array([wpt[(f"{cond}_s{x}:{s}", unit, 0.05, "B_auroc")] for x in SEEDS])
                row[unit] = (float(v.mean()), float(v.std(ddof=1)))
            dk = np.mean([wpt[(f"{cond}_s{x}:{s}", "bare", 0.05, "B_auroc")] - wpt[(f"{cond}_s{x}:{s}", "known_avg", 0.05, "B_auroc")] for x in SEEDS])
            dn = np.mean([wpt[(f"{cond}_s{x}:{s}", "bare", 0.05, "B_auroc")] - wpt[(f"{cond}_s{x}:{s}", "novel_avg", 0.05, "B_auroc")] for x in SEEDS])
            out["wrapped"][f"{cond}|{s}"] = {k: [round(a, 3), round(b, 3)] for k, (a, b) in row.items()} | {
                "drop_known": round(float(dk), 3), "drop_novel": round(float(dn), 3)}
            print(f"{cond + ' / ' + s:<26}" + "".join(f"{row[u][0]:.3f} +/- {row[u][1]:.3f}".ljust(18) for u in row)
                  + f"{dk:+.3f} / {dn:+.3f}")

    (HERE / "analysis_continuous_severity.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
