from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from experiments.exp03_baselines.analyze_exp03_seeds import load_models
from experiments.exp03_baselines.run_exp03_naive import load_spans
from experiments.exp03_baselines.run_exp03_wrapped import bootstrap, ci, one_pass, summarize, unique_in_order
from src.scoring.calibration import calibrate_threshold

HERE = Path(__file__).resolve().parent
F = 0.05
GUARDRAILS = ("protectai_v2", "third_party_deberta_v3_injection")


def make_ensemble(models: dict, members: list[str]) -> dict:
    conds = list(models[members[0]])
    return {c: (np.mean([models[m][c][0] for m in members], axis=0),
                np.mean([models[m][c][1] for m in members], axis=0)) for c in conds}


def make_rank_ensemble(models: dict, members: list[str]) -> dict:
    """Sensitivity variant: average each model's within-condition RANKS (over
    benign + injected pooled) instead of raw scores, so models with different
    score scales contribute equally."""
    from scipy.stats import rankdata

    out = {}
    for c in models[members[0]]:
        nb = len(models[members[0]][c][0])
        ranks = []
        for m in members:
            pooled = np.r_[models[m][c][0], models[m][c][1]]
            ranks.append(rankdata(pooled) / len(pooled))
        avg = np.mean(ranks, axis=0)
        out[c] = (avg[:nb], avg[nb:])
    return out


def auroc(b, i):
    return float(roc_auc_score(np.r_[np.zeros(len(b)), np.ones(len(i))], np.r_[b, i]))


def guardrail_bare_scores():
    """Guardrail scores for the unique strings, aligned to unique_in_order."""
    benign_all, injected_all = load_spans()
    npz = np.load(HERE / "scores_exp03_naive.npz")
    out = {}
    for g in GUARDRAILS:
        res = []
        for texts, kind in ((benign_all, "benign"), (injected_all, "injected")):
            first = {}
            for idx, t in enumerate(texts):
                first.setdefault(t, idx)
            order = unique_in_order(texts)
            res.append(npz[f"{g}__{kind}"][[first[t] for t in order]])
        out[g] = tuple(res)
    return out


def bare_bootstrap(bare: dict, n_boot=2000, seed=42):
    """bare[name] = (benign, injected). Shared resample; AUROC and TPR at
    1% / 5% FPR with the threshold recomputed from the resampled benign."""
    rng = np.random.default_rng(seed)
    names = list(bare)
    n_b, n_i = len(bare[names[0]][0]), len(bare[names[0]][1])

    def one(bi, ii):
        r = {}
        for n, (b, i) in bare.items():
            b, i = b[bi], i[ii]
            r[(n, "auroc")] = auroc(b, i)
            for f in (0.01, 0.05):
                r[(n, f"tpr_{f}")] = float(np.mean(i >= calibrate_threshold(b, f)))
        return r

    pt = one(np.arange(n_b), np.arange(n_i))
    acc = {k: [] for k in pt}
    for _ in range(n_boot):
        for k, v in one(rng.integers(0, n_b, n_b), rng.integers(0, n_i, n_i)).items():
            acc[k].append(v)
    return pt, {k: np.array(v) for k, v in acc.items()}


def main():
    models = load_models()
    naive = sorted(m for m in models if m.startswith("naive"))
    transplant = sorted(m for m in models if m.startswith("transplant"))
    ens = {"ours_naive": make_ensemble(models, naive), "ours_transplant": make_ensemble(models, transplant)}
    cond_names = list(ens["ours_naive"])
    out = {"members": {"naive": naive, "transplant": transplant}}

    # 1. wrapped evaluation of the two ensembles
    point, draws = bootstrap(ens, cond_names, n_boot=1000, seed=42)
    res = summarize(point, draws)
    out["wrapped"] = res
    print("ENSEMBLES (mean discrepancy score over 3 seeds), 5% FPR; point [95% CI over unique strings]")
    for unit in ("bare", "known_avg", "novel_avg"):
        print(f"\n[{unit}]")
        for d in ("ours_naive", "ours_transplant"):
            r = res["detectors"][d][unit]["0.05"]
            print(f"  {d:<16} AUROC {r['B_auroc']['point']:.3f} {r['B_auroc']['ci95']}  "
                  f"TPR(recal) {r['B_tpr']['point']:.3f}  benign FPR at bare thr {r['A_fpr_on_wrapped_benign']['point']:.3f}")
        p = res["paired_transplant_minus_naive"][unit]["0.05"]["B_auroc"]
        print(f"  transplant - naive AUROC: {p['point']:+.3f} {p['ci95']}{'  *' if p['ci_excludes_zero'] else ''}")

    print("\nDegradation when wrapped (bare minus wrapped), ensembles:")
    out["did"] = {}
    for metric, label, sign in (("B_auroc", "AUROC drop", 1), ("B_tpr", "TPR drop", 1),
                                ("A_fpr_on_wrapped_benign", "FPR inflation", -1)):
        for unit in ("known_avg", "novel_avg"):
            dr = {d: sign * (draws[(d, "bare", F, metric)] - draws[(d, unit, F, metric)]) for d in ens}
            pt = {d: sign * (point[(d, "bare", F, metric)] - point[(d, unit, F, metric)]) for d in ens}
            did = dr["ours_naive"] - dr["ours_transplant"]
            lo, hi = ci(did)
            out["did"].setdefault(label, {})[unit] = {
                "naive": round(float(pt["ours_naive"]), 3), "transplant": round(float(pt["ours_transplant"]), 3),
                "did": round(float(pt["ours_naive"] - pt["ours_transplant"]), 3), "ci95": [lo, hi],
                "ci_excludes_zero": bool(lo > 0 or hi < 0)}
            r = out["did"][label][unit]
            print(f"  {label:<14}{unit:<10}naive {r['naive']:+.3f}  transplant {r['transplant']:+.3f}  "
                  f"DiD {r['did']:+.3f} [{lo:+.3f}, {hi:+.3f}]{' *' if r['ci_excludes_zero'] else ''}")

    # 2. bare detection vs guardrails
    bare = {n: ens[n]["bare"] for n in ens}
    bare.update(guardrail_bare_scores())
    bpt, bdr = bare_bootstrap(bare)
    print("\nBARE detection vs guardrails (unique strings)")
    print(f"{'detector':<36}{'AUROC':<22}{'TPR@1%':<22}{'TPR@5%'}")
    out["bare_vs_guardrails"] = {"detectors": {}, "paired": {}}
    for n in bare:
        row = {k: {"point": round(bpt[(n, k)], 3), "ci95": ci(bdr[(n, k)])} for k in ("auroc", "tpr_0.01", "tpr_0.05")}
        out["bare_vs_guardrails"]["detectors"][n] = row
        print(f"{n:<36}{row['auroc']['point']:.3f} {str(row['auroc']['ci95']):<14}"
              f"{row['tpr_0.01']['point']:.3f} {str(row['tpr_0.01']['ci95']):<14}"
              f"{row['tpr_0.05']['point']:.3f} {row['tpr_0.05']['ci95']}")
    print("\nPaired differences (ensemble minus guardrail), same resampled strings:")
    for e in ("ours_naive", "ours_transplant"):
        for g in GUARDRAILS:
            rec = {}
            for k in ("auroc", "tpr_0.01", "tpr_0.05"):
                d = bdr[(e, k)] - bdr[(g, k)]
                lo, hi = ci(d)
                rec[k] = {"diff": round(bpt[(e, k)] - bpt[(g, k)], 3), "ci95": [lo, hi],
                          "ci_excludes_zero": bool(lo > 0 or hi < 0)}
            out["bare_vs_guardrails"]["paired"][f"{e}_vs_{g}"] = rec
            print(f"  {e:<16} vs {g:<34} " + "  ".join(
                f"{k} {v['diff']:+.3f} {v['ci95']}{'*' if v['ci_excludes_zero'] else ''}" for k, v in rec.items()))

    # 3. sensitivity: leave-one-seed-out ensembles (size 2)
    print("\nLeave-one-seed-out ensembles (size 2), point AUROC at 5% FPR: bare / known / novel")
    nb, ni = len(ens["ours_naive"]["bare"][0]), len(ens["ours_naive"]["bare"][1])
    out["leave_one_out"] = {}
    for cond, members in (("naive", naive), ("transplant", transplant)):
        for left_out in members:
            sub = [m for m in members if m != left_out]
            e2 = {f"loo_{cond}": make_ensemble(models, sub)}
            p = one_pass(e2, cond_names, np.arange(nb), np.arange(ni))
            a = [p[(f"loo_{cond}", u, F, "B_auroc")] for u in ("bare", "known_avg", "novel_avg")]
            out["leave_one_out"][f"{cond}_without_{left_out.split('_')[1]}"] = [round(float(x), 3) for x in a]
            print(f"  {cond:<11} without {left_out.split('_')[1]:<5} -> {a[0]:.3f} / {a[1]:.3f} / {a[2]:.3f}")

    # 4. ensemble vs its members
    print("\nMember AUROC (bare) vs ensemble:")
    for cond, members in (("naive", naive), ("transplant", transplant)):
        m_pts = [float(one_pass({m: models[m]}, cond_names, np.arange(nb), np.arange(ni))[(m, "bare", F, "B_auroc")])
                 for m in members]
        e_pt = point[(f"ours_{cond}", "bare", F, "B_auroc")]
        print(f"  {cond:<11} members {[round(x, 3) for x in m_pts]}  mean {np.mean(m_pts):.3f}  ensemble {e_pt:.3f}")
        out.setdefault("members_vs_ensemble", {})[cond] = {"members": [round(x, 3) for x in m_pts],
                                                          "ensemble": round(float(e_pt), 3)}

    # 5. sensitivity: rank-averaged instead of score-averaged ensembles
    print("\nSensitivity: RANK-averaged ensembles, point AUROC at 5% FPR: bare / known / novel")
    out["rank_ensemble"] = {}
    for cond, members in (("naive", naive), ("transplant", transplant)):
        e = {f"rank_{cond}": make_rank_ensemble(models, members)}
        p = one_pass(e, cond_names, np.arange(nb), np.arange(ni))
        a = [p[(f"rank_{cond}", u, F, "B_auroc")] for u in ("bare", "known_avg", "novel_avg")]
        sc = [point[(f"ours_{cond}", u, F, "B_auroc")] for u in ("bare", "known_avg", "novel_avg")]
        out["rank_ensemble"][cond] = {"rank_avg": [round(float(x), 3) for x in a],
                                      "score_avg": [round(float(x), 3) for x in sc]}
        print(f"  {cond:<11} rank-avg {a[0]:.3f} / {a[1]:.3f} / {a[2]:.3f}    score-avg {sc[0]:.3f} / {sc[1]:.3f} / {sc[2]:.3f}")

    with open(HERE / "analysis_exp03_ensemble.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\n(* = 95% CI over unique strings excludes zero. Ensemble intervals reflect string sampling only.)")


if __name__ == "__main__":
    main()
