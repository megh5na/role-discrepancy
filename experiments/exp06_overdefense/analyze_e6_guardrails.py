from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.exp03_baselines.analyze_exp03_ensemble import GUARDRAILS, guardrail_bare_scores
from experiments.exp06_overdefense.analyze_e6 import MODELS, ci, load, metrics, scores_for
from experiments.exp06_overdefense.analyze_e6_pooled import SAME_CHANNEL_CELLS, evaluate
from src.scoring.calibration import calibrate_threshold

HERE = Path(__file__).resolve().parent
OURS = [("ens_naive", "asym"), ("ens_transplant", "asym"), ("ens_naive", "none"), ("ens_transplant", "none")] \
       + [(m, "asym") for m in MODELS]


def name(u):
    return f"{u[0]}|{u[1]}" if u[1] != "raw" else u[0]


def main():
    meta, decl, control, cells, raw = load()
    sc = dict(scores_for(raw, decl))
    gb = guardrail_bare_scores()
    z = np.load(HERE / "e6_guardrail_scores.npz")
    for g in GUARDRAILS:
        sc[(g, "raw")] = {"corpus": z[f"{g}__corpus"], "notinject": z[f"{g}__notinject"],
                          "bipia_ben": gb[g][0], "bipia_inj": gb[g][1]}
    units = [(g, "raw") for g in GUARDRAILS] + OURS
    ctrl_ids, mixed_ids = np.where(control)[0], np.where(~control)[0]
    s0 = sc[units[0]]
    n_i, n_b, n_n = len(s0["bipia_inj"]), len(s0["bipia_ben"]), len(s0["notinject"])
    same = np.isin(cells, list(SAME_CHANNEL_CELLS))
    pools = {"bipia_benign": lambda s: s["bipia_ben"],
             "same_channel": lambda s: np.r_[s["bipia_ben"], s["corpus"][same]],
             "all_benign": lambda s: np.r_[s["bipia_ben"], s["corpus"], s["notinject"]]}
    psz = {p: len(f(s0)) for p, f in pools.items()}

    ident = (np.arange(len(ctrl_ids)), np.arange(len(mixed_ids)), np.arange(n_i), np.arange(n_b), np.arange(n_n))
    pt = {u: metrics(sc[u], decl, ctrl_ids, mixed_ids, *ident) for u in units}
    sep = {(u, p): evaluate(sc[u]["bipia_inj"], f(sc[u])) for u in units for p, f in pools.items()}

    rng = np.random.default_rng(42)
    dm = {u: {} for u in units}
    ds = {(u, p): ([], []) for u in units for p in pools}
    for _ in range(500):
        ii_ = rng.integers(0, n_i, n_i)
        idx = (rng.integers(0, len(ctrl_ids), len(ctrl_ids)), rng.integers(0, len(mixed_ids), len(mixed_ids)),
               ii_, rng.integers(0, n_b, n_b), rng.integers(0, n_n, n_n))
        nn = {p: rng.integers(0, n, n) for p, n in psz.items()}
        for u in units:
            for k, v in metrics(sc[u], decl, ctrl_ids, mixed_ids, *idx).items():
                dm[u].setdefault(k, []).append(v)
            for p, f in pools.items():
                a, t = evaluate(sc[u]["bipia_inj"][ii_], f(sc[u])[nn[p]])
                ds[(u, p)][0].append(a); ds[(u, p)][1].append(t)
    dm = {u: {k: np.array(v) for k, v in d.items()} for u, d in dm.items()}
    ds = {k: (np.array(a), np.array(t)) for k, (a, t) in ds.items()}

    out = {"operating_point": {}, "separability": {}, "paired_vs_guardrails": {}}
    print("A. DEPLOYMENT-LIKE OPERATING POINT (one threshold per detector = 5% of pooled control cells)")
    print(f"{'detector':<34}{'ctrl FPR':<10}{'mixed FPR':<18}{'NotInject FPR':<17}{'BIPIA benign FPR':<18}{'BIPIA TPR'}")
    for u in units:
        p = pt[u]
        out["operating_point"][name(u)] = {k: round(float(p[k]), 3) for k in
                                           ("ctrl_fpr", "mixed_fpr", "notinject_fpr", "bipia_benign_fpr", "bipia_tpr")}
        print(f"{name(u):<34}{p['ctrl_fpr']:<10.3f}{p['mixed_fpr']:.3f} {str(ci(dm[u]['mixed_fpr'])):<11}"
              f"{p['notinject_fpr']:.3f} {str(ci(dm[u]['notinject_fpr'])):<10}{p['bipia_benign_fpr']:<18.3f}"
              f"{p['bipia_tpr']:.3f} {ci(dm[u]['bipia_tpr'])}")

    print("\nB. SEPARABILITY: BIPIA injected vs benign pool, AUROC (TPR@5% FPR), 95% CI over records")
    for pname in pools:
        print(f"\n  pool = {pname} ({psz[pname]} benign)")
        for u in units:
            a, t = sep[(u, pname)]
            out["separability"][f"{name(u)}|{pname}"] = {"auroc": round(a, 3), "tpr5": round(t, 3),
                                                         "auroc_ci95": ci(ds[(u, pname)][0])}
            print(f"    {name(u):<34}AUROC {a:.3f} {ci(ds[(u, pname)][0])}   TPR@5% {t:.3f}")
        for e in (("ens_naive", "asym"), ("ens_transplant", "asym")):
            for g in GUARDRAILS:
                gu = (g, "raw")
                d = ds[(e, pname)][0] - ds[(gu, pname)][0]
                dt = ds[(e, pname)][1] - ds[(gu, pname)][1]
                rec = {"auroc_diff": round(sep[(e, pname)][0] - sep[(gu, pname)][0], 3), "auroc_ci95": ci(d),
                       "tpr_diff": round(sep[(e, pname)][1] - sep[(gu, pname)][1], 3), "tpr_ci95": ci(dt)}
                out["paired_vs_guardrails"][f"{name(e)}_minus_{g}|{pname}"] = rec
                print(f"    {name(e)} minus {g}: AUROC {rec['auroc_diff']:+.3f} {rec['auroc_ci95']}   "
                      f"TPR@5% {rec['tpr_diff']:+.3f} {rec['tpr_ci95']}")

    print("\nC. which cells do the guardrails flag (control-calibrated thresholds)?")
    for g in GUARDRAILS:
        thr = pt[(g, "raw")]["thr"]
        flagged = sc[(g, "raw")]["corpus"] >= thr
        rows = sorted(((c, float(flagged[cells == c].mean())) for c in dict.fromkeys(cells.tolist())), key=lambda x: -x[1])
        print(f"  {g} (threshold {thr:.4f}): " + ", ".join(f"{c} {v:.2f}" for c, v in rows[:5]))
        out.setdefault("guardrail_top_cells", {})[g] = rows[:5]
    (HERE / "analysis_e6_guardrails.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
