"""
experiments/exp06_overdefense/analyze_e6_pooled.py

Joint detection / over-defense view. analyze_e6.py fixes the threshold on matched
controls. Here the question is separability: BIPIA injected strings (120, declared
DOCUMENT) versus three different pools of BENIGN traffic, scored by the same
scorer, with the threshold recalibrated on each pool (so this measures how well
the score ORDERS attacks above benign traffic, not a deployment threshold):

  bipia_benign   549 benign BIPIA strings (the pool E3 used)
  same_channel   benign traffic in the attacks' own channel: BIPIA benign +
                 control_document + the 3 document_* mixed cells (legitimate
                 documents that quote requests, directives or worked solutions)
  all_benign     every benign record: BIPIA benign + all 18 corpus cells + NotInject

Metrics: AUROC and TPR at 5% FPR; bootstrap over records (threshold re-derived,
shared resample, paired across scorers/models). Same caveats as analyze_e6.py:
corpus is benign by construction, record-level intervals only, no seed uncertainty
in them. Scorers: asym (shipped) and none (severity-free).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from experiments.exp06_overdefense.analyze_e6 import MODELS, SCORERS, ci, load, scores_for
from src.scoring.calibration import calibrate_threshold

HERE = Path(__file__).resolve().parent
UNITS = MODELS + ["ens_naive", "ens_transplant"]
SAME_CHANNEL_CELLS = {"control_document", "document_quotes_request", "document_quotes_directive",
                      "document_contains_worked_solution"}


def pools(sc, cells):
    """pool name -> benign score array (for one model/scorer) and its record count."""
    same = np.isin(cells, list(SAME_CHANNEL_CELLS))
    return {
        "bipia_benign": lambda s: s["bipia_ben"],
        "same_channel": lambda s: np.r_[s["bipia_ben"], s["corpus"][same]],
        "all_benign": lambda s: np.r_[s["bipia_ben"], s["corpus"], s["notinject"]],
    }


def evaluate(pos, neg):
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    return float(roc_auc_score(y, np.r_[neg, pos])), float(np.mean(pos >= calibrate_threshold(neg, 0.05)))


def main():
    meta, decl, control, cells, raw = load()
    sc = scores_for(raw, decl)
    pool_fns = pools(sc, cells)
    # pool sizes
    s0 = sc[(MODELS[0], "asym")]
    print({k: len(f(s0)) for k, f in pool_fns.items()}, "positives:", len(s0["bipia_inj"]))
    rng = np.random.default_rng(42)
    out = {"point": {}, "ci": {}}
    keys = [(u, s, p) for u in UNITS for s in SCORERS for p in pool_fns]
    for u, s, p in keys:
        sd = sc[(u, s)]
        auc, tpr = evaluate(sd["bipia_inj"], pool_fns[p](sd))
        out["point"][f"{u}|{s}|{p}"] = {"auroc": round(auc, 3), "tpr5": round(tpr, 3)}
    draws = {k: {"auroc": [], "tpr5": []} for k in keys}
    n_pos = len(s0["bipia_inj"])
    sizes = {p: len(f(s0)) for p, f in pool_fns.items()}
    for _ in range(500):
        ii = rng.integers(0, n_pos, n_pos)
        nn = {p: rng.integers(0, n, n) for p, n in sizes.items()}
        for u, s, p in keys:
            sd = sc[(u, s)]
            a, t = evaluate(sd["bipia_inj"][ii], pool_fns[p](sd)[nn[p]])
            draws[(u, s, p)]["auroc"].append(a)
            draws[(u, s, p)]["tpr5"].append(t)
    for pname in pool_fns:
        print(f"\n=== negatives = {pname}: AUROC (TPR@5% FPR) ===")
        print(f"{'unit':<17}{'asym AUROC':<22}{'none AUROC':<22}{'asym TPR@5%':<13}{'none TPR@5%':<13}{'AUROC none - asym [CI]'}")
        for u in UNITS:
            a, n = out["point"][f"{u}|asym|{pname}"], out["point"][f"{u}|none|{pname}"]
            da = np.array(draws[(u, "asym", pname)]["auroc"]); dn = np.array(draws[(u, "none", pname)]["auroc"])
            d = dn - da
            out["ci"][f"{u}|{pname}"] = {"asym_auroc_ci": ci(da), "none_auroc_ci": ci(dn),
                                         "none_minus_asym": round(n["auroc"] - a["auroc"], 3), "diff_ci": ci(d)}
            print(f"{u:<17}{a['auroc']:.3f} {str(ci(da)):<14}{n['auroc']:.3f} {str(ci(dn)):<14}{a['tpr5']:<13.3f}{n['tpr5']:<13.3f}"
                  f"{n['auroc'] - a['auroc']:+.3f} {ci(d)}")
    (HERE / "analysis_e6_pooled.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
