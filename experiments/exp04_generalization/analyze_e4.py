"""
experiments/exp04_generalization/analyze_e4.py

E4 (generalization). Pre-registration: docs/decisions.md "E4 PRE-REGISTRATION".
 A. BIPIA per attack family (code 45 / targeted 25 / task_irrelevant 25 / task_relevant 25
    unique attacks, so intervals are wide) vs the benign pools.
 B. LLMail-Inject short emails (SUPPLEMENTARY; no benign emails exist, so negatives are
    the same benign pools; separation is confounded with source/genre style), overall and
    by level group, plus the PRE-REGISTERED non-inferiority check of cont_exp vs asym.
 C. Source-style diagnostic: cross-validated TF-IDF+LR AUROC separating each attack set
    from the benign pool by lexical style alone (a trained-on-attacks upper-ish bound on
    how much of any detector's AUROC could be style).
Detectors: naive / transplant seed ensembles x {asym, cont_exp, none}; six single models
(asym, cont_exp); the two guardrails. Bootstrap over records (shared resample, threshold
re-derived), primary pool all_benign; point estimates for the other pools. Record-level
intervals only (no seed uncertainty); corpus benign by construction.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from experiments.exp03_baselines.analyze_exp03_ensemble import GUARDRAILS, guardrail_bare_scores
from experiments.exp03_baselines.run_exp03_naive import load_spans
from experiments.exp03_baselines.run_exp03_wrapped import unique_in_order
from experiments.exp06_overdefense.analyze_continuous_severity import score
from experiments.exp06_overdefense.analyze_e6 import MODELS, SEEDS, ci, load
from experiments.exp06_overdefense.analyze_e6_pooled import SAME_CHANNEL_CELLS, evaluate
from experiments.exp06_overdefense.run_e6_scoring import load_inputs

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCORERS = ["asym", "cont_exp", "none"]
ENS = ["ens_naive", "ens_transplant"]
N_BOOT = 300


def main():
    meta4 = json.loads((HERE / "e4_sets_meta.json").read_text())
    ll = [json.loads(l) for l in open(REPO / "data" / "interim" / "e4_llmail_short.jsonl") if l.strip()]
    levels = np.array([r["level"] for r in ll])
    fam = np.array(meta4["bipia_family"])
    meta, decl, control, cells, raw = load()
    for m in MODELS:
        raw[m]["llmail"] = np.load(HERE / f"e4_probs_{m}.npz")["llmail"]

    sc = {}
    for m in MODELS:
        r = raw[m]
        for s in SCORERS:
            sc[(m, s)] = {"corpus": score(r["corpus"], decl, s),
                          "notinject": score(r["notinject"], np.array(["user"] * len(r["notinject"])), s),
                          "bipia_ben": score(r["bipia_ben"], np.array(["document"] * len(r["bipia_ben"])), s),
                          "bipia_inj": score(r["bipia_inj"], np.array(["document"] * len(r["bipia_inj"])), s),
                          "llmail": score(r["llmail"], np.array(["document"] * len(r["llmail"])), s)}
    for cond in ("naive", "transplant"):
        for s in SCORERS:
            mem = [f"{cond}_s{x}" for x in SEEDS]
            sc[(f"ens_{cond}", s)] = {k: np.mean([sc[(m, s)][k] for m in mem], axis=0) for k in sc[(mem[0], s)]}
    gb, gz, gl = guardrail_bare_scores(), np.load(REPO / "experiments" / "exp06_overdefense" / "e6_guardrail_scores.npz"), \
        np.load(HERE / "e4_guardrail_scores.npz")
    for g in GUARDRAILS:
        sc[(g, "raw")] = {"corpus": gz[f"{g}__corpus"], "notinject": gz[f"{g}__notinject"],
                          "bipia_ben": gb[g][0], "bipia_inj": gb[g][1], "llmail": gl[f"{g}__llmail"]}

    units = [(e, s) for e in ENS for s in SCORERS] + [(m, s) for m in MODELS for s in ("asym", "cont_exp")] \
            + [(g, "raw") for g in GUARDRAILS]
    same = np.isin(cells, list(SAME_CHANNEL_CELLS))
    pools = {"bipia_benign": lambda d: d["bipia_ben"],
             "same_channel": lambda d: np.r_[d["bipia_ben"], d["corpus"][same]],
             "all_benign": lambda d: np.r_[d["bipia_ben"], d["corpus"], d["notinject"]]}
    pos_sets = {"BIPIA_all": ("bipia_inj", np.ones(len(fam), bool))}
    for f in sorted(set(fam)):
        pos_sets[f"BIPIA_{f}"] = ("bipia_inj", fam == f)
    pos_sets["LLMail_all"] = ("llmail", np.ones(len(levels), bool))
    for lv in "123":
        pos_sets[f"LLMail_level{lv}"] = ("llmail", levels == lv)
    pos_sets["LLMail_level4(n=18, descriptive)"] = ("llmail", levels == "4")
    d0 = sc[units[0]]
    psz = {p: len(f(d0)) for p, f in pools.items()}
    name = lambda u: f"{u[0]}|{u[1]}" if u[1] != "raw" else u[0]  # noqa: E731

    point = {(u, ps, p): evaluate(sc[u][k][mask], pools[p](sc[u])) for u in units
             for ps, (k, mask) in pos_sets.items() for p in pools}
    rng = np.random.default_rng(42)
    draws = {(u, ps): [] for u in units for ps in pos_sets}
    for _ in range(N_BOOT):
        nn = rng.integers(0, psz["all_benign"], psz["all_benign"])
        for ps, (k, mask) in pos_sets.items():
            ii = rng.integers(0, int(mask.sum()), int(mask.sum()))
            for u in units:
                draws[(u, ps)].append(evaluate(sc[u][k][mask][ii], pools["all_benign"](sc[u])[nn])[0])
    draws = {k: np.array(v) for k, v in draws.items()}

    out = {"point": {f"{name(u)}|{ps}|{p}": [round(v[0], 3), round(v[1], 3)] for (u, ps, p), v in point.items()}}
    show = [(e, s) for e in ENS for s in SCORERS] + [(g, "raw") for g in GUARDRAILS]
    for ps, (k, mask) in pos_sets.items():
        print(f"\n=== positives = {ps} (n={int(mask.sum())}) vs all_benign: AUROC [95% CI] (TPR@5% FPR) ===")
        for u in show:
            a, t = point[(u, ps, "all_benign")]
            print(f"   {name(u):<34}{a:.3f} {ci(draws[(u, ps)])}  ({t:.2f})")

    # --- pre-registered non-inferiority check on LLMail_all vs all_benign
    print("\n=== PRE-REGISTERED CHECK: cont_exp - asym, LLMail_all vs all_benign ===")
    crit = {}
    for e in ENS:
        d = draws[((e, "cont_exp"), "LLMail_all")] - draws[((e, "asym"), "LLMail_all")]
        lo, hi = ci(d)
        pt = point[((e, "cont_exp"), "LLMail_all", "all_benign")][0] - point[((e, "asym"), "LLMail_all", "all_benign")][0]
        crit[e] = {"diff": round(pt, 3), "ci95": [lo, hi], "lower_gt_-0.03": bool(lo > -0.03)}
        print(f"   {e}: diff {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]   lower bound > -0.03: {lo > -0.03}")
    singles = [point[((m, "cont_exp"), "LLMail_all", "all_benign")][0] - point[((m, "asym"), "LLMail_all", "all_benign")][0]
               for m in MODELS]
    n_ok = sum(x > -0.03 for x in singles)
    print(f"   single models with point diff > -0.03: {n_ok}/6  ({[round(x, 3) for x in singles]})")
    verdict = all(v["lower_gt_-0.03"] for v in crit.values()) and n_ok >= 4
    print(f"   VERDICT (both ensembles AND >=4/6 singles): {'CONFIRMED non-inferior' if verdict else 'NOT confirmed'}")
    out["preregistered_check"] = {"ensembles": crit, "singles_ok": n_ok, "singles_diffs": [round(x, 3) for x in singles],
                                  "confirmed": bool(verdict)}
    none_minus = {e: round(point[((e, "none"), "LLMail_all", "all_benign")][0]
                           - point[((e, "asym"), "LLMail_all", "all_benign")][0], 3) for e in ENS}
    print(f"   (expectation only) none - asym on LLMail vs all_benign: {none_minus}")

    # --- single-model BIPIA-family table, asym vs cont_exp, point AUROC vs all_benign
    print("\n=== single models, AUROC vs all_benign, asym / cont_exp ===")
    for ps in ("BIPIA_code", "BIPIA_targeted", "BIPIA_task_irrelevant", "BIPIA_task_relevant", "LLMail_all"):
        print(f"   {ps:<24}" + "  ".join(f"{m}: {point[((m, 'asym'), ps, 'all_benign')][0]:.2f}/"
                                         f"{point[((m, 'cont_exp'), ps, 'all_benign')][0]:.2f}" for m in MODELS))

    # --- C. source-style diagnostic (TF-IDF + LR, 5-fold CV AUROC)
    _, _, c_texts, n_texts, _ = load_inputs()
    benign_all, injected_all = load_spans()
    bipia_ben, bipia_inj = unique_in_order(benign_all), unique_in_order(injected_all)
    all_benign_txt = bipia_ben + c_texts + n_texts
    same_txt = bipia_ben + [t for t, s in zip(c_texts, same) if s]
    ll_txt = [r["text"] for r in ll]

    def tfidf_auc(pos, neg):
        X, y = pos + neg, np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                             LogisticRegression(max_iter=2000, class_weight="balanced"))
        p = cross_val_predict(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        return float(roc_auc_score(y, p))

    print("\n=== C. SOURCE-STYLE DIAGNOSTIC: 5-fold CV TF-IDF+LR AUROC (attacks vs benign pool, trained ON attacks) ===")
    out["tfidf_style_auc"] = {}
    for label, pos in (("BIPIA_all", bipia_inj), ("LLMail_all", ll_txt)):
        for pname, neg in (("bipia_benign", bipia_ben), ("same_channel", same_txt), ("all_benign", all_benign_txt)):
            a = tfidf_auc(pos, neg)
            out["tfidf_style_auc"][f"{label}|{pname}"] = round(a, 3)
            print(f"   {label:<12} vs {pname:<13} TF-IDF AUROC {a:.3f}   (role detector, ens_naive|asym: "
                  f"{point[(('ens_naive', 'asym'), label, pname)][0]:.3f}; ens_transplant|asym: "
                  f"{point[(('ens_transplant', 'asym'), label, pname)][0]:.3f})")

    (HERE / "analysis_e4.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
