from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.exp03_baselines.analyze_exp03_ensemble import GUARDRAILS, guardrail_bare_scores
from experiments.exp06_overdefense.analyze_continuous_severity import score
from experiments.exp06_overdefense.analyze_e6 import MODELS, SEEDS, ci, load
from experiments.exp06_overdefense.analyze_e6_pooled import SAME_CHANNEL_CELLS, evaluate
from src.scoring.calibration import calibrate_threshold

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
N_BOOT = 500


def ecdf(ref_sorted, x):
    return (np.searchsorted(ref_sorted, x, side="left") + np.searchsorted(ref_sorted, x, side="right")) / (2 * len(ref_sorted))


def main():
    meta4 = json.loads((HERE / "e4_sets_meta.json").read_text())
    ll = [json.loads(l) for l in open(REPO / "data" / "interim" / "e4_llmail_short.jsonl") if l.strip()]
    levels, fam = np.array([r["level"] for r in ll]), np.array(meta4["bipia_family"])
    meta, decl, control, cells, raw = load()
    for m in MODELS:
        raw[m]["llmail"] = np.load(HERE / f"e4_probs_{m}.npz")["llmail"]

    def ours(cond, scorer="cont_exp"):
        d = {}
        for key, decl_arr in (("corpus", decl), ("notinject", None), ("bipia_ben", "document"), ("bipia_inj", "document"),
                              ("llmail", "document")):
            per = []
            for x in SEEDS:
                r = raw[f"{cond}_s{x}"][key]
                da = decl_arr if key == "corpus" else np.array([("user" if key == "notinject" else decl_arr)] * len(r))
                per.append(score(r, da, scorer))
            d[key] = np.mean(per, axis=0)
        return d

    gb, gz, gl = guardrail_bare_scores(), np.load(REPO / "experiments" / "exp06_overdefense" / "e6_guardrail_scores.npz"), \
        np.load(HERE / "e4_guardrail_scores.npz")
    det = {"N": ours("naive"), "T": ours("transplant")}
    for tag, g in (("P", "protectai_v2"), ("G", "third_party_deberta_v3_injection")):
        det[tag] = {"corpus": gz[f"{g}__corpus"], "notinject": gz[f"{g}__notinject"], "bipia_ben": gb[g][0],
                    "bipia_inj": gb[g][1], "llmail": gl[f"{g}__llmail"]}

    same = np.isin(cells, list(SAME_CHANNEL_CELLS))
    pool_fn = {"all_benign": lambda d: np.r_[d["bipia_ben"], d["corpus"], d["notinject"]],
               "bipia_benign": lambda d: d["bipia_ben"]}
    ben = {p: {k: f(det[k]) for k in det} for p, f in pool_fn.items()}
    pos_sets = {"BIPIA_all": ("bipia_inj", np.ones(len(fam), bool))}
    for f in sorted(set(fam)):
        pos_sets[f"BIPIA_{f}"] = ("bipia_inj", fam == f)
    pos_sets["LLMail_all"] = ("llmail", np.ones(len(levels), bool))
    for lv in "123":
        pos_sets[f"LLMail_level{lv}"] = ("llmail", levels == lv)

    units = {"N": ["N"], "T": ["T"], "P": ["P"], "G": ["G"],
             "N+P max": ["N", "P"], "N+P mean": ["N", "P"], "T+P max": ["T", "P"], "T+P mean": ["T", "P"],
             "N+G max": ["N", "G"], "N+P+G max": ["N", "P", "G"], "N+P+G mean": ["N", "P", "G"]}
    names = list(units)

    def combine(name, us):
        return np.maximum.reduce(us) if name.endswith("max") else (np.mean(us, axis=0) if name.endswith("mean") else us[0])

    rng = np.random.default_rng(42)
    res = {(p, ps, u): {"auroc": [], "tpr": []} for p in pool_fn for ps in pos_sets for u in names}
    venn = {(ps, k): [] for ps in ("BIPIA_all", "LLMail_all") + tuple(f"BIPIA_{f}" for f in sorted(set(fam)))
            for k in ("ours_only", "guard_only", "both", "neither")}
    for it in range(N_BOOT):
        for p in pool_fn:
            n_p = len(ben[p]["N"])
            perm = rng.permutation(n_p)
            cal, ev = perm[: n_p // 2], perm[n_p // 2:]
            ev = ev[rng.integers(0, len(ev), len(ev))]
            refs = {k: np.sort(ben[p][k][cal]) for k in det}
            u_ben = {k: ecdf(refs[k], ben[p][k][ev]) for k in det}
            u_posfull = {k: {"bipia_inj": ecdf(refs[k], det[k]["bipia_inj"]), "llmail": ecdf(refs[k], det[k]["llmail"])}
                         for k in det}
            for ps, (key, mask) in pos_sets.items():
                idx = np.where(mask)[0]
                idx = idx[rng.integers(0, len(idx), len(idx))]
                up = {k: u_posfull[k][key][idx] for k in det}
                for u in names:
                    a, t = evaluate(combine(u, [up[c] for c in units[u]]), combine(u, [u_ben[c] for c in units[u]]))
                    res[(p, ps, u)]["auroc"].append(a)
                    res[(p, ps, u)]["tpr"].append(t)
                if p == "all_benign" and (ps, "both") in venn:
                    thr = {c: calibrate_threshold(u_ben[c], 0.05) for c in ("N", "P")}
                    fN, fP = up["N"] >= thr["N"], up["P"] >= thr["P"]
                    for k, v in (("ours_only", fN & ~fP), ("guard_only", ~fN & fP), ("both", fN & fP), ("neither", ~fN & ~fP)):
                        venn[(ps, k)].append(float(v.mean()))
    res = {k: {m: np.array(v) for m, v in d.items()} for k, d in res.items()}
    out = {"results": {}, "venn": {}, "criterion": {}}
    for p in pool_fn:
        for ps in pos_sets:
            for u in names:
                out["results"][f"{p}|{ps}|{u}"] = {"auroc": round(float(res[(p, ps, u)]["auroc"].mean()), 3),
                                                   "auroc_ci95": ci(res[(p, ps, u)]["auroc"]),
                                                   "tpr5": round(float(res[(p, ps, u)]["tpr"].mean()), 3)}
    show = ["N", "P", "N+P max", "N+P mean", "T", "T+P max", "G", "N+G max", "N+P+G max", "N+P+G mean"]
    for p in pool_fn:
        print(f"\n######## pool = {p} : AUROC [95% CI] (TPR@5% FPR); mean over {N_BOOT} resamples ########")
        for ps in pos_sets:
            print(f"\n  positives = {ps}")
            for u in show:
                r = out["results"][f"{p}|{ps}|{u}"]
                print(f"     {u:<12}{r['auroc']:.3f} {r['auroc_ci95']}  ({r['tpr5']:.2f})")

    print("\n=== PRE-REGISTERED CRITERION (primary pair N+P, max rule, pool all_benign) ===")
    ok_all = True
    for ps in ("BIPIA_all", "LLMail_all"):
        aN, aP = res[("all_benign", ps, "N")]["auroc"], res[("all_benign", ps, "P")]["auroc"]
        best = "N" if aN.mean() >= aP.mean() else "P"
        diff = res[("all_benign", ps, "N+P max")]["auroc"] - res[("all_benign", ps, best)]["auroc"]
        lo, hi = ci(diff)
        passed = lo > 0
        ok_all &= passed
        out["criterion"][ps] = {"better_component": best, "diff": round(float(diff.mean()), 3), "ci95": [lo, hi], "pass": bool(passed)}
        print(f"   {ps:<12} better single = {best}; fused(max) minus it = {diff.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  -> {'PASS' if passed else 'fail'}")
    print(f"   VERDICT: {'COMPLEMENTARY (criterion met on both sets)' if ok_all else 'NOT complementary beyond the stronger detector (criterion not met on both sets)'}")
    out["criterion"]["verdict_complementary"] = bool(ok_all)

    print("\n=== who catches what (each detector at its own 5% FPR on the all_benign evaluation half; fractions of positives) ===")
    for ps in ("BIPIA_all", "LLMail_all") + tuple(f"BIPIA_{f}" for f in sorted(set(fam))):
        v = {k: float(np.mean(venn[(ps, k)])) for k in ("ours_only", "guard_only", "both", "neither")}
        out["venn"][ps] = {k: round(x, 3) for k, x in v.items()}
        print(f"   {ps:<22}ours-only {v['ours_only']:.2f}  ProtectAI-only {v['guard_only']:.2f}  both {v['both']:.2f}  neither {v['neither']:.2f}")
    (HERE / "analysis_fusion.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
