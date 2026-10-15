"""
experiments/exp06_overdefense/analyze_e6.py

Experiment 6: over-defense. How often does a discrepancy detector flag
LEGITIMATE role-mixed traffic, and does the asymmetric severity matrix do what
it was designed to do (spare benign mismatches such as a user pasting a
document) compared with the severity-free scorer?

Data: benign role-mixed corpus (13 mixed cells + 5 matched control cells, 1,710
texts; configs/data/benign_role_mixed.yaml), NotInject (339 benign prompts with
trigger words, declared USER), BIPIA (549 benign / 120 injected unique strings,
declared DOCUMENT). Six models (naive/transplant x seeds 42, 123, 456) from saved
role probabilities, plus seed ensembles (mean score over the 3 seeds).

Scorers (src/scoring, project functions, declared role per record):
  asym  prob_deficit x asymmetric severity   (the shipped scorer)
  none  prob_deficit alone                    (severity-free)

Operating point (deployment-like, ONE threshold per scorer and model): the
threshold is set so that 5% of the pooled matched CONTROL cells are flagged.
Then reported at that same threshold: false-positive rate on the mixed cells
(overall, by declared channel, by cell, by perceived->declared pair), the
over-defense inflation over controls, NotInject FPR, BIPIA benign FPR and BIPIA
TPR. Secondary: per-channel thresholds (5% on each channel's own controls).

Intervals: bootstrap over RECORDS (controls, mixed, BIPIA strings, NotInject
rows), threshold re-derived in every resample, one resample shared by all
models/scorers (paired). They cover sampling of records only, NOT training-seed
uncertainty (3 seeds per condition; seed-level evidence = exact permutation
test, minimum p 0.05 one-sided / 0.10 two-sided). The corpus is benign by
construction and its register labels are inherited, so these are rates on
THIS constructed corpus, not on real traffic.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.ablation_severity.analyze_severity_ablation import score_variant
from experiments.exp03_baselines.analyze_exp03_seeds_perm import permutation_test
from src.evaluation.over_defense import fpr_by_channel_pair
from src.ingestion.schema import ROLE_ORDER, Role
from src.scoring.calibration import calibrate_threshold
from src.scoring.discrepancy import perceived_role_argmax

HERE = Path(__file__).resolve().parent
ABL = HERE.parents[1] / "experiments" / "ablation_severity"
SEEDS = (42, 123, 456)
MODELS = [f"{c}_s{s}" for c in ("naive", "transplant") for s in SEEDS]
SCORERS = {"asym": "prob_deficit|asym", "none": "prob_deficit|none"}
CHANNELS = ["user", "document", "tool", "reasoning", "system"]
FPR = 0.05
N_BOOT = 1000


def score_decl(probs: np.ndarray, declared: np.ndarray, variant: str) -> np.ndarray:
    out = np.empty(len(probs))
    for ch in set(declared.tolist()):
        m = declared == ch
        out[m] = score_variant(probs[m], variant, declared=Role(ch))
    return out


def load():
    meta = json.loads((HERE / "e6_texts_meta.json").read_text())
    decl = np.array(meta["corpus_declared"])
    control = np.array(meta["corpus_control"])
    cells = np.array(meta["corpus_cells"])
    raw = {}
    for m in MODELS:
        z = np.load(HERE / f"e6_probs_{m}.npz")
        b = np.load(ABL / f"probs_{m}.npz")
        raw[m] = {"corpus": z["corpus"], "notinject": z["notinject"],
                  "bipia_ben": b["bare__benign"], "bipia_inj": b["bare__injected"]}
    return meta, decl, control, cells, raw


def scores_for(raw, decl):
    sc = {}
    for m in MODELS:
        for sname, variant in SCORERS.items():
            r = raw[m]
            sc[(m, sname)] = {
                "corpus": score_decl(r["corpus"], decl, variant),
                "notinject": score_decl(r["notinject"], np.array(["user"] * len(r["notinject"])), variant),
                "bipia_ben": score_decl(r["bipia_ben"], np.array(["document"] * len(r["bipia_ben"])), variant),
                "bipia_inj": score_decl(r["bipia_inj"], np.array(["document"] * len(r["bipia_inj"])), variant),
            }
    for cond in ("naive", "transplant"):
        members = [f"{cond}_s{s}" for s in SEEDS]
        for sname in SCORERS:
            sc[(f"ens_{cond}", sname)] = {k: np.mean([sc[(m, sname)][k] for m in members], axis=0)
                                          for k in sc[(members[0], sname)]}
    return sc


def metrics(s, decl, ctrl_ids, mixed_ids, ci, mi, ii, bi, ni):
    """All metrics for one (model, scorer) on one resample."""
    ctrl_rows, mixed_rows = ctrl_ids[ci], mixed_ids[mi]
    ctrl, mixed = s["corpus"][ctrl_rows], s["corpus"][mixed_rows]
    ctrl_decl, mixed_decl = decl[ctrl_rows], decl[mixed_rows]
    thr = calibrate_threshold(ctrl, FPR)
    out = {"thr": thr, "ctrl_fpr": float(np.mean(ctrl >= thr)), "mixed_fpr": float(np.mean(mixed >= thr)),
           "bipia_tpr": float(np.mean(s["bipia_inj"][ii] >= thr)),
           "bipia_benign_fpr": float(np.mean(s["bipia_ben"][bi] >= thr)),
           "notinject_fpr": float(np.mean(s["notinject"][ni] >= thr))}
    out["inflation"] = out["mixed_fpr"] - out["ctrl_fpr"]
    pc_flags = np.zeros(len(mixed), dtype=bool)
    for ch in CHANNELS:
        mm = mixed_decl == ch
        out[f"mixed_fpr_{ch}"] = float(np.mean(mixed[mm] >= thr)) if mm.any() else float("nan")
        cm = ctrl_decl == ch
        if cm.any():
            pc_flags[mm] = mixed[mm] >= calibrate_threshold(ctrl[cm], FPR)
    out["mixed_fpr_perchannel_thr"] = float(np.mean(pc_flags))
    return out


def ci(x):
    lo, hi = np.nanquantile(x, [0.025, 0.975])
    return [round(float(lo), 3), round(float(hi), 3)]


def main():
    meta, decl, control, cells, raw = load()
    sc = scores_for(raw, decl)
    ctrl_ids, mixed_ids = np.where(control)[0], np.where(~control)[0]
    n_i, n_b, n_n = len(raw[MODELS[0]]["bipia_inj"]), len(raw[MODELS[0]]["bipia_ben"]), len(raw[MODELS[0]]["notinject"])
    keys = list(sc)
    print(f"corpus: {len(decl)} texts, {len(ctrl_ids)} control, {len(mixed_ids)} mixed; NotInject {n_n}; "
          f"BIPIA {n_i} injected / {n_b} benign")

    ident = (np.arange(len(ctrl_ids)), np.arange(len(mixed_ids)), np.arange(n_i), np.arange(n_b), np.arange(n_n))
    point = {k: metrics(sc[k], decl, ctrl_ids, mixed_ids, *ident) for k in keys}

    rng = np.random.default_rng(42)
    draws = {k: {} for k in keys}
    for _ in range(N_BOOT):
        idx = (rng.integers(0, len(ctrl_ids), len(ctrl_ids)), rng.integers(0, len(mixed_ids), len(mixed_ids)),
               rng.integers(0, n_i, n_i), rng.integers(0, n_b, n_b), rng.integers(0, n_n, n_n))
        for k in keys:
            for name, v in metrics(sc[k], decl, ctrl_ids, mixed_ids, *idx).items():
                draws[k].setdefault(name, []).append(v)
    draws = {k: {n: np.array(v) for n, v in d.items()} for k, d in draws.items()}

    out = {"point": {f"{m}|{s}": v for (m, s), v in point.items()}}

    # ---- 1. per model, global threshold from pooled controls
    for sname in SCORERS:
        print(f"\n=== scorer {sname}: one threshold per model, 5% of pooled control cells flagged ===")
        print(f"{'model':<17}{'thr':<8}{'ctrl FPR':<10}{'mixed FPR':<12}{'inflation':<11}{'NotInject FPR':<15}{'BIPIA benign FPR':<18}{'BIPIA TPR'}")
        for m in MODELS + ["ens_naive", "ens_transplant"]:
            p = point[(m, sname)]
            print(f"{m:<17}{p['thr']:<8.3f}{p['ctrl_fpr']:<10.3f}{p['mixed_fpr']:<12.3f}{p['inflation']:<+11.3f}"
                  f"{p['notinject_fpr']:<15.3f}{p['bipia_benign_fpr']:<18.3f}{p['bipia_tpr']:.3f}")

    # ---- 2. asym vs none per model (paired) and seed-averaged
    print("\n=== asym minus none (negative = asymmetric flags LESS), paired over records; global threshold ===")
    out["asym_minus_none"] = {}
    for m in MODELS + ["ens_naive", "ens_transplant"]:
        row = {}
        for met in ("mixed_fpr", "notinject_fpr", "bipia_tpr", "bipia_benign_fpr"):
            d = draws[(m, "asym")][met] - draws[(m, "none")][met]
            row[met] = {"diff": round(point[(m, "asym")][met] - point[(m, "none")][met], 3), "ci95": ci(d)}
        out["asym_minus_none"][m] = row
        print(f"  {m:<17}" + "  ".join(f"{k} {v['diff']:+.3f} {v['ci95']}" for k, v in row.items()))
    for cond in ("naive", "transplant"):
        members = [f"{cond}_s{s}" for s in SEEDS]
        row = {}
        for met in ("mixed_fpr", "notinject_fpr", "bipia_tpr"):
            d = (np.mean([draws[(m, "asym")][met] for m in members], axis=0)
                 - np.mean([draws[(m, "none")][met] for m in members], axis=0))
            pt = np.mean([point[(m, "asym")][met] - point[(m, "none")][met] for m in members])
            row[met] = {"diff": round(float(pt), 3), "ci95": ci(d)}
        out["asym_minus_none"][f"{cond}_seed_avg"] = row
        print(f"  {cond+' (3-seed mean)':<17}" + "  ".join(f"{k} {v['diff']:+.3f} {v['ci95']}" for k, v in row.items()))

    # ---- 3. by declared channel and by cell (mean over the six single models)
    print("\n=== mixed-cell FPR by declared channel (mean over 6 models, global threshold) ===")
    out["by_channel"] = {}
    for ch in CHANNELS:
        a = np.nanmean([point[(m, "asym")][f"mixed_fpr_{ch}"] for m in MODELS])
        n = np.nanmean([point[(m, "none")][f"mixed_fpr_{ch}"] for m in MODELS])
        out["by_channel"][ch] = {"asym": round(float(a), 3), "none": round(float(n), 3)}
        print(f"  declared {ch:<10} asym {a:.3f}   none {n:.3f}")

    print("\n=== by cell (mean over 6 models, global threshold; dominant perceived role over models x spans) ===")
    print(f"{'cell':<36}{'declared':<10}{'content':<10}{'asym FPR':<10}{'none FPR':<10}{'top perceived roles (asym uses argmax)'}")
    out["by_cell"] = {}
    for cell in dict.fromkeys(cells.tolist()):
        rows = np.where(cells == cell)[0]
        fl = {}
        for sname in SCORERS:
            vals = [float(np.mean(sc[(m, sname)]["corpus"][rows] >= point[(m, sname)]["thr"])) for m in MODELS]
            fl[sname] = float(np.mean(vals))
        perc = Counter(ROLE_ORDER[int(np.argmax(raw[m]["corpus"][r]))].value for m in MODELS for r in rows)
        tot = sum(perc.values())
        top = ", ".join(f"{k} {v / tot:.0%}" for k, v in perc.most_common(3))
        is_ctrl = bool(control[rows[0]])
        out["by_cell"][cell] = {"asym": round(fl["asym"], 3), "none": round(fl["none"], 3), "control": is_ctrl, "perceived": top}
        print(f"{cell:<36}{decl[rows[0]]:<10}{meta['corpus_content_register'][rows[0]]:<10}"
              f"{fl['asym']:<10.3f}{fl['none']:<10.3f}{top}")

    # ---- 4. perceived->declared pairs on mixed cells, pooled over the six models
    print("\n=== FPR by (perceived -> declared) pair on MIXED cells, pooled over 6 models (global thresholds) ===")
    out["by_pair"] = {}
    for sname in SCORERS:
        per, dec, flg = [], [], []
        for m in MODELS:
            t = point[(m, sname)]["thr"]
            for r in mixed_ids:
                per.append(ROLE_ORDER[int(np.argmax(raw[m]["corpus"][r]))])
                dec.append(Role(decl[r]))
                flg.append(bool(sc[(m, sname)]["corpus"][r] >= t))
        table = fpr_by_channel_pair(per, dec, flg)
        out["by_pair"][sname] = table
        print(f"  scorer {sname}:")
        for pair, v in sorted(table.items(), key=lambda kv: -kv[1]["n"])[:10]:
            print(f"    {pair:<24} n={v['n']:<5} FPR {v['fpr']:.3f}")

    # ---- 5. naive vs transplant at seed level (3 vs 3)
    print("\n=== naive minus transplant, seed-level exact permutation (3 vs 3; min p 0.05 / 0.10) ===")
    out["permutation"] = {}
    is_naive = {m: m.startswith("naive") for m in MODELS}
    for sname in SCORERS:
        for met in ("mixed_fpr", "inflation", "notinject_fpr", "bipia_tpr"):
            vals = {m: point[(m, sname)][met] for m in MODELS}
            r = permutation_test(vals, is_naive)
            out["permutation"][f"{sname}|{met}"] = r
            fmt = lambda c: " ".join(f"{vals[f'{c}_s{s}']:.2f}" for s in SEEDS)  # noqa: E731
            print(f"  {sname:<5}{met:<14} naive {fmt('naive')}  transplant {fmt('transplant')}  "
                  f"diff {r['observed_naive_minus_transplant']:+.3f}  p1={r['p_one_sided_naive_greater']} p2={r['p_two_sided']}")

    # ---- 6. secondary: per-channel thresholds
    print("\n=== secondary: mixed FPR with PER-CHANNEL thresholds (5% on each channel's own controls) ===")
    for sname in SCORERS:
        vals = {m: point[(m, sname)]["mixed_fpr_perchannel_thr"] for m in MODELS}
        print(f"  {sname:<5}" + "  ".join(f"{m} {v:.3f}" for m, v in vals.items()))
    out["perchannel"] = {s: {m: round(point[(m, s)]["mixed_fpr_perchannel_thr"], 3) for m in MODELS} for s in SCORERS}

    with open(HERE / "analysis_e6.json", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print("\n(Intervals: bootstrap over records only; training-seed uncertainty is NOT included. "
          "Corpus is benign by construction with inherited, noisy register labels.)")


if __name__ == "__main__":
    main()
