from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.exp03_baselines.run_exp03_wrapped import bootstrap, ci

HERE = Path(__file__).resolve().parent
F = 0.05
UNITS = ("bare", "known_avg", "novel_avg")
# (label, metric key, sign): drop = sign * (bare - unit)
QUANTITIES = (
    ("AUROC drop", "B_auroc", +1),
    ("TPR drop", "B_tpr", +1),
    ("FPR inflation", "A_fpr_on_wrapped_benign", -1),
)


def _arrays_from(npz, prefix=""):
    out: dict = {}
    for key in npz.files:
        parts = key.split("__")
        cond, kind = parts[-2], parts[-1]
        out.setdefault(cond, [None, None])[0 if kind == "benign" else 1] = npz[key]
    return {c: tuple(v) for c, v in out.items()}


def load_models():
    """{"naive_s42": {cond: (benign, injected)}, ...}"""
    models = {}
    base = np.load(HERE / "scores_exp03_wrapped.npz")
    for condition, det in (("naive", "ours_naive"), ("transplant", "ours_transplant")):
        sub = {k: base[k] for k in base.files if k.startswith(det + "__")}
        out: dict = {}
        for key, arr in sub.items():
            _, cond, kind = key.split("__")
            out.setdefault(cond, [None, None])[0 if kind == "benign" else 1] = arr
        models[f"{condition}_s42"] = {c: tuple(v) for c, v in out.items()}
    for path in sorted(HERE.glob("scores_exp03_wrapped_seed*_*.npz")):
        stem = path.stem.replace("scores_exp03_wrapped_seed", "")
        seed, condition = stem.split("_", 1)
        models[f"{condition}_s{seed}"] = _arrays_from(np.load(path))
    return models


def by_condition(models):
    out = {"naive": [], "transplant": []}
    for name in sorted(models):
        out[name.split("_")[0]].append(name)
    return out


def main():
    models = load_models()
    groups = by_condition(models)
    cond_names = list(next(iter(models.values())))
    print("models:", {k: [m.split('_')[1] for m in v] for k, v in groups.items()})

    point, draws = bootstrap(models, cond_names, n_boot=1000, seed=42)

    def q(model, unit, metric, sign, src):
        return sign * (src[(model, "bare", F, metric)] - src[(model, unit, F, metric)])

    out = {"models": {k: v for k, v in groups.items()}, "per_model": {}, "pairings": {}, "seed_averaged": {}}

    # 1. per-model absolute levels
    print(f"\n{'model':<16}{'bare AUROC':<12}{'known AUROC':<13}{'novel AUROC':<13}"
          f"{'benFPR bare/known/novel (fixed thr)'}")
    for m in sorted(models):
        a = [point[(m, u, F, "B_auroc")] for u in UNITS]
        fp = [point[(m, u, F, "A_fpr_on_wrapped_benign")] for u in UNITS]
        out["per_model"][m] = {"auroc": dict(zip(UNITS, map(float, a))),
                               "fixed_threshold_benign_fpr": dict(zip(UNITS, map(float, fp)))}
        print(f"{m:<16}{a[0]:<12.3f}{a[1]:<13.3f}{a[2]:<13.3f}{fp[0]:.3f} / {fp[1]:.3f} / {fp[2]:.3f}")

    # 2. seed spread per condition
    print("\nSeed-to-seed spread of absolute AUROC (mean +/- sd, [min, max]):")
    for cond, names in groups.items():
        for u in UNITS:
            v = np.array([point[(m, u, F, "B_auroc")] for m in names])
            print(f"  {cond:<11}{u:<10}{v.mean():.3f} +/- {v.std(ddof=1) if len(v) > 1 else float('nan'):.3f}  "
                  f"[{v.min():.3f}, {v.max():.3f}]  (n={len(v)})")

    # 3. DiD for every (naive seed, transplant seed) pairing
    print("\nDiD = naive drop - transplant drop (positive: naive degrades more), "
          "over every (naive seed, transplant seed) pairing:")
    for label, metric, sign in QUANTITIES:
        for unit in ("known_avg", "novel_avg"):
            vals = [q(n, unit, metric, sign, point) - q(t, unit, metric, sign, point)
                    for n in groups["naive"] for t in groups["transplant"]]
            same_seed = [q(n, unit, metric, sign, point) - q(t, unit, metric, sign, point)
                         for n in groups["naive"] for t in groups["transplant"]
                         if n.split("_")[1] == t.split("_")[1]]
            pos = sum(v > 0 for v in vals)
            out["pairings"].setdefault(label, {})[unit] = {
                "all_pairings": [round(float(v), 3) for v in vals],
                "same_seed_pairings": [round(float(v), 3) for v in same_seed],
                "positive": f"{pos}/{len(vals)}"}
            print(f"  {label:<14}{unit:<10}min {min(vals):+.3f}  mean {np.mean(vals):+.3f}  "
                  f"max {max(vals):+.3f}   positive in {pos}/{len(vals)} pairings")

    # 4. seed-averaged models with string-bootstrap CIs
    print("\nSeed-averaged (mean over seeds within each resample), 95% CI over unique strings:")
    for label, metric, sign in QUANTITIES:
        for unit in ("known_avg", "novel_avg"):
            drop = {c: np.mean([q(m, unit, metric, sign, draws) for m in names], axis=0)
                    for c, names in groups.items()}
            drop_pt = {c: np.mean([q(m, unit, metric, sign, point) for m in names]) for c, names in groups.items()}
            did = drop["naive"] - drop["transplant"]
            lo, hi = ci(did)
            out["seed_averaged"].setdefault(label, {})[unit] = {
                "naive_drop": {"point": round(float(drop_pt["naive"]), 3), "ci95": ci(drop["naive"])},
                "transplant_drop": {"point": round(float(drop_pt["transplant"]), 3), "ci95": ci(drop["transplant"])},
                "did": {"point": round(float(drop_pt["naive"] - drop_pt["transplant"]), 3),
                        "ci95": [lo, hi], "ci_excludes_zero": bool(lo > 0 or hi < 0)}}
            r = out["seed_averaged"][label][unit]
            print(f"  {label:<14}{unit:<10}naive {r['naive_drop']['point']:+.3f} {r['naive_drop']['ci95']}  "
                  f"transplant {r['transplant_drop']['point']:+.3f} {r['transplant_drop']['ci95']}  "
                  f"DiD {r['did']['point']:+.3f} [{lo:+.3f}, {hi:+.3f}]{' *' if r['did']['ci_excludes_zero'] else ''}")

    # 5. bare AUROC gap, per seed-averaged condition
    bare = {c: np.mean([draws[(m, "bare", F, "B_auroc")] for m in names], axis=0) for c, names in groups.items()}
    gap = bare["transplant"] - bare["naive"]
    out["seed_averaged"]["bare_auroc_transplant_minus_naive"] = {
        "point": round(float(np.mean([point[(m, "bare", F, "B_auroc")] for m in groups["transplant"]])
                             - np.mean([point[(m, "bare", F, "B_auroc")] for m in groups["naive"]])), 3),
        "ci95": ci(gap)}
    print(f"\nBare AUROC, transplant minus naive (seed-averaged): "
          f"{out['seed_averaged']['bare_auroc_transplant_minus_naive']['point']:+.3f} {ci(gap)}")

    with open(HERE / "analysis_exp03_seeds.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\n(* = 95% CI over strings excludes zero. Seed uncertainty is NOT in those intervals; "
          "see the per-pairing agreement and the per-seed spread.)")


if __name__ == "__main__":
    main()
