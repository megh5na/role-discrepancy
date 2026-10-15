"""
experiments/ablation_severity/analyze_severity_ablation.py

Severity ablation (spec Section 14: "symmetric vs asymmetric scoring", and
"discrepancy formulation variants"). The final score is
    score = base_discrepancy(probs, declared)  x  severity(perceived, declared).
asymmetric severity (src/scoring/asymmetry.py) is the second claimed
contribution and had never been ablated: every earlier run used it.

All variants are computed from the SAME saved role probabilities
(run_severity_ablation.py), calling the project's own scoring functions, so
only the scoring rule differs.

  formulation  prob_deficit | neg_log_prob | margin   (src/scoring/discrepancy.py)
  severity     asym   channel-pair severity matrix (the contribution)
               sym    all mismatches 1.0, match 0.0 (SYMMETRIC_SEVERITY)
               none   severity fixed at 1.0 (base discrepancy alone)
  cmd_prob     EXPLORATORY simple baseline P(user) + P(system): ignores the
               declared role and the severity matrix entirely.

Declared role is DOCUMENT (BIPIA channel mapping). Evaluation as in
run_exp03_wrapped.py: 12 conditions, 5% FPR, bootstrap over UNIQUE strings
with the threshold re-derived inside every resample and one shared resample
(paired). Reference variant for paired differences: prob_deficit|asym (the
default scorer). SINGLE-SEED (seed 42) ablation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.exp03_baselines.run_exp03_wrapped import bootstrap, ci
from src.ingestion.schema import ROLE_TO_IDX, Role
from src.scoring.asymmetry import severity
from src.scoring.discrepancy import base_discrepancy, perceived_role_argmax

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DECLARED = Role.DOCUMENT
FORMS = ("prob_deficit", "neg_log_prob", "margin")
SEVS = ("asym", "sym", "none")
VARIANTS = [f"{f}|{s}" for f in FORMS for s in SEVS] + ["cmd_prob"]
REFERENCE = "prob_deficit|asym"
F = 0.05
CKPTS = ("naive_s42", "transplant_s42")


def score_variant(probs: np.ndarray, variant: str, declared: Role = DECLARED) -> np.ndarray:
    """Per-span score for one variant, from (n, 5) role probabilities."""
    if variant == "cmd_prob":
        return probs[:, ROLE_TO_IDX[Role.USER]] + probs[:, ROLE_TO_IDX[Role.SYSTEM]]
    form, sev = variant.split("|")
    out = np.empty(len(probs))
    for k, p in enumerate(probs):
        base = base_discrepancy(p, declared, formulation=form)
        if sev == "none":
            s = 1.0
        else:
            s = severity(perceived_role_argmax(p), declared, use_asymmetry=(sev == "asym"))
        out[k] = base * s
    return out


def load_probs(name: str) -> dict:
    z = np.load(HERE / f"probs_{name}.npz")
    out: dict = {}
    for key in z.files:
        cond, kind = key.split("__")
        out.setdefault(cond, [None, None])[0 if kind == "benign" else 1] = z[key]
    return {c: tuple(v) for c, v in out.items()}


def build_scores(probs_by_ckpt: dict) -> dict:
    """{f"{ckpt}:{variant}": {cond: (benign_scores, injected_scores)}}"""
    scores = {}
    for ck, per_cond in probs_by_ckpt.items():
        for v in VARIANTS:
            scores[f"{ck}:{v}"] = {c: (score_variant(b, v), score_variant(i, v)) for c, (b, i) in per_cond.items()}
    return scores


def validate_against_saved(probs_by_ckpt: dict) -> float:
    """The default variant must reproduce the scores saved by the earlier E3
    runs (same checkpoints, same inputs). Returns the max abs difference."""
    saved = np.load(REPO_ROOT / "experiments" / "exp03_baselines" / "scores_exp03_wrapped.npz")
    worst = 0.0
    for ck, det in (("naive_s42", "ours_naive"), ("transplant_s42", "ours_transplant")):
        for c, (b, i) in probs_by_ckpt[ck].items():
            for kind, arr in (("benign", b), ("injected", i)):
                worst = max(worst, float(np.max(np.abs(score_variant(arr, REFERENCE) - saved[f"{det}__{c}__{kind}"]))))
    return worst


def main():
    probs = {ck: load_probs(ck) for ck in CKPTS}
    worst = validate_against_saved(probs)
    print(f"validation: max |default-variant score - saved E3-wrapped score| = {worst:.2e}")
    assert worst < 1e-4, "saved probabilities do not reproduce the earlier scores"

    scores = build_scores(probs)
    cond_names = list(next(iter(scores.values())))
    print(f"{len(scores)} scorer variants x {len(cond_names)} conditions; bootstrapping ...", flush=True)
    point, draws = bootstrap(scores, cond_names, n_boot=500, seed=42)

    out = {"validation_max_abs_diff": worst, "variants": VARIANTS, "reference": REFERENCE, "results": {}}
    for ck in CKPTS:
        print(f"\n=== {ck}  (5% FPR; AUROC recalibrated; point [95% CI over unique strings]) ===")
        print(f"{'variant':<22}{'bare AUROC':<22}{'known AUROC':<14}{'novel AUROC':<14}"
              f"{'TPR@5% bare':<13}{'benFPR known/novel (bare thr)'}")
        ref = f"{ck}:{REFERENCE}"
        for v in VARIANTS:
            d = f"{ck}:{v}"
            rec = {}
            for unit in ("bare", "known_avg", "novel_avg"):
                rec[unit] = {"auroc": round(point[(d, unit, F, "B_auroc")], 3),
                             "auroc_ci95": ci(draws[(d, unit, F, "B_auroc")]),
                             "tpr": round(point[(d, unit, F, "B_tpr")], 3),
                             "benign_fpr_fixed_thr": round(point[(d, unit, F, "A_fpr_on_wrapped_benign")], 3)}
                diff = draws[(d, unit, F, "B_auroc")] - draws[(ref, unit, F, "B_auroc")]
                lo, hi = ci(diff)
                rec[unit]["auroc_diff_vs_reference"] = {
                    "diff": round(point[(d, unit, F, "B_auroc")] - point[(ref, unit, F, "B_auroc")], 3),
                    "ci95": [lo, hi], "ci_excludes_zero": bool(lo > 0 or hi < 0)}
            out["results"].setdefault(ck, {})[v] = rec
            print(f"{v:<22}{rec['bare']['auroc']:.3f} {str(rec['bare']['auroc_ci95']):<14}"
                  f"{rec['known_avg']['auroc']:<14.3f}{rec['novel_avg']['auroc']:<14.3f}"
                  f"{rec['bare']['tpr']:<13.3f}{rec['known_avg']['benign_fpr_fixed_thr']:.3f} / "
                  f"{rec['novel_avg']['benign_fpr_fixed_thr']:.3f}")

        print(f"\n  paired AUROC difference vs {REFERENCE} (variant minus reference), bare / known / novel:")
        for v in VARIANTS:
            if v == REFERENCE:
                continue
            r = out["results"][ck][v]
            cells = []
            for unit in ("bare", "known_avg", "novel_avg"):
                x = r[unit]["auroc_diff_vs_reference"]
                cells.append(f"{x['diff']:+.3f} {x['ci95']}{'*' if x['ci_excludes_zero'] else ''}")
            print(f"  {v:<22}" + "   ".join(cells))

    with open(HERE / "analysis_severity_ablation.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\n(* = 95% CI over unique strings excludes zero. Single seed (42): not a seed-level result.)")


if __name__ == "__main__":
    main()
