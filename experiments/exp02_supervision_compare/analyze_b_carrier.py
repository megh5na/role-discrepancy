"""
experiments/exp02_supervision_compare/analyze_b_carrier.py

Analysis of E2b (results_b_carrier.json). First use of src/evaluation/
significance.py on real results.

What is and is not testable here:
  * Per-span correctness (bare and wrapped) was saved, so naive vs transplant
    accuracy can be tested PAIRED: both conditions were scored on the same
    spans with the same seed. Done per seed, pooled across folds.
  * Bare/wrapped LABEL AGREEMENT was saved only as a per-run fraction, not per
    span, so it cannot be bootstrapped over spans. It is tested at the level
    of runs/folds instead (sign test), which is far coarser: the 6 folds, not
    the thousands of spans, are the independent units.
Pooled span-level tests treat spans within a fold as independent and weight
folds by size, so they overstate certainty. They are reported with that
caveat, alongside the fold-level view.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from src.evaluation.significance import mcnemar_test, paired_bootstrap_ci

HERE = Path(__file__).resolve().parent
ENC = "deberta_v3_base_encoder"


def load():
    with open(HERE / "results_b_carrier.json") as f:
        return json.load(f)


def index(rows):
    """(model, fold, condition, seed) -> row"""
    return {(r["model"], r["held_out_corpus"], r["condition"], r["seed"]): r for r in rows}


def folds_of(rows):
    seen = {}
    for r in rows:
        seen[r["held_out_corpus"]] = r["true_label"]
    return seen  # fold -> label


def agreement_tests(rows, idx, fold_label):
    """Transplant minus naive agreement, per (fold, seed) pair and per fold."""
    seeds = sorted({r["seed"] for r in rows if r["model"] == ENC})
    pair_diffs = [
        idx[(ENC, f, "transplant", s)]["bare_wrapped_agreement"]
        - idx[(ENC, f, "naive", s)]["bare_wrapped_agreement"]
        for f in fold_label for s in seeds
    ]
    fold_diffs = []
    for f in fold_label:
        t = np.mean([idx[(ENC, f, "transplant", s)]["bare_wrapped_agreement"] for s in seeds])
        n = np.mean([idx[(ENC, f, "naive", s)]["bare_wrapped_agreement"] for s in seeds])
        fold_diffs.append(t - n)
    pos_pairs = sum(d > 0 for d in pair_diffs)
    pos_folds = sum(d > 0 for d in fold_diffs)
    return {
        "mean_diff_transplant_minus_naive": float(np.mean(fold_diffs)),
        "per_fold_diffs": {f: float(d) for f, d in zip(fold_label, fold_diffs)},
        "pairs_positive": f"{pos_pairs}/{len(pair_diffs)}",
        "folds_positive": f"{pos_folds}/{len(fold_diffs)}",
        # exact two-sided sign test; the fold-level one is the conservative one
        "sign_test_p_over_pairs": float(stats.binomtest(pos_pairs, len(pair_diffs), 0.5).pvalue),
        "sign_test_p_over_folds": float(stats.binomtest(pos_folds, len(fold_diffs), 0.5).pvalue),
    }


def pooled_accuracy_tests(rows, idx, fold_subset, key):
    """Per seed: pool spans over the given folds, paired transplant vs naive."""
    out = {}
    seeds = sorted({r["seed"] for r in rows if r["model"] == ENC})
    for s in seeds:
        a, b = [], []  # transplant, naive per-span 0/1
        for f in fold_subset:
            a += idx[(ENC, f, "transplant", s)][key]
            b += idx[(ENC, f, "naive", s)][key]
        ci = paired_bootstrap_ci(a, b, n_boot=2000, seed=42)
        mc = mcnemar_test([bool(x) for x in a], [bool(x) for x in b])
        out[f"seed_{s}"] = {
            "n_spans": len(a),
            "transplant_acc": float(np.mean(a)),
            "naive_acc": float(np.mean(b)),
            "diff_ci95": [round(ci["ci_low"], 4), round(ci["ci_high"], 4)],
            "diff": round(ci["point_estimate"], 4),
            "bootstrap_excludes_zero": ci["significant"],
            "mcnemar_p": mc["p_value"],
            # mcnemar_test(a=transplant, b=naive): b = a right & b wrong,
            # c = a wrong & b right
            "transplant_only_right": mc["b"],
            "naive_only_right": mc["c"],
        }
    return out


def mean_drop(rows, fold_label):
    """Bare minus wrapped accuracy, per condition, averaged over seeds then folds."""
    res = {}
    for cond in ("naive", "transplant"):
        per_fold = []
        for f in fold_label:
            rs = [r for r in rows if r["model"] == ENC and r["held_out_corpus"] == f and r["condition"] == cond]
            per_fold.append(np.mean([r["bare_accuracy"] - r["wrapped_accuracy"] for r in rs]))
        res[cond] = {"mean_drop": float(np.mean(per_fold)),
                     "per_fold": {f: float(d) for f, d in zip(fold_label, per_fold)}}
    return res


def main():
    rows = load()
    idx = index(rows)
    fold_label = folds_of(rows)
    non_reasoning = [f for f, l in fold_label.items() if l != "reasoning"]
    reasoning = [f for f, l in fold_label.items() if l == "reasoning"]

    result = {
        "folds": fold_label,
        "agreement": agreement_tests(rows, idx, fold_label),
        "accuracy_drop_when_wrapped": mean_drop(rows, fold_label),
        "wrapped_accuracy_user_document_folds": pooled_accuracy_tests(rows, idx, non_reasoning, "wrapped_correct"),
        "bare_accuracy_user_document_folds": pooled_accuracy_tests(rows, idx, non_reasoning, "bare_correct"),
        "wrapped_accuracy_reasoning_folds": pooled_accuracy_tests(rows, idx, reasoning, "wrapped_correct"),
        "bare_accuracy_reasoning_folds": pooled_accuracy_tests(rows, idx, reasoning, "bare_correct"),
    }
    with open(HERE / "analysis_b_carrier.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
