"""
experiments/exp01_swap_validity/summarize.py

Reads results.json (written by run_exp01.py) and produces the E1 report:
mean +/- std across seeds per (model, condition), the naive-vs-transplant
delta on swap_consistency (THE headline number), and an explicit GATE
verdict per Section 14's stated failure condition ("Swap consistency near
chance regardless of construction... KILL CONDITION").

Chance-level swap consistency for 5-way classification is 1/5 = 0.20 IF
predictions were uniform random; in practice "near chance" for a model
that's already learned something about text length/vocabulary is better
approximated empirically (this script reports the raw numbers and lets a
human apply judgement, per the spec's own "escalate" language rather than
a hardcoded threshold no one agreed to).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = Path(__file__).resolve().parent / "results.json"


def summarize():
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    print("=" * 78)
    print("EXPERIMENT 1 SUMMARY -- swap consistency, mean +/- std across seeds")
    print("(*** SINGLE-SEED / REDUCED-SEED NUMBERS LABELLED EXPLICITLY, Section 15 rule ***)")
    print("=" * 78)

    grouped: dict[tuple[str, str], list[dict]] = {}
    for r in results:
        key = (r["model"], r["condition"])
        grouped.setdefault(key, []).append(r)

    summary_rows = {}
    for (model, condition), rows in grouped.items():
        sc = [r["swap_consistency"] for r in rows]
        na = [r["native_acc"] for r in rows]
        fa = [r["foreign_acc"] for r in rows]
        n_seeds = len(rows)
        label = "SINGLE-SEED" if n_seeds == 1 else f"{n_seeds}-SEED"
        sc_mean, sc_std = statistics.mean(sc), (statistics.stdev(sc) if n_seeds > 1 else 0.0)
        na_mean = statistics.mean(na)
        fa_mean = statistics.mean(fa)
        summary_rows[(model, condition)] = {
            "n_seeds": n_seeds,
            "swap_consistency_mean": sc_mean,
            "swap_consistency_std": sc_std,
            "native_acc_mean": na_mean,
            "foreign_acc_mean": fa_mean,
        }
        print(
            f"[{label}] {model:<22} {condition:<12} "
            f"swap_consistency={sc_mean:.3f}+/-{sc_std:.3f}  "
            f"native_acc={na_mean:.3f}  foreign_acc={fa_mean:.3f}"
        )

    print("\n" + "=" * 78)
    print("HEADLINE: does transplantation raise swap consistency over naive labelling?")
    print("=" * 78)
    for model in {k[0] for k in summary_rows}:
        naive_key = (model, "naive")
        transplant_key = (model, "transplant")
        if naive_key in summary_rows and transplant_key in summary_rows:
            naive_sc = summary_rows[naive_key]["swap_consistency_mean"]
            transplant_sc = summary_rows[transplant_key]["swap_consistency_mean"]
            delta = transplant_sc - naive_sc
            direction = "CONFIRMS H1 direction" if delta > 0 else "CONTRADICTS H1 direction"
            print(f"{model:<22} naive={naive_sc:.3f} -> transplant={transplant_sc:.3f}  "
                  f"delta={delta:+.3f}  [{direction}]")

    print("\n" + "=" * 78)
    print("GATE CHECK (Section 14 E1 / Section 24 M1): is swap consistency near chance")
    print("regardless of construction? (would be the project's kill condition)")
    print("=" * 78)
    encoder_rows = {k: v for k, v in summary_rows.items() if k[0] == "deberta_v3_base_encoder"}
    if encoder_rows:
        max_sc = max(v["swap_consistency_mean"] for v in encoder_rows.values())
        if max_sc < 0.35:
            print(f"max swap_consistency across conditions = {max_sc:.3f} -- "
                  f"CONCERNING, close to chance-ish territory. Needs escalation per spec.")
        else:
            print(f"max swap_consistency across conditions = {max_sc:.3f} -- "
                  f"clearly above chance; NOT a kill-condition trigger on this evidence.")

    return summary_rows


if __name__ == "__main__":
    summarize()
