from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
from pathlib import Path

import numpy as np

from experiments.exp01_swap_validity.run_exp01 import load_jsonl
from experiments.exp03_baselines.analyze_exp03_seeds_perm import permutation_test
from src.ingestion.schema import ROLE_TO_IDX
from src.models.perceived_role.checkpoint import load_checkpoint
from src.models.perceived_role.encoder import load_tokenizer
from src.models.perceived_role.inference import make_predict_fn
from src.supervision.types import SwapPairItem

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CK = REPO_ROOT / "checkpoints"
MODELS = {
    "naive_s42": CK / "perceived_role_v1_naive",
    "transplant_s42": CK / "perceived_role_v1",
    "naive_s123": CK / "kept_seed_models" / "perceived_role_v1_naive_seed123",
    "transplant_s123": CK / "kept_seed_models" / "perceived_role_v1_seed123",
    "naive_s456": CK / "kept_seed_models" / "perceived_role_v1_naive_seed456",
    "transplant_s456": CK / "kept_seed_models" / "perceived_role_v1_seed456",
}
SEEDS = (42, 123, 456)
N_BOOT = 2000
MAX_LENGTH = 32  # as in the original E1 and in training


def decide(diffs: list[float]) -> str:
    mean = float(np.mean(diffs))
    n_pos = sum(d > 0 for d in diffs)
    if n_pos == len(diffs) and mean >= 0.10:
        return "REPLICATED"
    if n_pos < 2 or mean < 0.05:
        return "NOT REPLICATED"
    return "INCONCLUSIVE"


def main():
    pairs = load_jsonl(REPO_ROOT / "data" / "processed" / "swap_test_pairs.jsonl", SwapPairItem)
    labels = np.array([ROLE_TO_IDX[p.label] for p in pairs])
    native = [p.native_text for p in pairs]
    foreign = [p.foreign_text for p in pairs]
    print(f"{len(pairs)} swap pairs", flush=True)

    per = {}
    preds = {}
    for name, ckpt in MODELS.items():
        model, meta = load_checkpoint(ckpt)
        tok = load_tokenizer(meta["backbone_name"])
        fn = make_predict_fn(model, tok, "cpu", max_length=MAX_LENGTH, batch_size=32)
        a = np.array([ROLE_TO_IDX[r] for r in fn(native)])
        b = np.array([ROLE_TO_IDX[r] for r in fn(foreign)])
        preds[f"{name}__native"], preds[f"{name}__foreign"] = a, b
        per[name] = {"swap_consistency": float(np.mean(a == b)), "native_acc": float(np.mean(a == labels)),
                     "foreign_acc": float(np.mean(b == labels))}
        print(f"  {name:16s} consistency {per[name]['swap_consistency']:.3f}  native {per[name]['native_acc']:.3f}  "
              f"foreign {per[name]['foreign_acc']:.3f}", flush=True)
        del model
    np.savez(HERE / "preds_kept_seeds.npz", labels=labels, **preds)

    cons = {m: per[m]["swap_consistency"] for m in per}
    diffs = {s: cons[f"transplant_s{s}"] - cons[f"naive_s{s}"] for s in SEEDS}
    mean_diff = float(np.mean(list(diffs.values())))
    # permutation_test returns (first-group mean - second-group mean); flag TRANSPLANT as the first group
    perm = permutation_test(cons, {m: m.startswith("transplant") for m in cons})
    perm["note"] = "'observed_naive_minus_transplant' here is transplant minus naive (groups flipped); p_one_sided = P(transplant-naive >= observed)"

    rng = np.random.default_rng(0)
    cons_arr = {m: (preds[f"{m}__native"] == preds[f"{m}__foreign"]).astype(float) for m in per}
    idx = rng.integers(0, len(pairs), size=(N_BOOT, len(pairs)))
    seed_boot = {s: (cons_arr[f"transplant_s{s}"][idx].mean(1) - cons_arr[f"naive_s{s}"][idx].mean(1)) for s in SEEDS}
    mean_boot = np.mean([seed_boot[s] for s in SEEDS], axis=0)
    ci = lambda x: [round(float(np.percentile(x, 2.5)), 3), round(float(np.percentile(x, 97.5)), 3)]  # noqa: E731

    out = {"per_model": {m: {k: round(v, 4) for k, v in d.items()} for m, d in per.items()},
           "seedwise_diff_transplant_minus_naive": {str(s): round(d, 4) for s, d in diffs.items()},
           "mean_diff": round(mean_diff, 4),
           "mean_by_condition": {c: round(float(np.mean([cons[f"{c}_s{s}"] for s in SEEDS])), 4) for c in ("naive", "transplant")},
           "permutation_seed_level": perm,
           "bootstrap_pairs_ci95_diff": {**{str(s): ci(seed_boot[s]) for s in SEEDS}, "mean_over_seeds": ci(mean_boot)},
           "verdict_preregistered": decide(list(diffs.values())),
           "original_e1_for_reference": {"naive_mean": 0.711, "transplant_mean": 0.937, "n_seeds": 2, "n_train_per_condition": 1275},
           "n_pairs": len(pairs), "max_length": MAX_LENGTH}
    (HERE / "results_kept_seeds.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "per_model"}, indent=1))


if __name__ == "__main__":
    main()
