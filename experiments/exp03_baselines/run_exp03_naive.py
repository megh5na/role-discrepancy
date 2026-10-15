"""
experiments/exp03_baselines/run_exp03_naive.py

E3-naive: does the transplant construction cause the detection result, or
would a naive-trained encoder, scored by the SAME discrepancy scorer, do as
well? The original E3 only ever scored the transplant checkpoint, so that
claim was untested.

Scores the transplant checkpoint (perceived_role_v1) and the naive
checkpoint (perceived_role_v1_naive; train_naive_checkpoint.py) on the SAME
BIPIA spans E3 used (first 800 benign / 800 injected after delimiter
stripping), through the SAME scorer (src/scoring/asymmetry.combined_score,
declared role DOCUMENT, asymmetric severity). Optionally re-scores the
guardrails so every detector has saved per-span scores and can be compared
in a paired way.

Outputs (this directory):
  results_exp03_naive.json   matched-FPR TPRs, AUROC, paired tests, role breakdown
  scores_exp03_naive.npz     raw per-span scores for every detector
The transplant row is a reproducibility check against results.json from the
first E3 (0.170 @1% FPR, 0.547 @5% FPR).
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from src.evaluation.matched_fpr import matched_fpr_comparison
from src.evaluation.significance import mcnemar_test, paired_bootstrap_ci
from src.ingestion.io_utils import read_records
from src.ingestion.schema import Role
from src.preprocessing.delimiters import strip_delimiters
from src.scoring.asymmetry import combined_score

REPO_ROOT = Path(__file__).resolve().parents[2]
BIPIA_PATH = REPO_ROOT / "data" / "interim" / "bipia.jsonl"
OUT_DIR = Path(__file__).resolve().parent
CHECKPOINTS = {
    "ours_transplant": REPO_ROOT / "checkpoints" / "perceived_role_v1",
    "ours_naive": REPO_ROOT / "checkpoints" / "perceived_role_v1_naive",
}
N_PER_CLASS = 800  # same cap as the first E3
FPRS = [0.01, 0.05]


def load_spans():
    records = list(read_records(BIPIA_PATH))
    benign = [strip_delimiters(r.text) for r in records if r.is_injected is False][:N_PER_CLASS]
    injected = [strip_delimiters(r.text) for r in records if r.is_injected is True][:N_PER_CLASS]
    return benign, injected


def score_checkpoint(ckpt_dir: Path, texts: list[str]):
    """Discrepancy scores (and perceived-role argmax) from one checkpoint.
    Same batching/max_length (16 / 64) as the first E3's get_our_scores, but
    the checkpoint is loaded once instead of once per call."""
    import torch

    from src.models.perceived_role.checkpoint import load_checkpoint
    from src.models.perceived_role.encoder import load_tokenizer

    model, meta = load_checkpoint(ckpt_dir)
    tok = load_tokenizer(meta["backbone_name"])
    scores, perceived = [], []
    for i in range(0, len(texts), 16):
        enc = tok(texts[i:i + 16], return_tensors="pt", padding=True, truncation=True, max_length=64)
        with torch.no_grad():
            probs = torch.softmax(model(enc["input_ids"], enc["attention_mask"]), dim=-1).numpy()
        for p in probs:
            r = combined_score(p, Role.DOCUMENT, use_asymmetry=True, severity_mode="argmax")
            scores.append(r["score"])
            perceived.append(r["perceived_role"].value)
    return np.array(scores), perceived


def paired_flag_test(benign_a, inj_a, benign_b, inj_b, fpr: float) -> dict:
    """Paired comparison of two detectors at the same FPR budget: each is
    thresholded on ITS OWN benign scores, then flags on the same injected
    spans are compared (McNemar + paired bootstrap on the flag indicators)."""
    from src.scoring.calibration import calibrate_threshold

    fa = (np.asarray(inj_a) >= calibrate_threshold(benign_a, fpr)).astype(int)
    fb = (np.asarray(inj_b) >= calibrate_threshold(benign_b, fpr)).astype(int)
    ci = paired_bootstrap_ci(fa, fb, n_boot=2000, seed=42)
    mc = mcnemar_test([bool(x) for x in fa], [bool(x) for x in fb])
    return {
        "tpr_a": float(fa.mean()), "tpr_b": float(fb.mean()),
        "diff_a_minus_b": round(ci["point_estimate"], 4),
        "diff_ci95": [round(ci["ci_low"], 4), round(ci["ci_high"], 4)],
        "mcnemar_p": mc["p_value"],
        "a_only_flagged": mc["b"],  # mcnemar_test: b = a right & b wrong
        "b_only_flagged": mc["c"],
    }


def auroc(benign, injected) -> float:
    y = np.r_[np.zeros(len(benign)), np.ones(len(injected))]
    return float(roc_auc_score(y, np.r_[benign, injected]))


def auroc_diff_ci(ben_a, inj_a, ben_b, inj_b, n_boot=1000, seed=42) -> dict:
    """Bootstrap CI for AUROC(a) - AUROC(b), resampling benign and injected
    spans (the same resampled spans are used for both detectors)."""
    rng = np.random.default_rng(seed)
    ben_a, inj_a, ben_b, inj_b = map(np.asarray, (ben_a, inj_a, ben_b, inj_b))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        bi = rng.integers(0, len(ben_a), len(ben_a))
        ii = rng.integers(0, len(inj_a), len(inj_a))
        diffs[i] = auroc(ben_a[bi], inj_a[ii]) - auroc(ben_b[bi], inj_b[ii])
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"diff": float(auroc(ben_a, inj_a) - auroc(ben_b, inj_b)),
            "ci95": [round(float(lo), 4), round(float(hi), 4)]}


def run(with_guardrails: bool = True):
    benign, injected = load_spans()
    print(f"BIPIA: {len(benign)} benign, {len(injected)} injected spans", flush=True)

    scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    perceived: dict[str, dict] = {}
    for name, ckpt in CHECKPOINTS.items():
        if not (ckpt / "model.pt").exists():
            print(f"SKIPPED {name}: no checkpoint at {ckpt}", flush=True)
            continue
        print(f"Scoring {name} ...", flush=True)
        b, b_role = score_checkpoint(ckpt, benign)
        i, i_role = score_checkpoint(ckpt, injected)
        scores[name] = (b, i)
        perceived[name] = {"benign": dict(Counter(b_role)), "injected": dict(Counter(i_role))}

    if with_guardrails:
        from src.baselines.guardrails.harness import KNOWN_GUARDRAILS, GuardrailBaseline

        for key, spec in KNOWN_GUARDRAILS.items():
            try:
                fn = GuardrailBaseline(spec).make_score_fn()
            except Exception as e:  # gated / not cached / etc: reported, never silent
                print(f"SKIPPED {spec.name}: {type(e).__name__}: {str(e)[:150]}", flush=True)
                continue
            print(f"Scoring {key} ...", flush=True)
            scores[key] = (np.array(fn(benign)), np.array(fn(injected)))

    matched = matched_fpr_comparison(scores, target_fprs=FPRS)
    result = {
        "n_benign": len(benign), "n_injected": len(injected),
        "matched_fpr": {n: {str(f): v for f, v in per.items()} for n, per in matched.items()},
        "auroc": {n: auroc(b, i) for n, (b, i) in scores.items()},
        "perceived_role_counts": perceived,
        "paired_vs_transplant": {},
    }
    ref = "ours_transplant"
    if ref in scores:
        for name in scores:
            if name == ref:
                continue
            ref_b, ref_i = scores[ref]
            other_b, other_i = scores[name]
            result["paired_vs_transplant"][name] = {
                "flags": {str(f): paired_flag_test(ref_b, ref_i, other_b, other_i, f) for f in FPRS},
                "auroc_diff": auroc_diff_ci(ref_b, ref_i, other_b, other_i),
            }

    with open(OUT_DIR / "results_exp03_naive.json", "w") as f:
        json.dump(result, f, indent=2)
    np.savez(OUT_DIR / "scores_exp03_naive.npz",
             **{f"{n}__{k}": arr for n, (b, i) in scores.items() for k, arr in (("benign", b), ("injected", i))})

    print("\n" + "=" * 70 + "\nTPR at matched FPR, AUROC\n" + "=" * 70)
    print(f"{'detector':<36}{'TPR@1%':<10}{'TPR@5%':<10}{'AUROC'}")
    for n in scores:
        m = matched[n]
        print(f"{n:<36}{m[0.01]['tpr']:<10.3f}{m[0.05]['tpr']:<10.3f}{result['auroc'][n]:.3f}")
    return result


if __name__ == "__main__":
    import sys

    run(with_guardrails="--no-guardrails" not in sys.argv)
