"""
experiments/ablation_severity/run_severity_ablation.py

Step 1 of the severity ablation: score the saved checkpoints once and SAVE
THE 5-WAY ROLE PROBABILITIES for every span in all 12 conditions (bare, 5
known carriers, 6 novel wrappers; 549 unique benign + 120 unique injected
BIPIA strings, same inputs as run_exp03_wrapped.py). The earlier runs only
saved final scores, which cannot be re-scored under a different rule.
Every scoring variant (analyze_severity_ablation.py) is then computed offline
from the SAME probabilities, so the only thing that differs between variants
is the scoring rule.

Checkpoints used: the seed-42 naive and transplant checkpoints, the only ones
still on disk (the seed-study checkpoints were deleted after scoring). Seed
42 is the weakest-looking seed for both conditions in the earlier seed study,
so this is a single-seed ablation and must be read as such.

Outputs: probs_naive_s42.npz, probs_transplant_s42.npz in this directory
(keys <cond>__benign / <cond>__injected, arrays of shape (n, 5) in ROLE_ORDER).
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import time
from pathlib import Path

import numpy as np

from experiments.exp03_baselines.run_exp03_naive import load_spans
from experiments.exp03_baselines.run_exp03_wrapped import build_conditions, unique_in_order

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CHECKPOINTS = {
    "naive_s42": REPO_ROOT / "checkpoints" / "perceived_role_v1_naive",
    "transplant_s42": REPO_ROOT / "checkpoints" / "perceived_role_v1",
}


def make_prob_scorer(ckpt_dir: Path):
    """texts -> (n, 5) role probabilities. Same batching and max_length
    (16 / 64) as the earlier E3 scoring, so probabilities match the scores
    those runs saved."""
    import torch

    from src.models.perceived_role.checkpoint import load_checkpoint
    from src.models.perceived_role.encoder import load_tokenizer

    model, meta = load_checkpoint(ckpt_dir)
    tok = load_tokenizer(meta["backbone_name"])

    def probs(texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 16):
            enc = tok(texts[i:i + 16], return_tensors="pt", padding=True, truncation=True, max_length=64)
            with torch.no_grad():
                out.append(torch.softmax(model(enc["input_ids"], enc["attention_mask"]), dim=-1).numpy())
        return np.concatenate(out, axis=0)

    return probs


def main():
    benign_all, injected_all = load_spans()
    benign, injected = unique_in_order(benign_all), unique_in_order(injected_all)
    ben_c, inj_c = build_conditions(benign), build_conditions(injected)
    print(f"unique benign={len(benign)} unique injected={len(injected)} conditions={len(ben_c)}", flush=True)
    for name, ckpt in CHECKPOINTS.items():
        out_path = HERE / f"probs_{name}.npz"
        if out_path.exists():
            print(f"SKIP {name}: {out_path.name} exists", flush=True)
            continue
        if not (ckpt / "model.pt").exists():
            raise SystemExit(f"missing checkpoint {ckpt}")
        print(f"Scoring {name} ...", flush=True)
        t0 = time.time()
        probs = make_prob_scorer(ckpt)
        arrays = {}
        for c in ben_c:
            arrays[f"{c}__benign"] = probs(ben_c[c])
            arrays[f"{c}__injected"] = probs(inj_c[c])
        np.savez(out_path, **arrays)
        print(f"  saved {out_path.name} in {time.time() - t0:.0f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
