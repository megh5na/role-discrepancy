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
