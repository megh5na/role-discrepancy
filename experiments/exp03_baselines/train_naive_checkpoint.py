"""
experiments/exp03_baselines/train_naive_checkpoint.py

Trains the NAIVE-condition production checkpoint for E3-naive: the control
that tests whether the transplant construction is what makes the detector
work, not just the existence of a fine-tuned DeBERTa.

Recipe is a deliberate copy of experiments/exp01_swap_validity/
train_production_checkpoint.py, which produced the transplant checkpoint
used in E3: full set (6,317 examples), same shuffle seed (7) and 90/10
split, TrainConfig(batch_size=8, max_length=32, n_epochs=2, seed=42). The
ONLY difference is the data file (naive_control.jsonl vs
transplant_train.jsonl). naive_control is built from the same balanced span
pool as transplant_train, so the two differ only in the construction.

Usage:
    python -m experiments.exp03_baselines.train_naive_checkpoint
    python -m experiments.exp03_baselines.train_naive_checkpoint --smoke   # tiny, writes to /tmp
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import random
from pathlib import Path

from experiments.exp01_swap_validity.train_production_checkpoint import load_jsonl
from src.models.perceived_role.checkpoint import save_checkpoint
from src.models.perceived_role.train import TrainConfig, train_one_run
from src.supervision.types import TrainingExample

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "perceived_role_v1_naive"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny run, saves to /tmp, for plumbing checks only")
    args = ap.parse_args()

    examples = load_jsonl(PROCESSED / "naive_control.jsonl", TrainingExample)
    print(f"Loaded {len(examples)} naive-condition training examples", flush=True)

    rng = random.Random(7)  # same split seed as the transplant production checkpoint
    shuffled = examples[:]
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * 0.1)
    val_ex, train_ex = shuffled[:n_val], shuffled[n_val:]

    out_dir = CHECKPOINT_DIR
    n_epochs = 2
    if args.smoke:
        train_ex, val_ex = train_ex[:40], val_ex[:16]
        out_dir, n_epochs = Path("/tmp/naive_ckpt_smoke"), 1
    print(f"Train: {len(train_ex)}  Val: {len(val_ex)}", flush=True)

    config = TrainConfig(batch_size=8, max_length=32, n_epochs=n_epochs, seed=42)
    model, history = train_one_run(train_ex, val_ex, config)
    print("Training complete. History:", history, flush=True)

    save_checkpoint(
        model, config, out_dir,
        extra={
            "condition": "naive",
            "n_train": len(train_ex),
            "n_val": len(val_ex),
            "val_acc_history": history["val_acc"],
            "purpose": "E3-naive control: same recipe as perceived_role_v1, naive data",
        },
    )
    print(f"Saved checkpoint to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
