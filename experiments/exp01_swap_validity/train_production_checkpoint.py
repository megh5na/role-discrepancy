"""
experiments/exp01_swap_validity/train_production_checkpoint.py

Trains ONE encoder on the FULL transplant_train.jsonl set (not E1's
subsample) and persists it — this is the checkpoint E3 (BIPIA detection vs
guardrails) will actually use to score spans. E1's runs were deliberately
throwaway (fast controlled comparison, in-memory only); this is the first
"real" artifact.

Uses the TRANSPLANT construction (not naive) since E1 already established
it's the better condition (swap_consistency 0.937 vs 0.711) — no reason to
ship the worse-performing naive model as our actual detector.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.models.perceived_role.checkpoint import save_checkpoint
from src.models.perceived_role.train import TrainConfig, train_one_run
from src.supervision.types import TrainingExample

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "perceived_role_v1"


def load_jsonl(path: Path, cls):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(cls.from_dict(json.loads(line)))
    return items


def main():
    examples = load_jsonl(PROCESSED / "transplant_train.jsonl", TrainingExample)
    print(f"Loaded {len(examples)} transplant-condition training examples")

    # 90/10 train/val split, stratified-ish by simple shuffle (full dataset
    # is already balanced by label from C3, so a plain shuffle keeps that).
    import random

    rng = random.Random(7)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * 0.1)
    val_ex = shuffled[:n_val]
    train_ex = shuffled[n_val:]
    print(f"Train: {len(train_ex)}  Val: {len(val_ex)}")

    config = TrainConfig(batch_size=8, max_length=32, n_epochs=2, seed=42)
    model, history = train_one_run(train_ex, val_ex, config)
    print("Training complete. History:", history)

    save_checkpoint(
        model,
        config,
        CHECKPOINT_DIR,
        extra={
            "condition": "transplant",
            "n_train": len(train_ex),
            "n_val": len(val_ex),
            "val_acc_history": history["val_acc"],
            "purpose": "E3 discrepancy scoring checkpoint",
        },
    )
    print(f"Saved checkpoint to {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
