"""
experiments/exp03_baselines/run_exp03_seeds.py

Seed study for E3-naive / E3-wrapped. Those results came from ONE training
seed (42) per checkpoint, and E2 showed large seed-to-seed swings, so seed
uncertainty was unmeasured. This trains the naive and transplant checkpoints
at additional seeds and scores each on the same 12 wrapped conditions
(bare + 5 known carriers + 6 novel wrappers) as run_exp03_wrapped.py.

Recipe is identical to the seed-42 production checkpoints (full 6,317-example
set, split seed 7, 5,686/631, batch 8, max_length 32, 2 epochs). ONLY the
training seed (classifier-head init, data order, dropout) changes; the
train/val split and the data are fixed, so this isolates training randomness
and does not measure sensitivity to the data split.

Per job: train -> save checkpoint -> score all conditions -> save scores ->
delete the checkpoint (disk is ~97% full; training is deterministic per seed,
so a checkpoint can be regenerated). Resume-safe: a job whose scores file
exists is skipped; a job whose checkpoint exists but has no scores is only
scored. Use --keep to retain checkpoints.

Outputs: scores_exp03_wrapped_seed{N}_{condition}.npz (keys <cond>__benign /
<cond>__injected), train histories in seed_study_train_meta.json.

Usage:
    python -m experiments.exp03_baselines.run_exp03_seeds
    python -m experiments.exp03_baselines.run_exp03_seeds --smoke
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import gc
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np

from experiments.exp01_swap_validity.train_production_checkpoint import load_jsonl
from experiments.exp03_baselines.run_exp03_naive import load_spans
from experiments.exp03_baselines.run_exp03_wrapped import build_conditions, make_scorer, unique_in_order
from src.models.perceived_role.checkpoint import save_checkpoint
from src.models.perceived_role.train import TrainConfig, train_one_run
from src.supervision.types import TrainingExample

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"
HERE = Path(__file__).resolve().parent
DATA_FILE = {"naive": "naive_control.jsonl", "transplant": "transplant_train.jsonl"}
# interleaved, so a partial run still has a second seed for BOTH conditions
JOBS = [("naive", 123), ("transplant", 123), ("naive", 456), ("transplant", 456)]


def ckpt_dir(root: Path, condition: str, seed: int) -> Path:
    suffix = "_naive" if condition == "naive" else ""
    return root / f"perceived_role_v1{suffix}_seed{seed}"


def train_one(condition: str, seed: int, out_dir: Path, smoke: bool = False) -> dict:
    examples = load_jsonl(PROCESSED / DATA_FILE[condition], TrainingExample)
    rng = random.Random(7)  # fixed split seed: same as the seed-42 production checkpoints
    shuffled = examples[:]
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * 0.1)
    val_ex, train_ex = shuffled[:n_val], shuffled[n_val:]
    n_epochs = 2
    if smoke:
        train_ex, val_ex, n_epochs = train_ex[:40], val_ex[:16], 1
    config = TrainConfig(batch_size=8, max_length=32, n_epochs=n_epochs, seed=seed)
    t0 = time.time()
    model, history = train_one_run(train_ex, val_ex, config)
    meta = {"condition": condition, "seed": seed, "n_train": len(train_ex), "n_val": len(val_ex),
            "val_acc_history": history["val_acc"], "train_loss": history["train_loss"],
            "train_seconds": round(time.time() - t0, 1)}
    save_checkpoint(model, config, out_dir, extra={**meta, "purpose": "E3 seed study"})
    del model
    gc.collect()
    return meta


def score_all(ckpt: Path, ben_c: dict, inj_c: dict) -> dict:
    score = make_scorer(ckpt)
    return {c: (score(ben_c[c]), score(inj_c[c])) for c in ben_c}


def run(smoke: bool = False, keep: bool = False):
    ckpt_root = Path("/tmp/seed_study_smoke_ckpts") if smoke else REPO_ROOT / "checkpoints"
    out_dir = Path("/tmp/seed_study_smoke_out") if smoke else HERE
    out_dir.mkdir(parents=True, exist_ok=True)

    benign_all, injected_all = load_spans()
    benign, injected = unique_in_order(benign_all), unique_in_order(injected_all)
    if smoke:
        benign, injected = benign[:12], injected[:6]
    ben_c, inj_c = build_conditions(benign), build_conditions(injected)
    print(f"unique benign={len(benign)} unique injected={len(injected)} conditions={len(ben_c)}", flush=True)

    meta_path = out_dir / "seed_study_train_meta.json"
    all_meta = json.loads(meta_path.read_text()) if meta_path.exists() else []

    for condition, seed in (JOBS[:1] if smoke else JOBS):
        scores_path = out_dir / f"scores_exp03_wrapped_seed{seed}_{condition}.npz"
        if scores_path.exists():
            print(f"SKIP {condition} seed {seed}: scores already exist", flush=True)
            continue
        ckpt = ckpt_dir(ckpt_root, condition, seed)
        if (ckpt / "model.pt").exists():
            print(f"RESUME {condition} seed {seed}: checkpoint exists, scoring only", flush=True)
        else:
            print(f"\nTRAIN {condition} seed {seed}", flush=True)
            meta = train_one(condition, seed, ckpt, smoke=smoke)
            all_meta = [m for m in all_meta if not (m["condition"] == condition and m["seed"] == seed)] + [meta]
            meta_path.write_text(json.dumps(all_meta, indent=2))
            print(f"  trained in {meta['train_seconds']}s, val_acc {meta['val_acc_history']}", flush=True)

        print(f"SCORE {condition} seed {seed}", flush=True)
        t0 = time.time()
        scored = score_all(ckpt, ben_c, inj_c)
        np.savez(scores_path, **{f"{c}__{k}": a for c, (b, i) in scored.items()
                                 for k, a in (("benign", b), ("injected", i))})
        print(f"  scored in {time.time() - t0:.0f}s -> {scores_path.name}", flush=True)
        if not keep:
            shutil.rmtree(ckpt, ignore_errors=True)
            print(f"  removed checkpoint {ckpt.name}", flush=True)
        gc.collect()
    print("\nALL JOBS DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny run into /tmp, plumbing check only")
    ap.add_argument("--keep", action="store_true", help="keep checkpoints after scoring")
    a = ap.parse_args()
    run(smoke=a.smoke, keep=a.keep)
