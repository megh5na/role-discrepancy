from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import gc
import json
import shutil
import time
from pathlib import Path

import numpy as np

from experiments.ablation_severity.analyze_severity_ablation import REFERENCE, score_variant
from experiments.ablation_severity.run_severity_ablation import make_prob_scorer
from experiments.exp03_baselines.run_exp03_naive import load_spans
from experiments.exp03_baselines.run_exp03_seeds import JOBS, ckpt_dir, train_one
from experiments.exp03_baselines.run_exp03_wrapped import build_conditions, unique_in_order

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = REPO_ROOT / "experiments" / "exp03_baselines"


def determinism_check(condition: str, seed: int, arrays: dict, train_meta: dict) -> dict:
    """Compare against what the seed study saved for the same model."""
    out = {"scores_max_abs_diff": None, "val_acc_match": None}
    saved_path = BASE / f"scores_exp03_wrapped_seed{seed}_{condition}.npz"
    if saved_path.exists():
        saved = np.load(saved_path)
        worst = 0.0
        for key, probs in arrays.items():
            worst = max(worst, float(np.max(np.abs(score_variant(probs, REFERENCE) - saved[key]))))
        out["scores_max_abs_diff"] = worst
    meta_path = BASE / "seed_study_train_meta.json"
    if meta_path.exists():
        for m in json.loads(meta_path.read_text()):
            if m["condition"] == condition and m["seed"] == seed:
                out["val_acc_match"] = bool(np.allclose(m["val_acc_history"], train_meta["val_acc_history"], atol=1e-9))
    return out


def run(smoke: bool = False, keep: bool = False):
    ckpt_root = Path("/tmp/sev_seeds_smoke_ckpts") if smoke else REPO_ROOT / "checkpoints"
    out_dir = Path("/tmp/sev_seeds_smoke_out") if smoke else HERE
    out_dir.mkdir(parents=True, exist_ok=True)

    benign_all, injected_all = load_spans()
    benign, injected = unique_in_order(benign_all), unique_in_order(injected_all)
    if smoke:
        benign, injected = benign[:12], injected[:6]
    ben_c, inj_c = build_conditions(benign), build_conditions(injected)
    print(f"unique benign={len(benign)} unique injected={len(injected)} conditions={len(ben_c)}", flush=True)

    meta_path = out_dir / "severity_seeds_meta.json"
    all_meta = json.loads(meta_path.read_text()) if meta_path.exists() else []

    for condition, seed in (JOBS[:1] if smoke else JOBS):
        probs_path = out_dir / f"probs_{condition}_s{seed}.npz"
        if probs_path.exists():
            print(f"SKIP {condition} seed {seed}: probabilities already saved", flush=True)
            continue
        ckpt = ckpt_dir(ckpt_root, condition, seed)
        if (ckpt / "model.pt").exists():
            print(f"RESUME {condition} seed {seed}: checkpoint exists, scoring only", flush=True)
            tmeta = {"val_acc_history": None, "train_seconds": None}
        else:
            print(f"\nTRAIN {condition} seed {seed}", flush=True)
            tmeta = train_one(condition, seed, ckpt, smoke=smoke)
            print(f"  trained in {tmeta['train_seconds']}s, val_acc {tmeta['val_acc_history']}", flush=True)

        print(f"PROBS {condition} seed {seed}", flush=True)
        t0 = time.time()
        probs = make_prob_scorer(ckpt)
        arrays = {}
        for c in ben_c:
            arrays[f"{c}__benign"] = probs(ben_c[c])
            arrays[f"{c}__injected"] = probs(inj_c[c])
        np.savez(probs_path, **arrays)
        print(f"  saved {probs_path.name} in {time.time() - t0:.0f}s", flush=True)

        if not smoke and tmeta["val_acc_history"] is not None:
            chk = determinism_check(condition, seed, arrays, tmeta)
            print(f"  determinism vs seed study: max|score diff|={chk['scores_max_abs_diff']}, "
                  f"val_acc history match={chk['val_acc_match']}", flush=True)
            all_meta = [m for m in all_meta if not (m["condition"] == condition and m["seed"] == seed)]
            all_meta.append({"condition": condition, "seed": seed, **tmeta, **chk})
            meta_path.write_text(json.dumps(all_meta, indent=2))
        if not keep:
            shutil.rmtree(ckpt, ignore_errors=True)
            print(f"  removed checkpoint {ckpt.name}", flush=True)
        gc.collect()
    print("\nALL JOBS DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    run(smoke=a.smoke, keep=a.keep)
