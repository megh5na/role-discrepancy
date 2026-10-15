"""
experiments/exp06_overdefense/run_e6_scoring.py

E6 step 1: score the benign role-mixed corpus (Category C, 1,710 texts,
data/interim/benign_role_mixed.jsonl) and NotInject (339 real benign prompts
containing injection-trigger words, declared USER) with all six models (naive
and transplant, seeds 42, 123, 456) and SAVE THE 5-WAY ROLE PROBABILITIES, so
every scoring rule can be applied offline in analyze_e6.py.

Same preprocessing as the BIPIA runs: strip_delimiters(text), max_length 64,
batch 16. Seed-42 checkpoints live in checkpoints/; seeds 123/456 were cloned
(APFS copy-on-write) to checkpoints/kept_seed_models/ during the retrain.

Safeguard: before scoring, each loaded checkpoint must reproduce the first 20
saved BIPIA bare-injected probability rows for that model
(experiments/ablation_severity/probs_<model>.npz) to within 1e-5; a clone that
is not the intended model aborts the run instead of producing quietly wrong
numbers.

Outputs (this directory): e6_probs_<model>.npz with arrays `corpus` (1710, 5)
and `notinject` (339, 5), plus e6_texts_meta.json describing the inputs.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import time
from pathlib import Path

import numpy as np

from experiments.ablation_severity.run_severity_ablation import make_prob_scorer
from experiments.exp03_baselines.run_exp03_naive import load_spans
from experiments.exp03_baselines.run_exp03_wrapped import unique_in_order
from src.ingestion.io_utils import read_records
from src.preprocessing.delimiters import strip_delimiters

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ABL = REPO_ROOT / "experiments" / "ablation_severity"
CK = REPO_ROOT / "checkpoints"
MODELS = {
    "naive_s42": CK / "perceived_role_v1_naive",
    "transplant_s42": CK / "perceived_role_v1",
    "naive_s123": CK / "kept_seed_models" / "perceived_role_v1_naive_seed123",
    "transplant_s123": CK / "kept_seed_models" / "perceived_role_v1_seed123",
    "naive_s456": CK / "kept_seed_models" / "perceived_role_v1_naive_seed456",
    "transplant_s456": CK / "kept_seed_models" / "perceived_role_v1_seed456",
}


def load_inputs():
    corpus = list(read_records(REPO_ROOT / "data" / "interim" / "benign_role_mixed.jsonl"))
    notinject = list(read_records(REPO_ROOT / "data" / "interim" / "notinject.jsonl"))
    c_texts = [strip_delimiters(r.text) for r in corpus]
    n_texts = [strip_delimiters(r.text) for r in notinject]
    changed = sum(a != r.text for a, r in zip(c_texts, corpus))
    return corpus, notinject, c_texts, n_texts, changed


def identity_check(name: str, probs_fn) -> float:
    _, injected = load_spans()
    first20 = unique_in_order(injected)[:20]
    saved = np.load(ABL / f"probs_{name}.npz")["bare__injected"][:20]
    return float(np.max(np.abs(probs_fn(first20) - saved)))


def main():
    corpus, notinject, c_texts, n_texts, changed = load_inputs()
    print(f"corpus: {len(c_texts)} texts ({changed} changed by strip_delimiters); "
          f"NotInject: {len(n_texts)} texts", flush=True)
    meta = {"n_corpus": len(c_texts), "n_notinject": len(n_texts), "corpus_changed_by_strip": changed,
            "corpus_cells": [r.meta["cell"] for r in corpus],
            "corpus_declared": [r.declared_role.value for r in corpus],
            "corpus_control": [bool(r.meta["control"]) for r in corpus],
            "corpus_content_register": [r.meta["content_register"] for r in corpus]}
    (HERE / "e6_texts_meta.json").write_text(json.dumps(meta))

    for name, ckpt in MODELS.items():
        out = HERE / f"e6_probs_{name}.npz"
        if out.exists():
            print(f"SKIP {name}: exists", flush=True)
            continue
        if not (ckpt / "model.pt").exists():
            raise SystemExit(f"missing checkpoint for {name}: {ckpt}")
        t0 = time.time()
        probs_fn = make_prob_scorer(ckpt)
        diff = identity_check(name, probs_fn)
        print(f"{name}: identity check max|diff| vs saved BIPIA probs = {diff:.2e}", flush=True)
        if diff > 1e-5:
            raise SystemExit(f"{name}: checkpoint does not reproduce its saved probabilities (diff {diff}); aborting")
        np.savez(out, corpus=probs_fn(c_texts), notinject=probs_fn(n_texts))
        print(f"  saved {out.name} in {time.time() - t0:.0f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
