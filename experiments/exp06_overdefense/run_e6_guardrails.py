from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import time
from pathlib import Path

import numpy as np

from experiments.exp06_overdefense.run_e6_scoring import load_inputs
from src.baselines.guardrails.harness import KNOWN_GUARDRAILS, GuardrailBaseline

HERE = Path(__file__).resolve().parent


def main():
    _, _, c_texts, n_texts, changed = load_inputs()
    print(f"corpus {len(c_texts)} texts ({changed} changed by strip), NotInject {len(n_texts)}", flush=True)
    arrays = {}
    for key, spec in KNOWN_GUARDRAILS.items():
        try:
            fn = GuardrailBaseline(spec).make_score_fn()
        except Exception as e:
            print(f"SKIPPED {spec.name}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        t0 = time.time()
        arrays[f"{key}__corpus"] = np.array(fn(c_texts))
        arrays[f"{key}__notinject"] = np.array(fn(n_texts))
        print(f"{key}: scored in {time.time() - t0:.0f}s; corpus score range "
              f"[{arrays[f'{key}__corpus'].min():.4f}, {arrays[f'{key}__corpus'].max():.4f}]", flush=True)
    np.savez(HERE / "e6_guardrail_scores.npz", **arrays)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
