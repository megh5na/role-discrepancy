from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import time
from pathlib import Path

import numpy as np

from experiments.ablation_severity.run_severity_ablation import make_prob_scorer
from experiments.exp06_overdefense.run_e6_scoring import MODELS, identity_check
from src.baselines.guardrails.harness import KNOWN_GUARDRAILS, GuardrailBaseline

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    texts = [json.loads(l)["text"] for l in open(REPO / "data" / "interim" / "e4_llmail_short.jsonl") if l.strip()]
    print(f"LLMail short set: {len(texts)} texts", flush=True)
    for name, ckpt in MODELS.items():
        out = HERE / f"e4_probs_{name}.npz"
        if out.exists():
            print(f"SKIP {name}", flush=True)
            continue
        t0 = time.time()
        fn = make_prob_scorer(ckpt)
        diff = identity_check(name, fn)
        if diff > 1e-5:
            raise SystemExit(f"{name}: identity check failed ({diff})")
        np.savez(out, llmail=fn(texts))
        print(f"{name}: identity {diff:.1e}, saved in {time.time() - t0:.0f}s", flush=True)
    arrays = {}
    for key, spec in KNOWN_GUARDRAILS.items():
        try:
            fn = GuardrailBaseline(spec).make_score_fn()
        except Exception as e:
            print(f"SKIPPED {spec.name}: {type(e).__name__}", flush=True)
            continue
        arrays[f"{key}__llmail"] = np.array(fn(texts))
        print(f"{key}: scored", flush=True)
    np.savez(HERE / "e4_guardrail_scores.npz", **arrays)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
