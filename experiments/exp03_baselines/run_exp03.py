"""
experiments/exp03_baselines/run_exp03.py

Experiment 3 (Section 14): BASELINE DETECTION COMPARISON.
"Does discrepancy detection beat trained guardrails on indirect injection?"

Setup (first cut): BIPIA only for this pass (InjecAgent/LLMail
follow once this harness is validated). Our discrepancy scorer (C5) fed by
the trained perceived-role encoder checkpoint (C4) vs. guardrail baselines
(C8b) at matched FPR (C10) on BIPIA's own benign/injected split.

Every guardrail in KNOWN_GUARDRAILS is attempted; if a checkpoint fails to
download/load (gated, network, etc.) it's skipped with a clear message
rather than silently omitted from the printed summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.baselines.guardrails.harness import KNOWN_GUARDRAILS, GuardrailBaseline
from src.evaluation.matched_fpr import matched_fpr_comparison, print_matched_fpr_table
from src.ingestion.io_utils import read_records
from src.models.perceived_role.checkpoint import load_checkpoint
from src.models.perceived_role.encoder import load_tokenizer
from src.preprocessing.delimiters import strip_delimiters
from src.scoring.asymmetry import combined_score

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "perceived_role_v1"
BIPIA_PATH = REPO_ROOT / "data" / "interim" / "bipia.jsonl"
OUT_DIR = Path(__file__).resolve().parent


def get_our_scores(texts: list[str]) -> np.ndarray:
    """Our discrepancy scorer: perceived-role encoder -> combined_score
    against BIPIA's declared_role (DOCUMENT, per src/ingestion/bipia.py's
    channel mapping)."""
    import torch

    from src.ingestion.schema import ROLE_ORDER, Role

    model, meta = load_checkpoint(CHECKPOINT_DIR)
    tokenizer = load_tokenizer(meta["backbone_name"])

    scores = []
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=64)
        with torch.no_grad():
            logits = model(enc["input_ids"], enc["attention_mask"])
            probs = torch.softmax(logits, dim=-1).numpy()
        for p in probs:
            result = combined_score(p, Role.DOCUMENT, use_asymmetry=True)
            scores.append(result["score"])
    return np.array(scores)

# returns (benign_scores, injected_scores)

# run it through matched_fpr_comparision on that detector's own benign-score distribution such that benign does not exceed target, and return TPR on injected at that threshold. Repeat for each detector, then print a table of TPR@FPR for each detector at each target FPR.
def run():
    print("=" * 70)
    print("EXPERIMENT 3 -- Baseline detection comparison (first cut: BIPIA)")
    print("=" * 70)#

    records = list(read_records(BIPIA_PATH))
    benign_texts = [strip_delimiters(r.text) for r in records if r.is_injected is False]
    injected_texts = [strip_delimiters(r.text) for r in records if r.is_injected is True]
    print(f"BIPIA: {len(benign_texts)} benign spans, {len(injected_texts)} injected spans")

    # Cap for tractable first-cut runtime on this hardware.
    benign_texts = benign_texts[:800]
    injected_texts = injected_texts[:800]

    detector_scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    print("\n--- Our discrepancy scorer ---")
    if not CHECKPOINT_DIR.exists():
        print(f"  SKIPPED: no checkpoint at {CHECKPOINT_DIR} -- run train_production_checkpoint.py first")
    else:
        benign_scores = get_our_scores(benign_texts)
        injected_scores = get_our_scores(injected_texts)
        detector_scores["ours_discrepancy"] = (benign_scores, injected_scores)
        print(f"  scored {len(benign_scores)} benign, {len(injected_scores)} injected")

    print("\n--- Guardrail baselines ---")
    for key, spec in KNOWN_GUARDRAILS.items():
        print(f"  Loading {spec.name} ({spec.hf_model_id}) ...")
        try:
            guardrail = GuardrailBaseline(spec)
        except Exception as e:
            print(f"  SKIPPED {spec.name}: {type(e).__name__}: {str(e)[:200]}")
            continue
        score_fn = guardrail.make_score_fn()
        benign_scores = np.array(score_fn(benign_texts))
        injected_scores = np.array(score_fn(injected_texts))
        detector_scores[key] = (benign_scores, injected_scores)
        print(f"  scored {len(benign_scores)} benign, {len(injected_scores)} injected")

    if not detector_scores:
        print("\nNo detectors produced scores -- nothing to compare. Exiting.")
        return

    results = matched_fpr_comparison(detector_scores, target_fprs=[0.01, 0.05])
    print("\n" + "=" * 70)
    print("MATCHED-FPR RESULTS")
    print("=" * 70)
    print_matched_fpr_table(results, target_fprs=[0.01, 0.05])

    out_path = OUT_DIR / "results.json"
    serializable = {
        name: {str(fpr): v for fpr, v in per_fpr.items()} for name, per_fpr in results.items()
    }
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    run()
