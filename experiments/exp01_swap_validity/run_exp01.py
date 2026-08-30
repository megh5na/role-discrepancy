"""
experiments/exp01_swap_validity/run_exp01.py

Experiment 1 (Section 14): DOES THE MODEL LEARN REGISTER OR POSITION?
*** THE GATING EXPERIMENT (Section 24 M1, Section 26 risks: "If swap
consistency is at chance after two supervision attempts, STOP and escalate
to the guide immediately.") ***

Compares FOUR conditions, all on IDENTICAL underlying spans (same balanced
pool, per src.supervision.transplant.build_all_supervision_sets):
  1. Perceived-role encoder trained on NAIVE-LABELLED data (the confound
     baseline -- position/label are always equal, exactly like natural data)
  2. Perceived-role encoder trained on TRANSPLANTED data (our construction)
  3. TF-IDF+LR trained on NAIVE-LABELLED data (falsification control)
  4. TF-IDF+LR trained on TRANSPLANTED data (falsification control)

...evaluated on the SAME held-out swap-test pairs (never seen in training,
Section 8 C3(c)), using swap_consistency as the primary metric (Section 15).

Expected outcome per spec: naive -> high natural accuracy, LOW swap
consistency (follows position). Transplanted -> somewhat lower natural
accuracy, substantially HIGHER swap consistency (follows register).

COMPUTE-DRIVEN SCOPE REDUCTION, STATED HONESTLY (not hidden): this is a
CPU/MPS laptop, not a GPU cluster (docs/decisions.md). DeBERTa-v3-base
trains at ~3.4s/step (bs=16) on this machine, measured directly (see
conversation/build log). At that throughput, the full balanced set
(~6,300 examples/condition) x 2 conditions x the spec's target 3-5 seeds is
not tractable in a single working session. This run uses a SUBSAMPLED
training pool (SUBSAMPLE_N per condition) and REDUCED_SEEDS, both named as
config fields so the reduction is visible in every result file, not silently
assumed. Scaling to the full set and full seed count is explicit follow-up
work (tracked in docs/decisions.md / Section 21 checklist), not a
substitute for it.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

from src.baselines.simple.tfidf_baseline import make_predict_fn as tfidf_predict_fn
from src.baselines.simple.tfidf_baseline import train_tfidf_lr
from src.evaluation.metrics import role_accuracy, swap_consistency
from src.ingestion.schema import ROLE_TO_IDX
from src.models.perceived_role.inference import make_predict_fn as encoder_predict_fn
from src.models.perceived_role.train import TrainConfig, evaluate_accuracy, train_one_run
from src.supervision.types import SwapPairItem, TrainingExample

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"
OUT_DIR = Path(__file__).resolve().parent

SUBSAMPLE_N = 1500       # examples per condition, per docstring note above
VAL_FRACTION = 0.15
SEEDS = [42, 123]        # REDUCED from spec's target 3-5 -- see docstring
N_EPOCHS = 2
BATCH_SIZE = 16
MAX_LENGTH = 48


def load_jsonl(path: Path, cls):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(cls.from_dict(json.loads(line)))
    return items


def stratified_subsample_and_split(
    examples: list[TrainingExample], n_total: int, val_fraction: float, seed: int
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """Subsample to n_total (stratified by label to keep class balance),
    then carve off a validation split. IMPORTANT: this val split comes from
    the TRAINING pool, never from swap_test_pairs -- selecting hyperparameters
    or checkpoints against swap pairs would itself be leakage (Section 21
    checklist: "best config selected on validation (NOT on LODO test
    folds)")."""
    rng = random.Random(seed)
    by_label: dict = {}
    for ex in examples:
        by_label.setdefault(ex.label, []).append(ex)

    n_labels = len(by_label)
    per_label = max(1, n_total // n_labels)

    subsampled = []
    for label, group in by_label.items():
        shuffled = group[:]
        rng.shuffle(shuffled)
        subsampled.extend(shuffled[:per_label])
    rng.shuffle(subsampled)

    n_val = int(len(subsampled) * val_fraction)
    val = subsampled[:n_val]
    train = subsampled[n_val:]
    return train, val


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def run():
    print("=" * 70)
    print("EXPERIMENT 1 -- Swap validity (THE GATING EXPERIMENT)")
    print("=" * 70)

    naive_all = load_jsonl(PROCESSED / "naive_control.jsonl", TrainingExample)
    transplant_all = load_jsonl(PROCESSED / "transplant_train.jsonl", TrainingExample)
    swap_pairs = load_jsonl(PROCESSED / "swap_test_pairs.jsonl", SwapPairItem)
    print(f"Loaded {len(naive_all)} naive, {len(transplant_all)} transplant, {len(swap_pairs)} swap pairs")

    results = []

    conditions = {
        "naive": naive_all,
        "transplant": transplant_all,
    }

    # --- TF-IDF+LR falsification control (fast, full data, one fit per condition) ---
    print("\n--- TF-IDF+LR baseline (deep-learning falsification control, C8a) ---")
    for cond_name, examples in conditions.items():
        pipe = train_tfidf_lr(examples, seed=42)
        pfn = tfidf_predict_fn(pipe)
        metrics = swap_consistency(pfn, swap_pairs)
        print(f"  [tfidf_lr | {cond_name}] swap_consistency={metrics['swap_consistency']:.3f} "
              f"native_acc={metrics['native_acc']:.3f} foreign_acc={metrics['foreign_acc']:.3f}")
        results.append({
            "model": "tfidf_lr",
            "condition": cond_name,
            "seed": 42,
            "n_train": len(examples),
            **metrics,
        })

    # --- Perceived-role encoder, both conditions, multiple seeds ---
    print("\n--- Perceived-role encoder (DeBERTa-v3-base) ---")
    for cond_name, examples in conditions.items():
        for seed in SEEDS:
            train_ex, val_ex = stratified_subsample_and_split(
                examples, SUBSAMPLE_N, VAL_FRACTION, seed=seed
            )
            cfg = TrainConfig(
                batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, seed=seed, max_length=MAX_LENGTH
            )
            print(f"\n  Training encoder | condition={cond_name} seed={seed} "
                  f"n_train={len(train_ex)} n_val={len(val_ex)} ...")
            t0 = time.time()
            model, history = train_one_run(train_ex, val_ex, cfg)
            elapsed = time.time() - t0
            print(f"  done in {elapsed:.0f}s. history={history}")

            from src.models.perceived_role.encoder import load_tokenizer
            tok = load_tokenizer(cfg.backbone_name)
            pfn = encoder_predict_fn(model, tok, cfg.device, max_length=cfg.max_length, batch_size=cfg.batch_size)

            sc = swap_consistency(pfn, swap_pairs)
            print(f"  [encoder | {cond_name} | seed={seed}] swap_consistency={sc['swap_consistency']:.3f} "
                  f"native_acc={sc['native_acc']:.3f} foreign_acc={sc['foreign_acc']:.3f}")

            cfg_dict = {
                "model": "deberta_v3_base_encoder",
                "condition": cond_name,
                "seed": seed,
                "n_train": len(train_ex),
                "n_val": len(val_ex),
                "n_epochs": N_EPOCHS,
                "batch_size": BATCH_SIZE,
                "max_length": MAX_LENGTH,
                "subsample_n": SUBSAMPLE_N,
            }
            results.append({
                **cfg_dict,
                "config_hash": config_hash(cfg_dict),
                "train_history": history,
                "elapsed_sec": elapsed,
                **sc,
            })

    out_path = OUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {len(results)} result rows to {out_path}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'model':<20}{'condition':<14}{'seed':<8}{'swap_consist':<15}{'native_acc':<12}{'foreign_acc':<12}")
    for r in results:
        print(f"{r['model']:<20}{r['condition']:<14}{r['seed']:<8}"
              f"{r['swap_consistency']:<15.3f}{r['native_acc']:<12.3f}{r['foreign_acc']:<12.3f}")


if __name__ == "__main__":
    run()
