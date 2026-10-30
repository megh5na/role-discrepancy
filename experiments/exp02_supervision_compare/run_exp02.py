from __future__ import annotations

import os

# Diagnosed during the first E2 run (docs/decisions.md): every tokenizer
# load was doing a live HEAD request to huggingface.co that repeatedly
# timed out (10s x up to 5 retries) before falling back to the local
# cache, adding real, avoidable wall-clock time across many training runs
# in one experiment. Both microsoft/deberta-v3-base and the sidecar models
# are already cached locally from prior runs, so offline mode doesn't
# block anything that hasn't been fetched before. Set before importing
# transformers so it's respected on first use.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import hashlib
import json
import random
import time
from pathlib import Path

import yaml

from src.baselines.simple.tfidf_baseline import make_predict_fn as tfidf_predict_fn
from src.baselines.simple.tfidf_baseline import train_tfidf_lr
from src.evaluation.lodo import lodo_splits
from src.ingestion.io_utils import read_records
from src.models.perceived_role.encoder import load_tokenizer
from src.models.perceived_role.inference import make_predict_fn as encoder_predict_fn
from src.models.perceived_role.train import TrainConfig, train_one_run
from src.supervision.transplant import build_all_supervision_sets
from src.supervision.types import TrainingExample

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "experiment" / "exp02_lodo.yaml"
OUT_DIR = Path(__file__).resolve().parent


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def stratified_subsample_and_split(examples, n_total, val_fraction, seed):
    """Same logic as run_exp01.py's helper (duplicated, not imported, per
    each experiment script being self-contained): subsample to n_total
    (stratified by label), then carve off a validation split from the
    TRAINING pool only -- never from the held-out corpus, which is the
    actual test fold here."""
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
    return subsampled[n_val:], subsampled[:n_val]


def accuracy_on_held_out(predict_fn, held_out_records) -> dict:
    """Held-out corpus is single-label by construction (every record in a
    LODO fold's test set shares the same declared_role -- it's one
    corpus). Returns accuracy plus n, so a thin fold doesn't get reported
    without its own sample-size caveat alongside it."""
    texts = [r.text for r in held_out_records]
    true_label = held_out_records[0].declared_role
    preds = predict_fn(texts)
    n_correct = sum(1 for p in preds if p == true_label)
    return {
        "true_label": true_label.value,
        "n": len(held_out_records),
        "accuracy": n_correct / len(held_out_records) if held_out_records else float("nan"),
    }


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def run():
    cfg = load_config()
    print("=" * 70)
    print("EXPERIMENT 2 -- Leave-one-dataset-out supervision comparison")
    print("=" * 70)
    print(f"Config: {CONFIG_PATH}")

    valid_corpora = set(cfg["valid_lodo_corpora"])
    excluded = cfg["excluded_corpora"]
    print(f"Valid LODO folds: {sorted(valid_corpora)}")
    print(f"Excluded (single-sourced label, would remove the label entirely): {excluded}")

    all_records = list(read_records(REPO_ROOT / cfg["data"]["clean_spans_path"]))
    print(f"Loaded {len(all_records)} clean register-source spans")

    sup_cfg = cfg["supervision"]
    tr_cfg = cfg["training"]

    results = []
    fold_count = 0
    for held_out, train_records, test_records in lodo_splits(all_records):
        if held_out not in valid_corpora:
            continue  # structurally excluded -- see config docstring
        fold_count += 1
        print(f"\n{'=' * 70}")
        print(f"FOLD {fold_count}: held out = {held_out}  "
              f"(train={len(train_records)} records, test={len(test_records)} records)")
        print(f"{'=' * 70}")

        sets = build_all_supervision_sets(
            train_records,
            transplant_rate=sup_cfg["transplant_rate"],
            n_swap_pairs_per_label=sup_cfg["n_swap_pairs_per_label"],
            seed=sup_cfg["seed"],
        )
        conditions = {"naive": sets["naive_control"], "transplant": sets["transplant_train"]}
        print(f"  naive_control={len(sets['naive_control'])}  "
              f"transplant_train={len(sets['transplant_train'])}  "
              f"swap_pairs={len(sets['swap_test_pairs'])} (unused here, built for parity w/ E1 construction)")

        # --- TF-IDF+LR falsification control, full per-fold training set ---
        for cond_name, examples in conditions.items():
            pipe = train_tfidf_lr(examples, seed=42)
            pfn = tfidf_predict_fn(pipe)
            ho = accuracy_on_held_out(pfn, test_records)
            print(f"  [tfidf_lr | {cond_name}] held_out={held_out} "
                  f"true_label={ho['true_label']} n={ho['n']} accuracy={ho['accuracy']:.3f}")
            results.append({
                "model": "tfidf_lr", "held_out_corpus": held_out, "condition": cond_name,
                "seed": 42, "n_train": len(examples), **ho,
            })

        # --- DeBERTa encoder, both conditions ---
        for cond_name, examples in conditions.items():
            for seed in tr_cfg["seeds"]:
                train_ex, val_ex = stratified_subsample_and_split(
                    examples, tr_cfg["subsample_n"], tr_cfg["val_fraction"], seed=seed
                )
                train_config = TrainConfig(
                    batch_size=tr_cfg["batch_size"], n_epochs=tr_cfg["n_epochs"],
                    seed=seed, max_length=tr_cfg["max_length"],
                )
                print(f"\n  Training encoder | held_out={held_out} condition={cond_name} "
                      f"seed={seed} n_train={len(train_ex)} n_val={len(val_ex)} ...")
                t0 = time.time()
                model, history = train_one_run(train_ex, val_ex, train_config)
                elapsed = time.time() - t0
                print(f"  done in {elapsed:.0f}s. val_acc history={history['val_acc']}")

                tok = load_tokenizer(train_config.backbone_name)
                pfn = encoder_predict_fn(
                    model, tok, train_config.device,
                    max_length=train_config.max_length, batch_size=train_config.batch_size,
                )
                ho = accuracy_on_held_out(pfn, test_records)
                print(f"  [encoder | {cond_name} | seed={seed}] held_out={held_out} "
                      f"true_label={ho['true_label']} n={ho['n']} accuracy={ho['accuracy']:.3f}")

                result_cfg = {
                    "model": "deberta_v3_base_encoder", "held_out_corpus": held_out,
                    "condition": cond_name, "seed": seed, "n_train": len(train_ex),
                    "n_val": len(val_ex), "n_epochs": tr_cfg["n_epochs"],
                    "batch_size": tr_cfg["batch_size"], "max_length": tr_cfg["max_length"],
                    "subsample_n": tr_cfg["subsample_n"],
                }
                results.append({
                    **result_cfg, "config_hash": config_hash(result_cfg),
                    "train_history": history, "elapsed_sec": elapsed, **ho,
                })

        # Write incrementally after every fold so a long multi-hour run
        # that gets interrupted still leaves completed folds on disk.
        out_path = OUT_DIR / "results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  [checkpoint] wrote {len(results)} result rows so far -> {out_path}")

    print("\n" + "=" * 70)
    print("SUMMARY -- held-out-corpus accuracy by model/condition")
    print("=" * 70)
    print(f"{'model':<24}{'held_out':<24}{'condition':<14}{'true_label':<12}{'accuracy'}")
    for r in results:
        print(f"{r['model']:<24}{r['held_out_corpus']:<24}{r['condition']:<14}"
              f"{r['true_label']:<12}{r['accuracy']:.3f}")

    # Aggregate: mean accuracy per (model, condition), across all folds.
    from collections import defaultdict
    agg: dict = defaultdict(list)
    for r in results:
        agg[(r["model"], r["condition"])].append(r["accuracy"])
    print("\nMean held-out-corpus accuracy, averaged across all LODO folds:")
    for (model, cond), accs in sorted(agg.items()):
        mean_acc = sum(accs) / len(accs)
        print(f"  {model:<24}{cond:<14} mean={mean_acc:.3f}  (n_folds={len(accs)})")


if __name__ == "__main__":
    run()
