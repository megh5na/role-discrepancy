from __future__ import annotations

import os

# Same reasoning as run_exp02.py: avoid the HuggingFace Hub network retries.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import random
import time
from collections import defaultdict
from pathlib import Path

import yaml

from experiments.exp02_supervision_compare.run_exp02 import (
    config_hash,
    stratified_subsample_and_split,
)
from src.baselines.simple.tfidf_baseline import make_predict_fn as tfidf_predict_fn
from src.baselines.simple.tfidf_baseline import train_tfidf_lr
from src.evaluation.lodo import lodo_splits
from src.ingestion.io_utils import read_records
from src.models.perceived_role.encoder import load_tokenizer
from src.models.perceived_role.inference import make_predict_fn as encoder_predict_fn
from src.models.perceived_role.train import TrainConfig, train_one_run
from src.supervision import carriers
from src.supervision.transplant import build_all_supervision_sets

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "experiment" / "exp02b_lodo_carrier.yaml"
OUT_DIR = Path(__file__).resolve().parent
RESULTS_NAME = "results_b_carrier.json"
FOLD_META_NAME = "fold_meta_b_carrier.json"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def build_eval_set(test_records, n_eval: int, eval_seed: int):
    """Fixed sample of the held-out corpus plus a foreign-carrier rendering
    of each sampled span. Returns (records, wrapped_texts, wrap_channels).

    Mirrors how the E1 swap pairs are built (src/supervision/transplant.py,
    build_swap_test_set): the carrier channel is any channel other than the
    span's true origin. Deterministic in eval_seed, so every model in the
    fold is scored on identical inputs."""
    rng = random.Random(eval_seed)
    pool = list(test_records)
    rng.shuffle(pool)
    sample = pool[:n_eval]

    wrapped, channels = [], []
    for rec in sample:
        foreign = [c for c in carriers.available_channels() if c != rec.declared_role]
        channel = rng.choice(foreign)
        wrapped.append(carriers.embed_in_channel(rec.text, channel, rng))
        channels.append(channel.value)
    return sample, wrapped, channels


def evaluate_bare_and_wrapped(predict_fn, sample_records, wrapped_texts) -> dict:
    """Score one trained model on the bare and the wrapped rendering of the
    same spans. The held-out corpus is single-label by construction, so
    accuracy is recall for that one label. Per-span 0/1 correctness is kept
    so paired tests can be run later."""
    true_label = sample_records[0].declared_role
    bare_preds = predict_fn([r.text for r in sample_records])
    wrapped_preds = predict_fn(wrapped_texts)

    bare_correct = [int(p == true_label) for p in bare_preds]
    wrapped_correct = [int(p == true_label) for p in wrapped_preds]
    agree = [int(a == b) for a, b in zip(bare_preds, wrapped_preds)]
    n = len(sample_records)
    return {
        "true_label": true_label.value,
        "n": n,
        "bare_accuracy": sum(bare_correct) / n,
        "wrapped_accuracy": sum(wrapped_correct) / n,
        "bare_wrapped_agreement": sum(agree) / n,
        "bare_correct": bare_correct,
        "wrapped_correct": wrapped_correct,
    }


def summarize(results: list[dict]) -> dict:
    """Mean per (model, condition) of bare accuracy, wrapped accuracy and
    agreement. Seeds are averaged within a fold first, then folds are
    averaged, so a fold with more seeds does not get more weight."""
    per_fold: dict = defaultdict(lambda: defaultdict(list))
    for r in results:
        key = (r["model"], r["condition"], r["held_out_corpus"])
        for metric in ("bare_accuracy", "wrapped_accuracy", "bare_wrapped_agreement"):
            per_fold[key][metric].append(r[metric])

    per_cond: dict = defaultdict(lambda: defaultdict(list))
    for (model, cond, _fold), metrics in per_fold.items():
        for metric, vals in metrics.items():
            per_cond[(model, cond)][metric].append(sum(vals) / len(vals))

    out = {}
    for (model, cond), metrics in sorted(per_cond.items()):
        out[f"{model}|{cond}"] = {
            m: sum(v) / len(v) for m, v in metrics.items()
        } | {"n_folds": len(next(iter(metrics.values())))}
    return out


def run():
    cfg = load_config()
    print("=" * 70)
    print("EXPERIMENT 2b -- LODO with carrier-wrapped evaluation")
    print("=" * 70)
    print(f"Config: {CONFIG_PATH}")

    valid = set(cfg["valid_lodo_corpora"])
    all_records = list(read_records(REPO_ROOT / cfg["data"]["clean_spans_path"]))
    print(f"Loaded {len(all_records)} clean register-source spans", flush=True)

    sup, tr, ev = cfg["supervision"], cfg["training"], cfg["evaluation"]
    results, fold_meta = [], {}
    fold_count = 0

    for held_out, train_records, test_records in lodo_splits(all_records):
        if held_out not in valid:
            continue
        fold_count += 1
        print(f"\n{'=' * 70}\nFOLD {fold_count}: held out = {held_out} "
              f"(train={len(train_records)}, test={len(test_records)})\n{'=' * 70}", flush=True)

        sample, wrapped, channels = build_eval_set(test_records, ev["n_eval_per_fold"], ev["eval_seed"])
        fold_meta[held_out] = {
            "n_eval": len(sample),
            "record_ids": [r.record_id for r in sample],
            "wrap_channels": channels,
        }
        print(f"  eval sample: {len(sample)} spans, bare + foreign-wrapped", flush=True)

        sets = build_all_supervision_sets(
            train_records, transplant_rate=sup["transplant_rate"],
            n_swap_pairs_per_label=sup["n_swap_pairs_per_label"], seed=sup["seed"],
        )
        conditions = {"naive": sets["naive_control"], "transplant": sets["transplant_train"]}

        for cond_name, examples in conditions.items():
            pipe = train_tfidf_lr(examples, seed=42)
            m = evaluate_bare_and_wrapped(tfidf_predict_fn(pipe), sample, wrapped)
            print(f"  [tfidf_lr | {cond_name}] bare={m['bare_accuracy']:.3f} "
                  f"wrapped={m['wrapped_accuracy']:.3f} agree={m['bare_wrapped_agreement']:.3f}", flush=True)
            results.append({"model": "tfidf_lr", "held_out_corpus": held_out,
                            "condition": cond_name, "seed": 42, "n_train": len(examples), **m})

        for cond_name, examples in conditions.items():
            for seed in tr["seeds"]:
                train_ex, val_ex = stratified_subsample_and_split(
                    examples, tr["subsample_n"], tr["val_fraction"], seed=seed)
                tcfg = TrainConfig(batch_size=tr["batch_size"], n_epochs=tr["n_epochs"],
                                   seed=seed, max_length=tr["max_length"])
                print(f"\n  Training encoder | held_out={held_out} condition={cond_name} "
                      f"seed={seed} n_train={len(train_ex)} n_val={len(val_ex)}", flush=True)
                t0 = time.time()
                model, history = train_one_run(train_ex, val_ex, tcfg)
                elapsed = time.time() - t0
                tok = load_tokenizer(tcfg.backbone_name)
                pfn = encoder_predict_fn(model, tok, tcfg.device,
                                         max_length=tcfg.max_length, batch_size=tcfg.batch_size)
                m = evaluate_bare_and_wrapped(pfn, sample, wrapped)
                print(f"  [encoder | {cond_name} | seed={seed}] bare={m['bare_accuracy']:.3f} "
                      f"wrapped={m['wrapped_accuracy']:.3f} agree={m['bare_wrapped_agreement']:.3f} "
                      f"(train {elapsed:.0f}s)", flush=True)
                row_cfg = {"model": "deberta_v3_base_encoder", "held_out_corpus": held_out,
                           "condition": cond_name, "seed": seed, "n_train": len(train_ex),
                           "n_val": len(val_ex), "subsample_n": tr["subsample_n"]}
                results.append({**row_cfg, "config_hash": config_hash(row_cfg),
                                "train_history": history, "elapsed_sec": elapsed, **m})

        # Written after every fold so an interrupted multi-hour run keeps what it finished.
        with open(OUT_DIR / RESULTS_NAME, "w") as f:
            json.dump(results, f)
        with open(OUT_DIR / FOLD_META_NAME, "w") as f:
            json.dump(fold_meta, f)
        print(f"\n  [checkpoint] {len(results)} rows written", flush=True)

    summary = summarize(results)
    print("\n" + "=" * 70 + "\nSUMMARY (mean over folds; seeds averaged within each fold)\n" + "=" * 70)
    print(f"{'model | condition':<40}{'bare':<10}{'wrapped':<10}{'agreement'}")
    for key, s in summary.items():
        print(f"{key:<40}{s['bare_accuracy']:<10.3f}{s['wrapped_accuracy']:<10.3f}"
              f"{s['bare_wrapped_agreement']:.3f}")
    with open(OUT_DIR / "summary_b_carrier.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    run()
