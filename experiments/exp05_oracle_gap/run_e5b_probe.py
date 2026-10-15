"""
experiments/exp05_oracle_gap/run_e5b_probe.py

E5b: white-box register probe and the oracle gap. Design, validity guard and interpretation rule were pre-registered in docs/decisions.md
("PRE-REGISTRATION: E5b ...") before this script was written or run.

NOT a reproduction of the prior work: the probe is trained on REGISTER-labelled text (the project's own naive_control and transplant_train sets, no role tags) and
used as a small-model oracle for how well a white-box linear reader recovers register, next to our black-box DeBERTa encoder.

EXISTING (reused): sidecar model loading, all-layer mean-pooled activation extraction, the logistic-regression probe, the 5,686 / 631 train/validation split recipe
(split seed 7, as the encoders), the 600 swap pairs, the E1 re-run as the encoder reference.
NEW: this driver. Layer is chosen by VALIDATION accuracy only, never on the swap pairs. Evaluation is the E1 metric trio (native acc, foreign acc, swap consistency).
Interpretation rule (fixed before running): per training condition, probe minus encoder mean on swap consistency (and the same rule on foreign accuracy):
>= +0.05 oracle ahead; within +/-0.05 no oracle advantage; <= -0.05 encoder ahead. Guard: validation accuracy at the selected layer >= 0.60, else the gap is undefined.
SECONDARY (descriptive): probe on the conflict set (reasoning-register content under <user> tags; expected to follow content BY CONSTRUCTION, not evidence about the
prior work); probe-based detection AUROC on BIPIA-only negatives next to the encoder ensembles under the same scorer.

Usage: python -m experiments.exp05_oracle_gap.run_e5b_probe [model_name] [tag]     (default SmolLM2-360M-Instruct, tag 360m)
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

from experiments.exp01_swap_validity.run_exp01 import load_jsonl
from experiments.exp03_baselines.run_exp03_naive import load_spans
from experiments.exp03_baselines.run_exp03_wrapped import unique_in_order
from src.ingestion.schema import ROLE_ORDER, ROLE_TO_IDX, Role
from src.models.sidecar.hooks import load_sidecar_model, n_layers
from src.models.sidecar.layer_sweep import extract_all_layers
from src.models.sidecar.probe import probe_readout, train_probe
from src.models.sidecar.validate_native_template import load_register_content
from src.models.sidecar.wrapper_construction import wrap
from src.scoring.asymmetry import expected_severity
from src.supervision.types import SwapPairItem, TrainingExample

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PROCESSED = REPO_ROOT / "data" / "processed"
ABL = REPO_ROOT / "experiments" / "ablation_severity"
SEEDS = (42, 123, 456)
GUARD = 0.60
BAND = 0.05
CHUNK = 256


def split(examples):
    rng = random.Random(7)                      # same split recipe as both encoder checkpoints
    sh = examples[:]
    rng.shuffle(sh)
    n_val = int(len(sh) * 0.1)
    return sh[n_val:], sh[:n_val]               # train, val


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))


def classify(gap: float) -> str:
    return "oracle ahead (white-box probe more robust)" if gap >= BAND else "encoder ahead" if gap <= -BAND else "no oracle advantage (black-box matches white-box)"


def main(model_name: str, tag: str):
    t0 = time.time()
    naive_tr, naive_va = split(load_jsonl(PROCESSED / "naive_control.jsonl", TrainingExample))
    trans_tr, trans_va = split(load_jsonl(PROCESSED / "transplant_train.jsonl", TrainingExample))
    swaps = load_jsonl(PROCESSED / "swap_test_pairs.jsonl", SwapPairItem)
    benign, injected = load_spans()
    benign_u, injected_u = unique_in_order(benign), unique_in_order(injected)
    conflict_texts = [wrap(t, Role.USER) for t in load_register_content(Role.REASONING, 60)]
    print(f"naive {len(naive_tr)}/{len(naive_va)}  transplant {len(trans_tr)}/{len(trans_va)}  swaps {len(swaps)}  bipia {len(benign_u)}+{len(injected_u)}  conflict {len(conflict_texts)}", flush=True)

    groups = {"naive_tr": [e.text for e in naive_tr], "naive_va": [e.text for e in naive_va], "trans_tr": [e.text for e in trans_tr], "trans_va": [e.text for e in trans_va],
              "swap_native": [p.native_text for p in swaps], "swap_foreign": [p.foreign_text for p in swaps], "bipia_benign": benign_u, "bipia_injected": injected_u,
              "conflict": conflict_texts}
    uniq = list(dict.fromkeys(t for g in groups.values() for t in g))
    pos = {t: i for i, t in enumerate(uniq)}
    print(f"{len(uniq)} unique texts to embed", flush=True)

    model, tok = load_sidecar_model(model_name)
    L = n_layers(model) + 1
    hidden = model.config.hidden_size
    acts = np.zeros((L, len(uniq), hidden), dtype=np.float16)
    for s in range(0, len(uniq), CHUNK):
        layers = extract_all_layers(model, tok, uniq[s:s + CHUNK])
        for li, t in enumerate(layers):
            acts[li, s:s + len(t)] = t.numpy().astype(np.float16)
        if (s // CHUNK) % 5 == 0:
            print(f"  embedded {min(s + CHUNK, len(uniq))}/{len(uniq)}  {time.time() - t0:.0f}s", flush=True)
    del model
    idx = lambda name: np.array([pos[t] for t in groups[name]])  # noqa: E731
    A = lambda li, name: torch.from_numpy(acts[li][idx(name)].astype(np.float32))  # noqa: E731
    y = lambda ex: [e.label for e in ex]  # noqa: E731
    true = lambda ex: np.array([ROLE_ORDER.index(e.label) for e in ex])  # noqa: E731
    swap_labels = np.array([ROLE_ORDER.index(p.label) for p in swaps])

    enc = json.loads((HERE.parent / "exp01_swap_validity" / "results_kept_seeds.json").read_text())["per_model"]
    ref = {c: {k: float(np.mean([enc[f"{c}_s{s}"][k] for s in SEEDS])) for k in ("swap_consistency", "native_acc", "foreign_acc")} for c in ("naive", "transplant")}

    # encoder ensembles under the default scorer on the same BIPIA unique strings (for the secondary comparison)
    def enc_auroc(c):
        sc = {}
        for key in ("bare__benign", "bare__injected"):
            sc[key] = np.mean([[expected_severity(p, Role.DOCUMENT) for p in np.load(ABL / f"probs_{c}_s{s}.npz")[key]] for s in SEEDS], axis=0)
        return auroc(sc["bare__injected"], sc["bare__benign"])
    enc_det = {c: enc_auroc(c) for c in ("naive", "transplant")}

    out = {"model": model_name, "n_layers_incl_embedding": L, "hidden": hidden, "encoder_reference_3_seed_mean": ref, "guard": GUARD, "band": BAND, "conditions": {}}
    for cond, (tr, va, trg, vag) in {"naive": (naive_tr, naive_va, "naive_tr", "naive_va"), "transplant": (trans_tr, trans_va, "trans_tr", "trans_va")}.items():
        per_layer, clfs = [], {}
        for li in range(L):
            tt = time.time()
            clf = train_probe(A(li, trg), y(tr), seed=42)
            acc = float((clf.predict(A(li, vag).numpy()) == true(va)).mean())
            per_layer.append(acc); clfs[li] = clf
            print(f"  [{cond}] layer {li:2d}  val acc {acc:.3f}  ({time.time() - tt:.0f}s)", flush=True)
        best = int(np.argmax(per_layer)); clf = clfs[best]            # selected by validation accuracy only; first maximum on ties
        nat = clf.predict(A(best, "swap_native").numpy()); frn = clf.predict(A(best, "swap_foreign").numpy())
        m = {"swap_consistency": float((nat == frn).mean()), "native_acc": float((nat == swap_labels).mean()), "foreign_acc": float((frn == swap_labels).mean())}
        gaps = {k: m[k] - ref[cond][k] for k in m}
        guard_ok = per_layer[best] >= GUARD
        ro = probe_readout(clf, A(best, "conflict"))
        pb = {"benign": probe_readout(clf, A(best, "bipia_benign")), "injected": probe_readout(clf, A(best, "bipia_injected"))}
        det = auroc(np.array([expected_severity(p, Role.DOCUMENT) for p in pb["injected"]]), np.array([expected_severity(p, Role.DOCUMENT) for p in pb["benign"]]))
        out["conditions"][cond] = {
            "val_acc_by_layer": [round(a, 4) for a in per_layer], "selected_layer": best, "val_acc_selected": round(per_layer[best], 4), "guard_passed": bool(guard_ok),
            "probe_swap_metrics": {k: round(v, 4) for k, v in m.items()}, "gap_probe_minus_encoder": {k: round(v, 4) for k, v in gaps.items()},
            "interpretation_swap_consistency": classify(gaps["swap_consistency"]) if guard_ok else "undefined: probe cannot read register here (guard failed)",
            "interpretation_foreign_acc": classify(gaps["foreign_acc"]) if guard_ok else "undefined (guard failed)",
            "secondary_conflict_reasoning_under_user_tag": {"mean_P_reasoning": round(float(ro[:, ROLE_TO_IDX[Role.REASONING]].mean()), 3), "mean_P_user": round(float(ro[:, ROLE_TO_IDX[Role.USER]].mean()), 3),
                                                            "caveat": "probe trained without tags: following content is expected by construction"},
            "secondary_bipia_auroc_probe_vs_encoder_ensemble": {"probe": round(det, 3), "encoder_ensemble_same_scorer": round(enc_det[cond], 3)},
        }
        print(f"[{cond}] selected layer {best} val {per_layer[best]:.3f} | swap {m} | gap {({k: round(v, 3) for k, v in gaps.items()})} | {out['conditions'][cond]['interpretation_swap_consistency']}", flush=True)
    out["elapsed_sec"] = round(time.time() - t0)
    (HERE / f"e5b_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"DONE {tag} in {out['elapsed_sec']}s", flush=True)


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolLM2-360M-Instruct"
    main(name, sys.argv[2] if len(sys.argv) > 2 else "360m")
