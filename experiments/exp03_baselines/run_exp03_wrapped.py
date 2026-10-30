from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from src.ingestion.schema import Role
from src.scoring.asymmetry import combined_score
from src.scoring.calibration import calibrate_threshold
from src.supervision import carriers

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
CHECKPOINTS = {
    "ours_transplant": REPO_ROOT / "checkpoints" / "perceived_role_v1",
    "ours_naive": REPO_ROOT / "checkpoints" / "perceived_role_v1_naive",
}
FPRS = [0.01, 0.05]
N_BOOT = 1000
WRAP_SEED = 11

NOVEL_TEMPLATES = {
    "markdown_quote": "> {span}",
    "xml_message": "<message>\n{span}\n</message>",
    "email": "Subject: Re: update\n\nHi,\n\n{span}\n\nBest regards",
    "log_line": "[2026-03-02 10:14:07] INFO retrieved: {span}",
    "citation": "Source [3]: {span} (accessed 2026-02-11)",
    "yaml": "content: |\n  {span}",
}


def unique_in_order(texts: list[str]) -> list[str]:
    seen, out = set(), []
    for t in texts:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def build_conditions(texts: list[str], seed: int = WRAP_SEED) -> dict[str, list[str]]:
    """condition name -> wrapped version of every text (same order). Each
    condition has its own seeded rng, so the wrapped inputs are identical
    across detectors and across re-runs."""
    conds: dict[str, list[str]] = {"bare": list(texts)}
    for ch in carriers.available_channels():
        rng = random.Random(f"{seed}-known-{ch.value}")
        conds[f"known_{ch.value}"] = [carriers.embed_in_channel(t, ch, rng) for t in texts]
    for name, tpl in NOVEL_TEMPLATES.items():
        conds[f"novel_{name}"] = [tpl.format(span=t) for t in texts]
    return conds


def make_scorer(ckpt_dir: Path):
    """Load a checkpoint once; return texts -> discrepancy scores. Same
    batching and max_length (16 / 64) as the earlier E3 scoring."""
    import torch

    from src.models.perceived_role.checkpoint import load_checkpoint
    from src.models.perceived_role.encoder import load_tokenizer

    model, meta = load_checkpoint(ckpt_dir)
    tok = load_tokenizer(meta["backbone_name"])

    def score(texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 16):
            enc = tok(texts[i:i + 16], return_tensors="pt", padding=True, truncation=True, max_length=64)
            with torch.no_grad():
                probs = torch.softmax(model(enc["input_ids"], enc["attention_mask"]), dim=-1).numpy()
            out += [combined_score(p, Role.DOCUMENT, use_asymmetry=True, severity_mode="argmax")["score"] for p in probs]
        return np.array(out)

    return score


def metrics_for(bare_b, ben_c, inj_c, fpr):
    """One condition, one resample. All arrays already indexed to the resample."""
    thr_a = calibrate_threshold(bare_b, fpr)
    thr_b = calibrate_threshold(ben_c, fpr)
    y = np.r_[np.zeros(len(ben_c)), np.ones(len(inj_c))]
    return {
        "A_tpr": float(np.mean(inj_c >= thr_a)),
        "A_fpr_on_wrapped_benign": float(np.mean(ben_c >= thr_a)),
        "B_tpr": float(np.mean(inj_c >= thr_b)),
        "B_auroc": float(roc_auc_score(y, np.r_[ben_c, inj_c])),
    }


def groups_of(cond_names):
    return {
        "known_avg": [c for c in cond_names if c.startswith("known_")],
        "novel_avg": [c for c in cond_names if c.startswith("novel_")],
    }


def one_pass(scores, cond_names, bi, ii):
    """Every metric for every detector / condition / group on one resample
    (bi, ii index the unique benign / injected strings). Returns
    {(detector, unit, fpr, metric): value}."""
    groups = groups_of(cond_names)
    out = {}
    for d in scores:
        bare_b = scores[d]["bare"][0][bi]
        per_cond = {}
        for c in cond_names:
            ben, inj = scores[d][c]
            for f in FPRS:
                per_cond[(c, f)] = metrics_for(bare_b, ben[bi], inj[ii], f)
        for (c, f), m in per_cond.items():
            for k, v in m.items():
                out[(d, c, f, k)] = v
        for g, members in groups.items():
            for f in FPRS:
                for k in per_cond[(members[0], f)]:
                    out[(d, g, f, k)] = float(np.mean([per_cond[(c, f)][k] for c in members]))
    return out


def bootstrap(scores, cond_names, n_boot=N_BOOT, seed=42):
    """scores[det][cond] = (benign_arr, injected_arr), aligned to the unique-
    string lists. Returns (point, draws): point on the full data, and
    draws[key] = array of n_boot resampled values. The same resample is used
    for every detector and condition, so differences are paired."""
    rng = np.random.default_rng(seed)
    dets = list(scores)
    n_b = len(scores[dets[0]]["bare"][0])
    n_i = len(scores[dets[0]]["bare"][1])
    point = one_pass(scores, cond_names, np.arange(n_b), np.arange(n_i))
    acc: dict = {k: [] for k in point}
    for _ in range(n_boot):
        res = one_pass(scores, cond_names, rng.integers(0, n_b, n_b), rng.integers(0, n_i, n_i))
        for k, v in res.items():
            acc[k].append(v)
    return point, {k: np.array(v) for k, v in acc.items()}


def ci(x):
    lo, hi = np.quantile(x, [0.025, 0.975])
    return [round(float(lo), 3), round(float(hi), 3)]


def summarize(point, draws):
    """Point estimate (full data) plus 95% bootstrap interval for every
    detector / unit / FPR / metric, and paired transplant-minus-naive
    differences."""
    out = {"detectors": {}, "paired_transplant_minus_naive": {}}
    for (d, unit, f, k), v in draws.items():
        out["detectors"].setdefault(d, {}).setdefault(unit, {}).setdefault(str(f), {})[k] = {
            "point": round(point[(d, unit, f, k)], 3), "ci95": ci(v)}
    for (d, unit, f, k), v in draws.items():
        if d != "ours_transplant":
            continue
        diff = v - draws[("ours_naive", unit, f, k)]
        lo, hi = ci(diff)
        out["paired_transplant_minus_naive"].setdefault(unit, {}).setdefault(str(f), {})[k] = {
            "point": round(point[(d, unit, f, k)] - point[("ours_naive", unit, f, k)], 3),
            "ci95": [lo, hi], "ci_excludes_zero": bool(lo > 0 or hi < 0)}
    return out


def run():
    from experiments.exp03_baselines.run_exp03_naive import load_spans

    benign_all, injected_all = load_spans()
    benign, injected = unique_in_order(benign_all), unique_in_order(injected_all)
    print(f"unique benign={len(benign)} unique injected={len(injected)}", flush=True)
    ben_c, inj_c = build_conditions(benign), build_conditions(injected)
    cond_names = list(ben_c)
    print(f"{len(cond_names)} conditions: {cond_names}", flush=True)

    scores: dict = {}
    for name, ckpt in CHECKPOINTS.items():
        if not (ckpt / "model.pt").exists():
            raise SystemExit(f"missing checkpoint {ckpt}")
        print(f"\nScoring {name} ...", flush=True)
        score = make_scorer(ckpt)
        scores[name] = {c: (score(ben_c[c]), score(inj_c[c])) for c in cond_names}
        print(f"  done {name}", flush=True)

    np.savez(OUT_DIR / "scores_exp03_wrapped.npz",
             **{f"{d}__{c}__{k}": arr for d, per in scores.items()
                for c, (b, i) in per.items() for k, arr in (("benign", b), ("injected", i))})

    print("\nBootstrapping over unique strings ...", flush=True)
    point, draws = bootstrap(scores, cond_names)
    result = summarize(point, draws)
    result["n_unique_benign"], result["n_unique_injected"] = len(benign), len(injected)
    result["conditions"] = cond_names
    with open(OUT_DIR / "results_exp03_wrapped.json", "w") as f:
        json.dump(result, f, indent=2)

    f5 = "0.05"
    print("\n" + "=" * 78 + "\n5% FPR budget; point estimate [95% CI over unique-string bootstrap]\n" + "=" * 78)
    for unit in ["bare", "known_avg", "novel_avg"]:
        print(f"\n[{unit}]")
        for d in ("ours_transplant", "ours_naive"):
            r = result["detectors"][d][unit][f5]
            print(f"  {d:<17} A: TPR {r['A_tpr']['point']:.3f} {r['A_tpr']['ci95']}  "
                  f"benign FPR {r['A_fpr_on_wrapped_benign']['point']:.3f}   "
                  f"B: TPR {r['B_tpr']['point']:.3f}  AUROC {r['B_auroc']['point']:.3f} {r['B_auroc']['ci95']}")
        p = result["paired_transplant_minus_naive"][unit][f5]
        print(f"  paired transplant-naive  A_tpr {p['A_tpr']['point']:+.3f} {p['A_tpr']['ci95']}   "
              f"B_auroc {p['B_auroc']['point']:+.3f} {p['B_auroc']['ci95']}")
    return result


if __name__ == "__main__":
    run()
