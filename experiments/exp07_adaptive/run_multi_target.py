from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import yaml

from src.attacks.asr_harness import (
    CallBudget,
    _build_agent_prompt,
    _judge_success,
    _TARGET_SYSTEM_PROMPT,
)
from src.attacks.destyle import destyle_instruction, extract_text, make_client
from src.ingestion.io_utils import read_records
from src.preprocessing.delimiters import strip_delimiters

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CK = REPO_ROOT / "checkpoints"
MODELS = {
    "naive_s42": CK / "perceived_role_v1_naive",
    "transplant_s42": CK / "perceived_role_v1",
    "naive_s123": CK / "kept_seed_models" / "perceived_role_v1_naive_seed123",
    "transplant_s123": CK / "kept_seed_models" / "perceived_role_v1_seed123",
    "naive_s456": CK / "kept_seed_models" / "perceived_role_v1_naive_seed456",
    "transplant_s456": CK / "kept_seed_models" / "perceived_role_v1_seed456",
}


def load_unique_attacks(cfg: dict, n_total: int) -> list:
    """Unique injected strings, stratified evenly over injection_type, deterministic (seed)."""
    src = cfg["attack_source"]
    first_by_text: dict = {}
    for r in read_records(REPO_ROOT / src["path"]):
        if r.is_injected and r.text not in first_by_text:
            first_by_text[r.text] = r
    by_cat = defaultdict(list)
    for r in first_by_text.values():
        by_cat[r.injection_type].append(r)
    cats = sorted(by_cat)
    per_cat = max(1, n_total // len(cats))
    rng = random.Random(src["seed"])
    sample = []
    for c in cats:
        pool = sorted(by_cat[c], key=lambda r: r.record_id)
        rng.shuffle(pool)
        sample.extend(pool[:per_cat])
    return sample[:n_total]


def target_response(prov: str, client, model: str, prompt: str, usage: dict) -> str:
    """Same system prompt and 512-token cap as the Anthropic pilot, for every provider."""
    if prov == "anthropic":
        resp = client.messages.create(model=model, max_tokens=512, system=_TARGET_SYSTEM_PROMPT,
                                      messages=[{"role": "user", "content": prompt}])
        u = resp.usage
        usage["in"] += u.input_tokens; usage["out"] += u.output_tokens
        return extract_text(resp)
    resp = client.chat.completions.create(
        model=model, max_completion_tokens=512,
        messages=[{"role": "system", "content": _TARGET_SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
    usage["in"] += resp.usage.prompt_tokens; usage["out"] += resp.usage.completion_tokens
    return (resp.choices[0].message.content or "").strip()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def build_variants(sample, levels, clients, mt, budget, vpath: Path):
    """Destyle once per (attack, level>0); score locally with all six checkpoints; append as done."""
    done = {(v["record_id"], v["destyling_level"]) for v in read_jsonl(vpath)}
    todo = [(r, lv) for r in sample for lv in levels if (r.record_id, lv) not in done]
    if not todo:
        return
    from experiments.ablation_severity.run_severity_ablation import make_prob_scorer
    scorers = {n: make_prob_scorer(p) for n, p in MODELS.items()}
    rclient = clients[mt["rewriter_provider"]]
    with vpath.open("a") as f:
        for r, lv in todo:
            if lv > 0.0:
                variant = destyle_instruction(rclient, r.text, lv, mt["rewriter_model"])
                budget.spend(1)
            else:
                variant = r.text
            clean = strip_delimiters(variant)
            probs = {n: fn([clean])[0].tolist() for n, fn in scorers.items()}
            f.write(json.dumps({"record_id": r.record_id, "injection_type": r.injection_type,
                                "destyling_level": lv, "original_text": r.text, "variant_text": variant,
                                "probs": probs}) + "\n"); f.flush()
            print(f"  variant {r.injection_type:15s} level={lv}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--targets", default="", help="comma-separated target names to run (default: all in config)")
    args = ap.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "experiment" / "exp07_adaptive.yaml").read_text())
    mt = cfg["multi_target"]
    n = args.smoke or mt["n_attacks"]
    tag = "mt_smoke" if args.smoke else "mt"
    levels = cfg["destyling"]["levels"]
    sample = load_unique_attacks(cfg, n)
    targets = mt["targets"]
    if args.targets:
        names = set(args.targets.split(","))
        unknown = names - {x["name"] for x in targets}
        if unknown:
            raise SystemExit(f"unknown targets {sorted(unknown)}")
        targets = [x for x in targets if x["name"] in names]
    n_variants_calls = len(sample) * sum(1 for lv in levels if lv > 0)
    n_target_calls = len(sample) * len(levels) * len(targets) * 2  # target + judge
    done_target_calls = 2 * len(read_jsonl(HERE / f"{tag}_results.jsonl"))
    need = n_variants_calls + n_target_calls  # whole planned set; resume reconstructs usage below
    print(f"{tag}: {len(sample)} unique attacks x {len(levels)} levels x {len(targets)} targets; "
          f"planned API calls {need} (cap {mt['max_total_calls']}, {done_target_calls} already made)", flush=True)
    if need > mt["max_total_calls"]:
        raise SystemExit("planned calls exceed max_total_calls; raise it deliberately in the config")

    provs = {mt["rewriter_provider"], mt["judge_provider"], *(t["provider"] for t in targets)}
    clients = {p: make_client(p) for p in provs}
    budget = CallBudget(mt["max_total_calls"])
    # a rerun resumes: count calls already made so the cap is for the whole experiment
    vpath, rpath, upath = HERE / f"{tag}_variants.jsonl", HERE / f"{tag}_results.jsonl", HERE / f"{tag}_usage.json"
    prior = read_jsonl(rpath)
    budget.used = sum(1 for _ in prior) * 2 + sum(1 for v in read_jsonl(vpath) if v["destyling_level"] > 0)

    build_variants(sample, levels, clients, mt, budget, vpath)
    variants = {(v["record_id"], v["destyling_level"]): v for v in read_jsonl(vpath)}
    done = {(r["target"], r["record_id"], r["destyling_level"]) for r in prior}
    usage = json.loads(upath.read_text()) if upath.exists() else {}
    jclient = clients[mt["judge_provider"]]
    t0 = time.time()
    with rpath.open("a") as f:
        for t in targets:
            u = usage.setdefault(t["name"], {"in": 0, "out": 0})
            for r in sample:
                for lv in levels:
                    if (t["name"], r.record_id, lv) in done:
                        continue
                    v = variants[(r.record_id, lv)]
                    resp = target_response(t["provider"], clients[t["provider"]], t["model"],
                                           _build_agent_prompt(v["variant_text"]), u)
                    budget.spend(1)
                    ok = _judge_success(jclient, r.text, resp, mt["judge_model"])
                    budget.spend(1)
                    f.write(json.dumps({"target": t["name"], "record_id": r.record_id,
                                        "injection_type": r.injection_type, "destyling_level": lv,
                                        "target_response": resp, "attack_succeeded": ok}) + "\n"); f.flush()
                    upath.write_text(json.dumps(usage, indent=1))
                    print(f"  {t['name']:16s} {r.injection_type:15s} level={lv} succeeded={ok}", flush=True)
    print(f"\nDONE {tag}: API calls used {budget.used}/{budget.max_calls}; {time.time() - t0:.0f}s; "
          f"token usage per target {json.dumps(usage)}", flush=True)


if __name__ == "__main__":
    main()
