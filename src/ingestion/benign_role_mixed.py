"""
src/ingestion/benign_role_mixed.py

Builds the Category C corpus (DataCategory.BENIGN_ROLE_MIXED) used by
Experiment 6 (over-defense): benign text whose natural register differs from
the channel it legitimately sits in, e.g. a user pasting a Wikipedia
paragraph into their own message, or a tool returning a snippet of prose.

EXISTING data (register-source spans), NEW construction. All cell
definitions, source mappings and frame wordings live in
configs/data/benign_role_mixed.yaml so they can be read and edited in one
place; read the comments there for what "benign" does and does not mean here.

Guarantees enforced here (and by tests/test_benign_role_mixed.py):
  * every content span was NEVER used in training or swap-test construction
    (record ids checked against data/processed/*.jsonl; missing files raise,
    they are not skipped);
  * no span or final text is reused across cells;
  * deterministic: same config + same inputs -> same corpus, independent of
    the order records appear in the input file;
  * is_injected is False for every record.

Usage:
    python -m src.ingestion.benign_role_mixed
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import yaml

from src.ingestion.io_utils import read_records, write_records
from src.ingestion.schema import DataCategory, Record, Role

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "data" / "benign_role_mixed.yaml"
CLEAN_SPANS_PATH = REPO_ROOT / "data" / "interim" / "clean_spans_register.jsonl"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
OUT_PATH = REPO_ROOT / "data" / "interim" / "benign_role_mixed.jsonl"
MANIFEST_PATH = REPO_ROOT / "data" / "interim" / "benign_role_mixed_manifest.json"
SOURCE_NAME = "benign_role_mixed"
PLACEHOLDER = "{span}"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def training_used_ids(processed_dir: Path = PROCESSED_DIR) -> set[str]:
    """origin_record_id of every span that went into a training set or the
    swap-test pool. Raises if any file is missing: a silent skip here would
    let training data leak into the over-defense evaluation."""
    used: set[str] = set()
    for name in ("naive_control.jsonl", "transplant_train.jsonl", "swap_test_pairs.jsonl"):
        path = processed_dir / name
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; cannot verify the corpus is disjoint from training data")
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    used.add(json.loads(line)["origin_record_id"])
    return used


def candidate_pools(spans: list[Record], cfg: dict, used_ids: set[str]) -> dict[str, list[Record]]:
    """register -> unique-text candidate spans from the configured sources that
    were never used in training and fall inside the word-length bounds."""
    lo, hi = cfg["min_words"], cfg["max_words"]
    pools: dict[str, list[Record]] = {}
    for register, sources in cfg["sources_by_register"].items():
        allowed = set(sources)
        seen_text: set[str] = set()
        pool = []
        # sort for determinism that does not depend on input file order
        for r in sorted((r for r in spans if r.source_dataset in allowed), key=lambda r: r.record_id):
            if r.record_id in used_ids or not (lo <= len(r.text.split()) <= hi):
                continue
            if r.text in seen_text:
                continue
            seen_text.add(r.text)
            pool.append(r)
        pools[register] = pool
    return pools


def apply_frame(span_text: str, frame: str | None, quote_swap: bool = False) -> str:
    if frame is None:
        return span_text
    text = span_text.replace('"', "'") if quote_swap else span_text
    return frame.replace(PLACEHOLDER, text)


def build(spans: list[Record], cfg: dict, used_ids: set[str]) -> tuple[list[Record], dict]:
    pools = candidate_pools(spans, cfg, used_ids)
    assigned_ids: set[str] = set()
    out: list[Record] = []
    cell_stats = []

    for cell in cfg["cells"]:
        n = int(cell.get("n", cfg["default_n_per_cell"]))
        pool = [r for r in pools[cell["content"]] if r.record_id not in assigned_ids]
        if len(pool) < n:
            raise ValueError(f"cell {cell['name']}: need {n} spans of register {cell['content']!r}, "
                             f"only {len(pool)} unused candidates remain")
        rng = random.Random(f"{cfg['seed']}-{cell['name']}")
        rng.shuffle(pool)
        chosen = pool[:n]
        frames = None if cell.get("control") else cell["frames"]
        for i, span in enumerate(chosen):
            assigned_ids.add(span.record_id)
            frame_id = None if frames is None else i % len(frames)
            text = apply_frame(span.text, None if frames is None else frames[frame_id],
                               quote_swap=bool(cell.get("quote_swap")))
            out.append(Record(
                record_id=Record.make_id(SOURCE_NAME, text, len(out)),
                text=text,
                declared_role=Role(cell["declared"]),
                source_dataset=SOURCE_NAME,
                category=DataCategory.BENIGN_ROLE_MIXED,
                is_injected=False,
                injection_type=None,
                meta={"cell": cell["name"], "control": bool(cell.get("control")),
                      "content_register": cell["content"], "frame_id": frame_id,
                      "source_dataset": span.source_dataset, "source_record_id": span.record_id},
            ))
        cell_stats.append({
            "cell": cell["name"], "declared": cell["declared"], "content_register": cell["content"],
            "control": bool(cell.get("control")), "n": n,
            "n_frames": 0 if frames is None else len(frames),
            "source_datasets": dict(Counter(s.source_dataset for s in chosen)),
        })

    manifest = {
        "seed": cfg["seed"], "n_records": len(out), "n_cells": len(cfg["cells"]),
        "n_mixed": sum(not c["control"] for c in cell_stats),
        "n_control": sum(c["control"] for c in cell_stats),
        "by_declared_role": dict(Counter(r.declared_role.value for r in out)),
        "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16],
        "candidate_pool_sizes": {k: len(v) for k, v in pools.items()},
        "cells": cell_stats,
    }
    return out, manifest


def main():
    cfg = load_config()
    spans = list(read_records(CLEAN_SPANS_PATH))
    used = training_used_ids()
    records, manifest = build(spans, cfg, used)
    n = write_records(records, OUT_PATH)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {n} records -> {OUT_PATH}")
    print(f"Manifest -> {MANIFEST_PATH}")
    print(f"{manifest['n_mixed']} mixed cells + {manifest['n_control']} control cells; "
          f"by declared role: {manifest['by_declared_role']}")
    print(f"never-used candidate pool sizes: {manifest['candidate_pool_sizes']}")


if __name__ == "__main__":
    main()
