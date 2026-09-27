"""
src/models/perceived_role/checkpoint.py

NEW glue: save/load a trained PerceivedRoleEncoder's state_dict + the config
it was trained with, so a checkpoint can be reused across experiments (E3's
discrepancy scoring, the eventual demo) instead of retraining from scratch
every time -- E1's runs were deliberately throwaway (in-memory only, for a
fast controlled comparison); this is what persists the encoder we'll
actually SCORE things with.
"""
# this is the checkpointing mechanism for the perceived-role encoder model, allowing it to be saved and loaded for future use without retraining.
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from src.models.perceived_role.encoder import PerceivedRoleEncoder
from src.models.perceived_role.train import TrainConfig


def save_checkpoint(model: PerceivedRoleEncoder, config: TrainConfig, out_dir: str | Path, extra: dict | None = None) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    meta = {"backbone_name": model.backbone_name, "config": asdict(config), "extra": extra or {}}
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_checkpoint(checkpoint_dir: str | Path, device: str = "cpu") -> tuple[PerceivedRoleEncoder, dict]:
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / "meta.json") as f:
        meta = json.load(f)
    model = PerceivedRoleEncoder(backbone_name=meta["backbone_name"])
    state = torch.load(checkpoint_dir / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, meta
