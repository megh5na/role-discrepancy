"""
src/models/perceived_role/train.py

EXISTING training loop pattern (Section 6: "Fine-tuning procedure ...
EXISTING"), wired to OUR supervision construction and label space.
Architecture component: C4/C9 (Section 8).

Trains a PerceivedRoleEncoder on a list of TrainingExample (from
src/supervision/transplant.py) with hard cross-entropy loss. Multi-seed by
construction (`train_one_run(seed=...)` is the unit CT9's spec calls for --
"Fomin reports high seed variance in this exact setting; single-seed results
prove nothing", Section 8 C9).

*** ABSOLUTE PROHIBITION A enforced here too ***: the Dataset class exposes
ONLY (text, label) to the model. `position_channel`, `source_dataset` etc.
are read for bookkeeping/eval but never tokenised or passed to forward().

Per-batch delimiter assertion (Section 27 rule 10 / Section 8 C2: "assert on
every batch during training") -- cheap regex check, run every batch, not
just at data-build time, so a bug anywhere upstream fails loudly during
training rather than silently producing an invalid result.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.ingestion.schema import ROLE_TO_IDX
from src.models.perceived_role.encoder import PerceivedRoleEncoder, load_tokenizer
from src.preprocessing.delimiters import assert_clean_batch
from src.supervision.types import TrainingExample


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


class RoleDataset(Dataset):
    """Wraps TrainingExample list. __getitem__ returns ONLY (text, label_idx)
    -- see module docstring, Absolute Prohibition A."""

    def __init__(self, examples: list[TrainingExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        ex = self.examples[idx]
        return ex.text, ROLE_TO_IDX[ex.label]


def make_collate_fn(tokenizer, max_length: int = 64):
    def collate(batch: list[tuple[str, int]]):
        texts = [t for t, _ in batch]
        labels = torch.tensor([l for _, l in batch], dtype=torch.long)
        assert_clean_batch(texts)  # SECURITY-CRITICAL, every batch, not just at build time
        enc = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        )
        return enc, labels

    return collate


@dataclass
class TrainConfig:
    backbone_name: str = "microsoft/deberta-v3-base"
    max_length: int = 32
    batch_size: int = 8
    lr: float = 2e-5 # learning rate - conventionally small for DeBERTa-v3-base, which is already pretrained
    n_epochs: int = 2
    seed: int = 42
    # DEVICE, DISCLOSED CONSTRAINT (docs/decisions.md / docs/lab_notebook.md):
    # MPS was measured faster per-step in isolation, but on THIS machine (8GB
    # total RAM, shared with the IDE / other apps) MPS wires
    # ~3.5GB for the Metal driver, which pushed the whole system into
    # memory-pressure thrashing (215M+ page translation faults, effective
    # near-0% useful CPU) during a real background run -- not a code bug,
    # confirmed by killing the process and watching wired memory drop by
    # that same ~3.5GB. CPU avoids this entirely and was verified stable
    # (memory flat, ~9.8 examples/sec at bs=8/len=32) under real system
    # load. Defaulting to CPU here is a measured fix, not a guess.
    device: str = "cpu"
    log_every: int = 20


def train_one_run(
    train_examples: list[TrainingExample],
    val_examples: list[TrainingExample] | None,
    config: TrainConfig,
) -> tuple[PerceivedRoleEncoder, dict]:
    """Trains one model on `train_examples` with a fixed seed. Returns the
    trained model and a small history dict (loss curve, val accuracy per
    epoch if val_examples given). One call of this function = one point in
    the multi-seed grid (Section 8 C9)."""
    set_seed(config.seed)

    tokenizer = load_tokenizer(config.backbone_name)
    model = PerceivedRoleEncoder(backbone_name=config.backbone_name)
    model.to(config.device)

    train_ds = RoleDataset(train_examples)
    collate = make_collate_fn(tokenizer, config.max_length)
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate
    )

    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)

    history = {"train_loss": [], "val_acc": []}

    model.train()
    step = 0
    t_start = time.time()
    for epoch in range(config.n_epochs):
        epoch_losses = []
        for enc, labels in train_loader:
            enc = {k: v.to(config.device) for k, v in enc.items()}
            labels = labels.to(config.device)

            opt.zero_grad()
            logits = model(enc["input_ids"], enc["attention_mask"])
            loss = nn.functional.cross_entropy(logits, labels) # loss
            loss.backward()
            opt.step()

            epoch_losses.append(loss.item())
            step += 1
            if step % config.log_every == 0:
                elapsed = time.time() - t_start
                print(
                    f"  seed={config.seed} epoch={epoch} step={step} "
                    f"loss={np.mean(epoch_losses[-config.log_every:]):.4f} "
                    f"elapsed={elapsed:.0f}s"
                )

        history["train_loss"].append(float(np.mean(epoch_losses)))

        if val_examples:
            acc = evaluate_accuracy(model, tokenizer, val_examples, config)
            history["val_acc"].append(acc)
            print(f"  seed={config.seed} epoch={epoch} val_acc={acc:.4f}")
            model.train()

    return model, history


@torch.no_grad()
def evaluate_accuracy( # runs after each epoch if val_examples is provided, or can be called standalone for a final evaluation
    model: PerceivedRoleEncoder, tokenizer, examples: list[TrainingExample], config: TrainConfig
) -> float:
    model.eval()
    ds = RoleDataset(examples)
    collate = make_collate_fn(tokenizer, config.max_length)
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate)
    correct, total = 0, 0
    for enc, labels in loader:
        enc = {k: v.to(config.device) for k, v in enc.items()}
        labels = labels.to(config.device)
        logits = model(enc["input_ids"], enc["attention_mask"])
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return correct / total if total else 0.0
