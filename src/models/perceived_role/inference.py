"""
src/models/perceived_role/inference.py

NEW glue: wraps a trained PerceivedRoleEncoder as a
`predict_fn: list[str] -> list[Role]`, the same interface the TF-IDF
baseline exposes (src/baselines/simple/tfidf_baseline.py), so
src/evaluation/metrics.py can evaluate either one identically.
"""

from __future__ import annotations

import torch

from src.ingestion.schema import IDX_TO_ROLE, Role
from src.models.perceived_role.encoder import PerceivedRoleEncoder
from src.preprocessing.delimiters import assert_clean_batch


def make_predict_fn(model: PerceivedRoleEncoder, tokenizer, device: str, max_length: int = 64, batch_size: int = 32):
    model.eval()

    @torch.no_grad()
    def predict_fn(texts: list[str]) -> list[Role]:
        assert_clean_batch(texts)
        preds: list[Role] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
            ).to(device)
            logits = model(enc["input_ids"], enc["attention_mask"])
            idxs = logits.argmax(dim=-1).cpu().tolist()
            preds.extend(IDX_TO_ROLE[i] for i in idxs)
        return preds

    return predict_fn
