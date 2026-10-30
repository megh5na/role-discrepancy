from __future__ import annotations

import torch

from src.ingestion.schema import IDX_TO_ROLE, Role
from src.models.perceived_role.encoder import PerceivedRoleEncoder
from src.preprocessing.delimiters import assert_clean_batch


def make_predict_fn(model: PerceivedRoleEncoder, tokenizer, device: str, max_length: int = 64, batch_size: int = 32): # given a trained model and a tokenizer, it returns a function that takes a list of texts and returns a list of predicted roles for those texts. The function handles batching, tokenization, and moving data to the correct device (CPU or GPU) for inference. It also ensures that the input texts are clean and free of any unwanted characters or formatting issues before making predictions.
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
