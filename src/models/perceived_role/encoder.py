"""
src/models/perceived_role/encoder.py

EXISTING architecture, UNMODIFIED (Section 11: "CONSTRAINT: DO NOT modify
the backbone architecture. Any claim of architectural novelty is false and
will be caught."). Backbone = DeBERTa-v3-base (resolved TO-BE-DECIDED), mean-pooling, one linear classification head -- all
off-the-shelf. The novelty of this project lives in what this model is
TRAINED ON (src/supervision/) and how its output is USED (src/scoring/), not
in this file.

Architecture component: C4 (Section 8).

*** ABSOLUTE PROHIBITION A (Section 27) ENFORCED HERE ***: `forward()` takes
ONLY input_ids/attention_mask produced from `text`. There is no code path in
this class that can accept a declared_role, a channel id, or a delimiter
token as a feature. Adding one -- even for debugging -- would make the
downstream discrepancy comparison (declared vs. perceived role) circular,
which would silently invalidate the entire project. Anyone modifying this
file must not add such a path without flagging it first, per Section 27
rule 16.
"""
# 183,835,397 parameters, 12 layers, 768 hidden, 12 heads, 512 max seq length, 64k vocab
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from src.ingestion.schema import ROLE_ORDER

DEFAULT_BACKBONE = "microsoft/deberta-v3-base"


class PerceivedRoleEncoder(nn.Module):
    """Text -> distribution over the 5-role vocabulary (ROLE_ORDER).

    tokeniser -> DeBERTa-v3-base -> mean-pool over non-pad tokens -> dropout
    -> linear head -> logits (softmax applied by the loss / at inference).
    """

    def __init__(
        self,
        backbone_name: str = DEFAULT_BACKBONE,
        num_labels: int = len(ROLE_ORDER),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor: # doesn't take delimiter tokens, channel ids, or declared roles -- see Absolute Prohibition A
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (batch, seq, hidden)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # (batch, seq, 1)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts  # mean pooling over real tokens only
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

    @torch.no_grad()
    def predict_proba(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        logits = self.forward(input_ids, attention_mask)
        return torch.softmax(logits, dim=-1)


def load_tokenizer(backbone_name: str = DEFAULT_BACKBONE):
    return AutoTokenizer.from_pretrained(backbone_name)
