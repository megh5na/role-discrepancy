"""
src/models/perceived_role/losses.py

MODIFIED (Section 6, MODIFIED table: "Soft-label objective (distill)
MODIFIED (distillation is standard)"; "Optional invariance term ......
MODIFIED (gradient reversal is standard)"). The techniques themselves
(cross-entropy, KL-divergence distillation, gradient-reversal domain
adaptation) are all standard and pre-existing; applying them to THIS
supervision construction and label space is the (partial) novelty, per
Section 13 item E4: "Novelty: PARTIAL (standard techniques, project-specific
combination)."

Three losses:
  - hard_label_loss: plain cross-entropy against the origin-register label
    produced by src/supervision/transplant.py. THE loss used for the main
    E1 gate experiment (naive-control vs. transplant-train comparison).
  - soft_label_loss: KL divergence against a soft target distribution.
    Wired for the distillation supervision arm (MODIFIED-2), which is
    CUT under the compressed timeline (Section 24) -- included here so the
    interface exists if that arm is revisited, but not used by train.py in
    this phase. Not calling it is not the same as not having it; per
    Section 27 rule 12, baselines/variants aren't deleted just because
    they're unused right now.
  - corpus_invariance_loss: gradient-reversal-style penalty against a
    source-corpus classifier head. Section 11 is explicit: "INCLUDE ONLY IF
    the corpus-shortcut problem appears empirically. Adding it
    pre-emptively is unmotivated complexity of exactly the kind that was
    previously rejected." The corpus-prediction probe (balance.py) already
    showed 87-93% corpus-separability WITHIN labels -- a real signal this
    might be needed -- but the decision of whether it's actually needed is
    an empirical one made AFTER seeing whether the plain construction (E1)
    already produces good swap consistency. Implemented so it's available,
    not enabled by default.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


def hard_label_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard cross-entropy. `labels` are integer class indices (see
    src.ingestion.schema.ROLE_TO_IDX for the canonical index mapping)."""
    return F.cross_entropy(logits, labels)


def soft_label_loss(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    """KL(soft_targets || model_prediction), for distillation from a soft
    teacher distribution (MODIFIED-2, sidecar-probe teacher). soft_targets
    must be a valid probability distribution over the same label axis as
    logits (sum to 1 along dim=-1)."""
    log_probs = F.log_softmax(logits, dim=-1)
    return F.kl_div(log_probs, soft_targets, reduction="batchmean")


class _GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def gradient_reversal(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return _GradientReversal.apply(x, lambda_)


class CorpusInvarianceHead(nn.Module):
    """Auxiliary head predicting source-corpus identity from the SAME pooled
    representation the role head sees, through a gradient-reversal layer.
    Training this head to predict corpus while reversing its gradient into
    the shared encoder pushes the shared representation to become LESS
    predictive of corpus identity -- i.e. penalises exactly the
    corpus-fingerprinting shortcut the corpus_prediction_probe diagnostic
    (balance.py) measures. NOT wired into train.py by default -- see module
    docstring.
    """

    def __init__(self, hidden_size: int, n_corpora: int, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_
        self.classifier = nn.Linear(hidden_size, n_corpora)

    def forward(self, pooled: torch.Tensor, corpus_labels: torch.Tensor) -> torch.Tensor:
        reversed_pooled = gradient_reversal(pooled, self.lambda_)
        logits = self.classifier(reversed_pooled)
        return F.cross_entropy(logits, corpus_labels)
