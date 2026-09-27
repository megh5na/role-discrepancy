"""
src/models/sidecar/hooks.py

EXISTING technique (activation extraction via output_hidden_states), NEW
integration code. Architecture component: C7 (Section 8), "use Ye et al.'s
released code where possible."

Purpose (restated from the spec, Section 8 C7): (i) validate the premise
before building on it, (ii) supply the white-box UPPER BOUND for Experiment
5 (oracle gap), (iii) optional teacher for the (cut) distillation arm.

*** THIS FILE, UNLIKE src/models/perceived_role/encoder.py, IS ALLOWED TO
READ ACTIVATIONS AND SEE WRAPPER TAGS. *** That is the entire point of the
sidecar: it reproduces the WHITE-BOX method (which our text-only encoder
cannot use -- Section 27, Absolute Prohibition A applies to the text-only
encoder, NOT to this reproduction of prior work). Do not let this file's
patterns leak into src/models/perceived_role/.

MODEL CHOICE, A DISCLOSED COMPUTE SUBSTITUTION: Ye et al. test four
open-weight LLMs (unnamed in the source material available to this build).
This machine has 8GB total RAM and, per the E1 runs,
real headroom for a model is on the order of ~1-2GB once the rest of the
session's footprint is accounted for. We substitute a much smaller
open-weight instruction-tuned model -- SmolLM2-135M-Instruct -- which still
has a genuine chat template (system/user/assistant roles) and is small
enough to run activation extraction on CPU without repeating the thrashing
diagnosed for C4. This changes the ABSOLUTE numbers we can expect (a 135M
model's internal role geometry is presumably less crisp than the larger
models Ye et al. tested) but not the STRUCTURE of the validation: we are
checking whether the qualitative conflict pattern (reasoning-style text
under a USER tag reads as user-style, not reasoning-style) reproduces at
all, which is the GATE (Section 8 C7: "GATE: If this does not reproduce,
escalate before proceeding"), not whether the exact percentages match.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_SIDECAR_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def load_sidecar_model(model_name: str = DEFAULT_SIDECAR_MODEL, device: str = "cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
    model.to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def extract_activations(
    model, tokenizer, texts: list[str], layer: int, device: str = "cpu", batch_size: int = 8
) -> torch.Tensor:
    """Mean-pooled hidden-state activations at `layer` for each text.

    `layer` indexes into `hidden_states` (0 = embedding output, 1..N = each
    transformer block's output) -- the same "fixed layer" a linear probe
    reads from in Ye et al.'s method (spec Section 2, "Linear probe" term
    definition: "Logistic regression over activation vectors from a fixed
    layer").
    """
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        out = model(**enc, output_hidden_states=True)
        hidden = out.hidden_states[layer]  # (batch, seq, hidden)
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        all_vecs.append(pooled)
    return torch.cat(all_vecs, dim=0)


def n_layers(model) -> int:
    return model.config.num_hidden_layers
 # returns one vector per unit text