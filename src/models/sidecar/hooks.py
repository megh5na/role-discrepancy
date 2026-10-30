from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_SIDECAR_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def load_sidecar_model(model_name: str = DEFAULT_SIDECAR_MODEL, device: str = "cpu", dtype=None):
    """`dtype=None` keeps the original behaviour (default precision). A larger sidecar (E5,
    e.g. 1.7B on an 8 GB machine) can be loaded with dtype=torch.bfloat16; pooled
    activations are cast back to float32 in extract_activations so the sklearn probe works."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True, **({"torch_dtype": dtype} if dtype else {}))
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
    return torch.cat(all_vecs, dim=0).float()


def n_layers(model) -> int:
    return model.config.num_hidden_layers
 # returns one vector per unit text