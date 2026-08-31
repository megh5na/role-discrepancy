"""
src/models/sidecar/layer_sweep.py

DIAGNOSTIC, not part of the main C7 pipeline. Both validate.py and
validate_native_template.py showed the conflict-condition probe following
the wrapper tag at layer 21/30 (a single heuristically-chosen layer,
`layer_fraction=0.7`). Before concluding the mechanism doesn't reproduce at
all on this substitute model, check whether SOME OTHER layer shows
content-following behaviour -- cheap to check because `output_hidden_states
=True` already computes every layer in one forward pass; we were just
discarding all but one.

Runs on a SMALL subset (fast diagnostic, not the full validation N) purely
to see whether the effect appears ANYWHERE across depth.
"""

from __future__ import annotations

import torch

from src.ingestion.schema import ROLE_ORDER, Role
from src.models.sidecar.hooks import load_sidecar_model, n_layers
from src.models.sidecar.probe import probe_readout, readout_for_role, train_probe
from src.models.sidecar.validate_native_template import (
    _NATIVE_ROLES,
    _wrap_native,
    load_neutral_texts,
    load_register_content,
)


@torch.no_grad()
def extract_all_layers(model, tokenizer, texts: list[str], device: str = "cpu", batch_size: int = 8):
    """Returns list of (n_layers+1) tensors, each (n_texts, hidden) -- one
    per hidden_states index, mean-pooled -- computed from ONE forward pass
    per batch instead of one pass per layer."""
    n_total_layers = n_layers(model) + 1  # +1 for embedding output
    per_layer_vecs = [[] for _ in range(n_total_layers)]
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        out = model(**enc, output_hidden_states=True)
        mask = enc["attention_mask"].unsqueeze(-1).to(out.hidden_states[0].dtype)
        for layer_idx, hidden in enumerate(out.hidden_states):
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            per_layer_vecs[layer_idx].append(pooled)
    return [torch.cat(v, dim=0) for v in per_layer_vecs]


def sweep(model_name: str | None = None, n_neutral: int = 40, n_conflict: int = 30):
    from src.models.sidecar.hooks import DEFAULT_SIDECAR_MODEL

    model_name = model_name or DEFAULT_SIDECAR_MODEL
    model, tokenizer = load_sidecar_model(model_name)
    total_layers = n_layers(model) + 1
    print(f"Sweeping all {total_layers} layers (embedding + {n_layers(model)} transformer blocks)")

    neutral = load_neutral_texts(n_neutral)
    wrapped_texts, labels = [], []
    for text in neutral:
        for role in _NATIVE_ROLES:
            wrapped_texts.append(_wrap_native(tokenizer, text, role))
            labels.append(role)

    reasoning_content = load_register_content(Role.REASONING, n=n_conflict)
    conflict_texts = [_wrap_native(tokenizer, t, Role.USER) for t in reasoning_content]

    all_wrapper_acts = extract_all_layers(model, tokenizer, wrapped_texts)
    all_conflict_acts = extract_all_layers(model, tokenizer, conflict_texts)

    print(f"\n{'layer':<8}{'in_dist_acc':<14}{'P(reasoning)':<14}{'P(user)':<10}{'follows_content?'}")
    results = []
    for layer_idx in range(total_layers):
        clf = train_probe(all_wrapper_acts[layer_idx], labels, seed=42)
        readout = probe_readout(clf, all_conflict_acts[layer_idx])
        mean_reasoningness = float(readout_for_role(readout, Role.REASONING).mean())
        mean_userness = float(readout_for_role(readout, Role.USER).mean())

        preds = clf.predict(all_wrapper_acts[layer_idx].cpu().numpy())
        true_idx = [ROLE_ORDER.index(r) for r in labels]
        in_dist_acc = sum(1 for a, b in zip(preds, true_idx) if a == b) / len(true_idx)

        follows_content = mean_reasoningness > mean_userness
        print(f"{layer_idx:<8}{in_dist_acc:<14.3f}{mean_reasoningness:<14.3f}{mean_userness:<10.3f}{follows_content}")
        results.append({
            "layer": layer_idx, "in_dist_acc": in_dist_acc,
            "P_reasoning": mean_reasoningness, "P_user": mean_userness,
            "follows_content": follows_content,
        })
    return results


if __name__ == "__main__":
    sweep()
