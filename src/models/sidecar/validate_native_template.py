"""
src/models/sidecar/validate_native_template.py

*** CORRECTED REPRODUCTION ATTEMPT, kept alongside validate.py's original
attempt rather than replacing it ***

WHY THIS SECOND SCRIPT EXISTS: validate.py's first run used synthetic XML
tags (<user>...</user>) and got a clean GATE FAILURE -- the probe followed
the tag (P(user)=0.990) even for reasoning-register content. Before
accepting that as evidence against the mechanism, checked whether the
XML tags were the problem: `tokenizer.apply_chat_template` shows
SmolLM2-135M-Instruct's actual trained format is ChatML
(`<|im_start|>role\\n...<|im_end|>`), which does NOT include `<user>` or
`<reasoning>` as tokens the model was ever trained to treat as role
markers. The first test may therefore have measured "does an
out-of-distribution string get ignored" rather than "does register beat
structural position" -- a different, uninteresting question.

This version uses the model's OWN chat template via
`tokenizer.apply_chat_template`, restricted to the roles it actually
supports: system, user, assistant (SmolLM2's template has no native
document/tool/reasoning role -- so this is a 3-class, not 5-class,
reproduction, honestly narrower in scope than validate.py's attempt).

Both results are kept and reported -- neither is deleted in favour of the
other (Section 27 rule 13: preserve ablations/variants, including
unfavourable ones).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.ingestion.io_utils import read_records
from src.ingestion.schema import Role
from src.models.sidecar.hooks import extract_activations, load_sidecar_model, n_layers
from src.models.sidecar.probe import probe_readout, readout_for_role, train_probe

REPO_ROOT = Path(__file__).resolve().parents[3]

_NATIVE_ROLES = [Role.SYSTEM, Role.USER, Role.REASONING]  # REASONING stands in for "assistant" here
_ROLE_TO_CHATML_ROLE = {Role.SYSTEM: "system", Role.USER: "user", Role.REASONING: "assistant"}


def _wrap_native(tokenizer, text: str, role: Role) -> str:
    chatml_role = _ROLE_TO_CHATML_ROLE[role]
    return tokenizer.apply_chat_template(
        [{"role": chatml_role, "content": text}], tokenize=False
    )


def load_neutral_texts(n: int = 150) -> list[str]:
    path = REPO_ROOT / "data" / "interim" / "clean_spans_register.jsonl"
    texts = []
    for r in read_records(path):
        if r.source_dataset == "c4-en":
            texts.append(r.text)
            if len(texts) >= n:
                break
    return texts


def load_register_content(role: Role, n: int = 60) -> list[str]:
    path = REPO_ROOT / "data" / "interim" / "clean_spans_register.jsonl"
    texts = []
    for r in read_records(path):
        if r.declared_role == role:
            texts.append(r.text)
            if len(texts) >= n:
                break
    return texts


def run_validation(model_name: str | None = None, n_neutral: int = 150, layer_fraction: float = 0.7, seed: int = 42) -> dict:
    from src.models.sidecar.hooks import DEFAULT_SIDECAR_MODEL

    model_name = model_name or DEFAULT_SIDECAR_MODEL
    print(f"Loading sidecar model: {model_name}")
    model, tokenizer = load_sidecar_model(model_name)
    layer = max(1, round(layer_fraction * n_layers(model)))
    print(f"Model has {n_layers(model)} layers; probing at layer {layer}")

    neutral = load_neutral_texts(n_neutral)
    wrapped_texts, labels = [], []
    for text in neutral:
        for role in _NATIVE_ROLES:
            wrapped_texts.append(_wrap_native(tokenizer, text, role))
            labels.append(role)
    print(f"Built {len(wrapped_texts)} native-template wrapper-probe examples ({len(_NATIVE_ROLES)} roles)")

    acts = extract_activations(model, tokenizer, wrapped_texts, layer=layer)
    clf = train_probe(acts, labels, seed=seed)

    held_out_neutral = load_neutral_texts(40)[20:40]
    ho_wrapped, ho_labels = [], []
    for text in held_out_neutral:
        for role in _NATIVE_ROLES:
            ho_wrapped.append(_wrap_native(tokenizer, text, role))
            ho_labels.append(role)
    ho_acts = extract_activations(model, tokenizer, ho_wrapped, layer=layer)
    ho_preds = clf.predict(ho_acts.cpu().numpy())
    from src.ingestion.schema import ROLE_ORDER

    ho_true = [ROLE_ORDER.index(r) for r in ho_labels]
    ho_acc = sum(1 for a, b in zip(ho_preds, ho_true) if a == b) / len(ho_true)
    print(f"In-distribution wrapper-recovery accuracy: {ho_acc:.3f}")

    # CONFLICT: real reasoning-register content wrapped as a native <user> turn.
    reasoning_content = load_register_content(Role.REASONING, n=60)
    conflict_texts = [_wrap_native(tokenizer, t, Role.USER) for t in reasoning_content]
    conflict_acts = extract_activations(model, tokenizer, conflict_texts, layer=layer)
    readout = probe_readout(clf, conflict_acts)
    mean_reasoningness = float(readout_for_role(readout, Role.REASONING).mean())
    mean_userness = float(readout_for_role(readout, Role.USER).mean())
    print(
        f"\nCONFLICT CONDITION (native template) -- reasoning-register content "
        f"under native <|im_start|>user tags:\n"
        f"  mean P(reasoning/'assistant-like') = {mean_reasoningness:.3f}\n"
        f"  mean P(user)                       = {mean_userness:.3f}"
    )
    follows_content_register = mean_reasoningness > mean_userness
    verdict = (
        "REPRODUCES (qualitatively), native template"
        if follows_content_register
        else "STILL DOES NOT REPRODUCE, native template -- genuine negative signal at this model scale"
    )
    print(f"\nGATE VERDICT (native template variant): {verdict}")

    return {
        "model_name": model_name,
        "layer": layer,
        "template": "native_chatml",
        "roles_tested": [r.value for r in _NATIVE_ROLES],
        "in_distribution_accuracy": ho_acc,
        "conflict_mean_reasoningness": mean_reasoningness,
        "conflict_mean_userness": mean_userness,
        "follows_content_register": follows_content_register,
        "verdict": verdict,
    }


if __name__ == "__main__":
    result = run_validation()
    out_path = REPO_ROOT / "experiments" / "sidecar_validation" / "results_native_template.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")
