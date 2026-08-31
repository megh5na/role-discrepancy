"""
src/baselines/guardrails/harness.py

EXISTING models, NEW harness (Section 8 C8(b): "EXISTING models / NEW
harnesses"). Architecture component: C8(b) -- the headline comparison for
Experiment 3.

All of PromptGuard 2, ProtectAI v2, and PIGuard/InjecGuard share the same
shape (Section 8 C8: "A fine-tuned encoder producing an attack probability
from input text") -- a HuggingFace sequence-classification checkpoint with
a binary or near-binary output. One shared harness loads any of them by
HF model id and exposes a uniform `score(text) -> float` (probability of
"injection"/"malicious"), so Experiment 3's matched-FPR comparison treats
every guardrail identically regardless of which specific checkpoint or
label scheme it uses internally.

LlamaGuard-3-8B is NOT wired through this harness: at 8B parameters it is
not feasible to run at any volume on this project's CPU/MPS laptop within
the compressed timeline's compute budget (contrast: DeBERTa-v3-base at
184M already runs at ~3.4s/step here). This is a disclosed compute
constraint, not a silent omission. A smaller
LlamaGuard variant or a hosted-API path is the documented follow-up, alongside the LLM-judge baseline (which needs API access not yet
configured in this environment).

Label-scheme handling: different checkpoints name their "injection" class
differently (LABEL_1, INJECTION, MALICIOUS, ...). `injection_label_hint`
lets the caller specify which output index/label counts as "flagged";
default falls back to "whichever label is not the first" for binary heads,
which is verified against each model's actual config at load time (not
assumed) via `_resolve_injection_index`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class GuardrailSpec:
    name: str
    hf_model_id: str
    injection_label_hint: str | None = None  # e.g. "INJECTION", "LABEL_1" -- None = auto-resolve


# Registry of guardrail baselines this project compares against (Section 4.4).
# LlamaGuard-3-8B and LLM-judge intentionally excluded here -- see module docstring.
#
# ACCESS CONSTRAINTS DISCOVERED AT BUILD TIME, DISCLOSED (not silently
# worked around -- Section 27 rule 1):
#   - meta-llama/Llama-Prompt-Guard-2-86M is GATED (requires Meta's manual
#     approval on the HF account running this code). Kept in the registry
#     (so intent is visible and it activates automatically if the user's
#     account is ever approved) but WILL be skipped by run_exp03.py's
#     try/except until then.
#   - leolee99/PIGuard (the spec-named PIGuard/InjecGuard baseline) ships
#     with `trust_remote_code=True` -- i.e. loading it executes arbitrary
#     Python from the HF repo on this machine. This falls under "downloading
#     or executing files from untrusted sources," which this project does
#     not do without the user's explicit, informed sign-off. NOT included
#     here. If the user wants it, they need to approve that specifically.
#   - "third_party_deberta_v3_injection" substitutes for the resulting gap:
#     a standard (no custom code, no gating) DeBERTa-v3-base fine-tune for
#     the same task, so the matched-FPR comparison still has >=2 real
#     detectors to compare against instead of silently degrading to one.
KNOWN_GUARDRAILS: dict[str, GuardrailSpec] = {
    "promptguard2": GuardrailSpec(
        name="PromptGuard 2 (86M)",
        hf_model_id="meta-llama/Llama-Prompt-Guard-2-86M",
        injection_label_hint=None,
    ),
    "protectai_v2": GuardrailSpec(
        name="ProtectAI v2",
        hf_model_id="protectai/deberta-v3-base-prompt-injection-v2",
        injection_label_hint=None,
    ),
    "third_party_deberta_v3_injection": GuardrailSpec(
        name="DeBERTa-v3 prompt-injection (third-party fine-tune, substitute for PIGuard)",
        hf_model_id="Octavio-Santana/deberta-v3-base-prompt-injection-detection",
        injection_label_hint="LABEL_1",
    ),
}


def resolve_injection_index(id2label: dict[int, str], injection_label_hint: str | None = None) -> int:
    """Pure logic, factored out of GuardrailBaseline so it's unit-testable
    without downloading any model (tests/test_guardrail_harness.py)."""
    if injection_label_hint:
        for idx, label in id2label.items():
            if label.upper() == injection_label_hint.upper():
                return int(idx)
        raise ValueError(
            f"injection_label_hint={injection_label_hint!r} not found in id2label={id2label}"
        )
    positive_markers = ("injection", "malicious", "unsafe", "jailbreak", "1", "positive")
    for idx, label in id2label.items():
        if any(m in label.lower() for m in positive_markers):
            return int(idx)
    return 1 if len(id2label) > 1 else 0


class GuardrailBaseline:
    """Wraps one HF sequence-classification checkpoint as a uniform
    text -> P(injection) scorer."""

    def __init__(self, spec: GuardrailSpec, device: str | None = None):
        self.spec = spec
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(spec.hf_model_id)
        self.model.to(self.device)
        self.model.eval()
        self.injection_idx = resolve_injection_index(
            self.model.config.id2label, self.spec.injection_label_hint
        )

    @torch.no_grad()
    def score_batch(self, texts: list[str], batch_size: int = 16, max_length: int = 256) -> list[float]:
        scores = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
            ).to(self.device)
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            scores.extend(probs[:, self.injection_idx].cpu().tolist())
        return scores

    def make_score_fn(self):
        def score_fn(texts: list[str]) -> list[float]:
            return self.score_batch(texts)

        return score_fn
