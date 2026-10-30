from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

import numpy as np

from src.ingestion.schema import ROLE_ORDER, Role
from src.preprocessing.delimiters import strip_delimiters
from src.scoring.asymmetry import expected_severity, severity

REPO_ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
CK = REPO_ROOT / "checkpoints"
CHECKPOINTS = {
    "naive": [CK / "perceived_role_v1_naive", CK / "kept_seed_models" / "perceived_role_v1_naive_seed123",
              CK / "kept_seed_models" / "perceived_role_v1_naive_seed456"],
    "transplant": [CK / "perceived_role_v1", CK / "kept_seed_models" / "perceived_role_v1_seed123",
                   CK / "kept_seed_models" / "perceived_role_v1_seed456"],
}
UNRELIABLE_CHANNELS = {"tool", "reasoning"}   # per-channel control thresholds are degenerate (see module docstring)
MAX_LENGTH = 64                               # as in the E3/E4/E6 scoring scripts


def _checkpoint_prob_fn(ckpt: Path) -> Callable[[list[str]], np.ndarray]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch

    from src.models.perceived_role.checkpoint import load_checkpoint
    from src.models.perceived_role.encoder import load_tokenizer

    model, meta = load_checkpoint(ckpt)
    model.eval()
    tok = load_tokenizer(meta["backbone_name"])

    def probs(texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 16):
            enc = tok(texts[i:i + 16], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
            with torch.no_grad():
                out.append(torch.softmax(model(enc["input_ids"], enc["attention_mask"]), dim=-1).numpy())
        return np.concatenate(out, axis=0)

    return probs


class RoleDiscrepancyScorer:
    def __init__(self, prob_fns: list[Callable[[list[str]], np.ndarray]], thresholds: dict, condition: str = "custom"):
        """prob_fns: one callable per ensemble member, texts -> (n, 5) role probabilities in ROLE_ORDER.
        Injected so tests (and any front end) can use fakes; `from_checkpoints` builds the real ones."""
        self.prob_fns, self.thresholds, self.condition = prob_fns, thresholds, condition

    @classmethod
    def from_checkpoints(cls, condition: str, thresholds_path: Path = HERE / "thresholds.json") -> "RoleDiscrepancyScorer":
        if condition not in CHECKPOINTS:
            raise ValueError(f"condition must be one of {sorted(CHECKPOINTS)}")
        missing = [str(p) for p in CHECKPOINTS[condition] if not (p / "model.pt").exists()]
        if missing:
            raise FileNotFoundError(f"missing checkpoints: {missing}")
        th = json.loads(Path(thresholds_path).read_text())
        return cls([_checkpoint_prob_fn(p) for p in CHECKPOINTS[condition]],
                   {"pooled": th["thresholds"][condition], "per_channel": th["thresholds_per_channel"][condition]}, condition)

    def score(self, text: str, declared_role: Role | str, policy: str = "pooled") -> dict:
        if policy not in ("pooled", "per_channel"):
            raise ValueError("policy must be 'pooled' or 'per_channel'")
        declared = Role(declared_role) if not isinstance(declared_role, Role) else declared_role
        clean = strip_delimiters(text)
        member_probs = np.stack([fn([clean])[0] for fn in self.prob_fns])          # (members, 5)
        member_scores = [float(expected_severity(p, declared)) for p in member_probs]
        mean_probs = member_probs.mean(axis=0)
        ens_score = float(np.mean(member_scores))                                  # same ensembling as E4 (mean of per-model scores)
        perceived = ROLE_ORDER[int(mean_probs.argmax())]
        thr = self.thresholds["pooled"] if policy == "pooled" else self.thresholds["per_channel"][declared.value]
        return {
            "declared_role": declared.value,
            "perceived_role": perceived.value,
            "role_probabilities": {r.value: float(p) for r, p in zip(ROLE_ORDER, mean_probs)},
            "p_declared": float(mean_probs[ROLE_ORDER.index(declared)]),
            "severity_of_perceived_vs_declared": severity(perceived, declared),
            "score": ens_score,
            "member_scores": member_scores,
            "threshold": float(thr),
            "policy": policy,
            "threshold_reliable": not (policy == "per_channel" and declared.value in UNRELIABLE_CHANNELS),
            "flagged": bool(ens_score >= thr),
            "text_changed_by_delimiter_stripping": clean != text,
            "condition": self.condition,
        }
