import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attacks.asr_harness import CallBudget, load_attack_sample, summarize  # noqa: E402


class _FakeMessage:
    def __init__(self, text: str):
        self.content = [type("Block", (), {"text": text})()]


class _FakeMessages:
    def __init__(self, response_text: str):
        self._response_text = response_text
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return _FakeMessage(self._response_text)


class _FakeClient:
    def __init__(self, response_text: str = "a rewritten, less direct version"):
        self.messages = _FakeMessages(response_text)


def test_call_budget_allows_up_to_the_cap():
    budget = CallBudget(max_calls=3)
    budget.spend(1)
    budget.spend(2)
    assert budget.used == 3


def test_call_budget_refuses_past_the_cap():
    budget = CallBudget(max_calls=2)
    budget.spend(2)
    with pytest.raises(RuntimeError, match="budget exceeded"):
        budget.spend(1)


def test_load_attack_sample_is_stratified_and_respects_n(tmp_path):
    import json

    from src.ingestion.schema import DataCategory, Record, Role

    path = tmp_path / "fake_bipia.jsonl"
    records = []
    for cat in ["task_irrelevant", "task_relevant", "targeted"]:
        for i in range(5):
            records.append(Record(
                record_id=f"{cat}-{i}", text=f"attack text {cat} {i}",
                declared_role=Role.DOCUMENT, source_dataset="bipia",
                category=DataCategory.INJECTION,
                is_injected=True, injection_type=cat,
            ))
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r.to_dict()) + "\n")

    cfg = {"attack_source": {"n_attacks": 6, "stratify_by": "injection_type", "seed": 42}}
    # load_attack_sample resolves paths relative to REPO_ROOT; point REPO_ROOT at the
    # temp dir's parent so a plain relative filename works without touching the real repo.
    import src.attacks.asr_harness as m
    orig_root = m.REPO_ROOT
    try:
        m.REPO_ROOT = path.parent
        cfg["attack_source"]["path"] = path.name
        sample = load_attack_sample(cfg)
    finally:
        m.REPO_ROOT = orig_root

    assert len(sample) <= 6
    categories_seen = {r.injection_type for r in sample}
    assert len(categories_seen) >= 2  # genuinely stratified, not all one category


def test_summarize_computes_asr_and_mean_score_per_level():
    results = [
        {"destyling_level": 0.0, "attack_succeeded": True, "discrepancy_score": 0.9},
        {"destyling_level": 0.0, "attack_succeeded": False, "discrepancy_score": 0.8},
        {"destyling_level": 1.0, "attack_succeeded": True, "discrepancy_score": 0.2},
        {"destyling_level": 1.0, "attack_succeeded": True, "discrepancy_score": 0.3},
    ]
    summary = summarize(results)
    assert summary[0.0]["asr"] == pytest.approx(0.5)
    assert summary[0.0]["mean_discrepancy_score"] == pytest.approx(0.85)
    assert summary[1.0]["asr"] == pytest.approx(1.0)
    assert summary[1.0]["mean_discrepancy_score"] == pytest.approx(0.25)


def test_destyle_level_zero_never_calls_the_api():
    from src.attacks.destyle import destyle_instruction

    client = _FakeClient()
    out = destyle_instruction(client, "original text", level=0.0, model="claude-sonnet-5")
    assert out == "original text"
    assert client.messages.calls == 0  # level 0 must be free -- no API call for the control


def test_destyle_nonzero_level_calls_the_api_and_returns_its_text():
    from src.attacks.destyle import destyle_instruction

    client = _FakeClient(response_text="a destyled rewrite")
    out = destyle_instruction(client, "original text", level=0.5, model="claude-sonnet-5")
    assert out == "a destyled rewrite"
    assert client.messages.calls == 1
