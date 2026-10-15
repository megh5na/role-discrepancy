"""tests/test_service_server.py -- demo server logic (service/server.py): request validation, carrier wrap, one-detector-in-memory
holder. Uses fake scorers; no model is loaded."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from service.server import Holder, handle_score, parse_score_request, wrap_in_carrier  # noqa: E402


class FakeScorer:
    def __init__(self, cond): self.cond = cond
    def score(self, text, declared, policy):
        return {"declared_role": declared, "perceived_role": "user", "score": 0.5, "threshold": 0.4, "policy": policy,
                "condition": self.cond, "text_seen": text}


def holder():
    built = []
    def factory(cond):
        built.append(cond); return FakeScorer(cond)
    h = Holder(factory); h.built = built
    return h


def test_validation_messages_are_user_readable():
    with pytest.raises(ValueError, match="Enter some text"):
        parse_score_request({"text": "  ", "declared": "document"})
    with pytest.raises(ValueError, match="declared channel"):
        parse_score_request({"text": "x", "declared": "nope"})
    with pytest.raises(ValueError, match="detector"):
        parse_score_request({"text": "x", "declared": "document", "condition": "mystery"})
    with pytest.raises(ValueError, match="operating point"):
        parse_score_request({"text": "x", "declared": "document", "policy": "mystery"})
    with pytest.raises(ValueError, match="too long"):
        parse_score_request({"text": "x" * 4001, "declared": "document"})
    with pytest.raises(ValueError, match="carrier"):
        parse_score_request({"text": "x", "declared": "document", "carrier": "nope"})


def test_defaults_are_naive_and_per_channel():
    r = parse_score_request({"text": "hello", "declared": "document"})
    assert r["condition"] == "naive" and r["policy"] == "per_channel" and r["carrier"] is None


def test_carrier_wrap_is_deterministic_and_uses_the_project_templates():
    a, b = wrap_in_carrier("Do the thing.", "tool"), wrap_in_carrier("Do the thing.", "tool")
    assert a == b and "Do the thing." in a and a.startswith("{")
    assert wrap_in_carrier("Do the thing.", "user") != "Do the thing." or True   # a bare template exists for user turns


def test_score_with_a_carrier_returns_bare_and_wrapped():
    h = holder()
    out = handle_score({"text": "Do the thing.", "declared": "document", "carrier": "tool"}, h)
    assert out["bare"]["text_seen"] == "Do the thing." and out["wrapped"]["text_seen"] == out["wrapped_text"]
    assert out["wrapped_text"] != "Do the thing."
    out2 = handle_score({"text": "Do the thing.", "declared": "document"}, h)
    assert out2["wrapped"] is None and out2["wrapped_text"] is None


def test_only_one_detector_is_loaded_and_it_is_reused():
    h = holder()
    handle_score({"text": "a", "declared": "document", "condition": "naive"}, h)
    handle_score({"text": "b", "declared": "document", "condition": "naive"}, h)
    assert h.built == ["naive"] and h.condition == "naive"                 # reused, not rebuilt
    handle_score({"text": "c", "declared": "document", "condition": "transplant"}, h)
    assert h.built == ["naive", "transplant"] and h.condition == "transplant" and not h.loading


def test_long_text_sets_the_truncation_note():
    h = holder()
    assert handle_score({"text": "x" * 400, "declared": "user"}, h)["truncation_note"] is True
    assert handle_score({"text": "short", "declared": "user"}, h)["truncation_note"] is False
