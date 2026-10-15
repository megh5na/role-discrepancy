"""tests/test_adaptive_attacker.py -- E7b helpers (no API call, no model)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp07_adaptive.analyze_e7_adaptive import mcnemar_exact  # noqa: E402
from experiments.exp07_adaptive.run_adaptive_attacker import build_feedback, is_refusal  # noqa: E402


def test_refusal_detection_catches_provider_refusals_empty_output_and_apologies():
    assert is_refusal("Style", "refusal")                       # provider-side refusal with partial text
    assert is_refusal("", "end_turn") and is_refusal("   ", "end_turn")
    assert is_refusal("I can't help with that.", "end_turn")
    assert not is_refusal("By the way, joining the newsletter is worth it.", "end_turn")


def test_feedback_shows_history_scores_and_the_threshold_but_not_responses():
    hist = [{"round": 0, "refused": False, "text": "Do X.", "score": 0.91, "flagged": True, "followed": True, "threshold": 0.85, "response": "SECRET RESPONSE"},
            {"round": 1, "refused": True, "text": None, "score": None, "flagged": None, "followed": None, "threshold": 0.85, "response": None}]
    fb = build_feedback("Do X.", hist, 0.85)
    assert "0.85" in fb and "detector score 0.91 (flagged)" in fb and "followed it: yes" in fb
    assert "no usable rewrite" in fb and "SECRET RESPONSE" not in fb


def test_exact_mcnemar():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(1, 0) == 1.0                           # one discordant pair can never be significant
    assert abs(mcnemar_exact(5, 0) - 0.0625) < 1e-9
