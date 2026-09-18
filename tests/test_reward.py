"""Outcome reward and its separation from the detector (sections 10, 11, 16)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ssc.detector.rollout import Outcome, Rollout, Turn, detector_payload_json  # noqa: E402
from ssc.env.reward import exact_match, f1, normalize_answer, score_rollout  # noqa: E402


def test_normalisation_matches_squad_convention():
    assert normalize_answer("The Beatles.") == "beatles"
    assert normalize_answer("  A  Book ") == "book"


def test_exact_match_and_f1():
    assert exact_match("Ann Rule", "ann rule.") == 1
    assert exact_match("Ann Rule", "Anne Rice") == 0
    assert f1("Ann Rule wrote it", "Ann Rule") > 0.5
    assert f1("completely different", "Ann Rule") == 0.0


def test_substring_mode_is_not_the_default_because_it_inflates():
    """A one-token gold matches almost any sentence containing it."""
    pred = "No, Chumbawamba is from England while Spin Doctors are American."
    assert score_rollout(pred, "no", mode="substring")["success"] == 1
    assert score_rollout(pred, "no", mode="em")["success"] == 0


def test_scoring_populates_outcome_without_touching_detector_payload():
    r = Rollout(task_id="t", question="q",
                turns=[Turn(0, "act", "search", {"query": "x"}, "obs", 5)],
                final_answer="Ann Rule")
    s = score_rollout(r.final_answer, "Ann Rule")
    r.outcome = Outcome(success=s["success"], reward=s["reward"],
                        gold_answer="Ann Rule", judge_verdict="em")
    assert r.outcome.success == 1
    blob = detector_payload_json(r)
    # The outcome KEYS must be absent. The final answer itself is allowed by
    # section 11 (input 4), so it is not checked here -- the gold VALUE is
    # checked in the next test with a string that cannot collide with it.
    for banned in ("reward", "success", "gold", "judge"):
        assert banned not in blob.lower(), f"{banned} leaked into detector payload"


def test_gold_string_never_appears_in_detector_payload_after_scoring():
    r = Rollout(task_id="t", question="Who wrote it?",
                turns=[Turn(0, "act", "search", {"query": "x"}, "obs", 5)],
                final_answer="UNIQUEPREDICTION")
    r.outcome = Outcome(success=1, reward=1.0, gold_answer="UNIQUEGOLDSTRING")
    assert "UNIQUEGOLDSTRING" not in detector_payload_json(r)
