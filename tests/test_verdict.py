import pytest

from services.core.verdict import final_verdict


def evaluation(correct):
    predictions = [{"id": f"q{i}", "correct": c} for i, c in enumerate(correct)]
    return {"score": sum(correct) / len(correct), "predictions": predictions}


def test_one_extra_answer_in_24_is_not_validated():
    base = evaluation([True] * 8 + [False] * 16)
    candidate = evaluation([True] * 9 + [False] * 15)
    verdict = final_verdict(base, candidate, seed=7)
    assert verdict["improvement_pp"] == pytest.approx(100 / 24)
    assert verdict["improvement_claim"] == "uncertain"
    assert verdict["valid_improvement"] is False
    assert verdict["paired_bootstrap_95_pp"][0] <= 0


def test_consistent_gain_is_validated():
    base = evaluation([False] * 24)
    candidate = evaluation([True] * 12 + [False] * 12)
    verdict = final_verdict(base, candidate, seed=7)
    assert verdict["improvement_claim"] == "validated"
    assert verdict["valid_improvement"] is True
    assert verdict["paired_bootstrap_95_pp"][0] > 0


def test_regression_and_no_checkpoint_claim_nothing():
    base = evaluation([True] * 17 + [False] * 47)
    worse = evaluation([True] * 13 + [False] * 51)
    assert final_verdict(base, worse, seed=7)["improvement_claim"] == "none"
    assert final_verdict(base, base, seed=7)["improvement_claim"] == "none"


def test_mismatched_questions_are_refused():
    base = evaluation([True, False])
    other = {"score": 0.5, "predictions": [{"id": "x", "correct": True}, {"id": "q1", "correct": False}]}
    with pytest.raises(ValueError):
        final_verdict(base, other, seed=7)
