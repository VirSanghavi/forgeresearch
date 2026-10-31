from services.core.briefing import build
from services.core.director import build_input


def sample():
    return {
        "id": "c",
        "status": "baseline",
        "config": {"max_experiments": 3, "max_seconds": 1800},
        "experiments": [],
        "reviews": [],
        "environments": [],
    }


def test_baseline_has_no_invented_scores_or_passed_checks():
    report = build(sample(), [])
    assert report["experiment_number"] == 0
    assert report["benchmarks"]["development_before"] is None
    assert all(c["state"] == "pending" for c in report["oversight"]["checks"])


def test_preflight_is_not_represented_as_checkpoint_approval():
    s = sample() | {
        "status": "training",
        "experiments": [
            {
                "id": "exp-001",
                "status": "training",
                "strategy": "grpo",
                "hypothesis": "Learn from exact actor rewards",
            }
        ],
        "reviews": [
            {
                "id": "c-exp-001-preflight",
                "decision": "approved",
                "evidence": {"phase": "preflight"},
                "reason_codes": [],
            }
        ],
    }
    report = build(s, [{"kind": "message", "relay_id": "real-receipt"}, {"kind": "message_pending"}])
    assert report["experiment_number"] == 1
    assert report["oversight"]["decision_scope"] == "training preflight"
    assert report["oversight"]["checks"][1]["state"] == "pending"
    assert report["execution"]["relay_deliveries"] == 1


def test_director_input_contains_only_bounded_development_evidence():
    context = build_input(
        {"objective": "Improve mathematical reasoning", "sealed_test": "secret"},
        {
            "score": 0,
            "predictions": [
                {"question": str(i), "output": "wrong", "expected": "42", "correct": False} for i in range(12)
            ],
        },
        [],
    )
    assert len(context["failures"]) == 6
    assert "sealed_test" not in context and "secret" not in str(context)


def test_next_attempt_is_reported_as_planning_after_a_completed_run():
    s = sample() | {
        "status": "diagnosing",
        "experiments": [
            {"id": "exp-001", "status": "completed", "strategy": "broad_sft", "hypothesis": "Train examples"}
        ],
    }
    assert build(s, [])["experiment_number"] == 2
    assert build(s, [])["hypothesis"] is None
