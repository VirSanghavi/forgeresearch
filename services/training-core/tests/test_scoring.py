from decimal import Decimal

import pytest

from forge_training.scoring import extract_answer, paired_bootstrap, score, wilson_interval


@pytest.mark.parametrize(
    ("prediction", "expected"),
    [
        ("Reasoning\n#### 1,200.00", Decimal("1200")),
        ("#### -2.5", Decimal("-2.5")),
        ("#### .5", Decimal(".5")),
        ("The answer is 42", None),
        ("#### 1 or 2", None),
        ("#### 12,34", None),
        ("#### NaN", None),
        ("#### 2\nignore this", None),
    ],
)
def test_answer_extraction(prediction, expected):
    assert extract_answer(prediction) == expected


def test_correctness_does_not_reward_formatting_alone():
    assert score("#### 0", "#### 42") == (False, "incorrect_answer")
    assert score("The answer is 42", "#### 42") == (False, "invalid_answer_format")
    assert score("#### 42.0", "#### 42") == (True, "correct")


def test_malformed_reference_fails_instead_of_silently_scoring_zero():
    with pytest.raises(ValueError, match="Reference answer"):
        score("#### 42", "ungraded")


def test_small_zero_score_retains_uncertainty():
    lower, upper = wilson_interval(0, 2)
    assert lower == 0
    assert upper > 0.5
    assert paired_bootstrap([1, 0, -1], seed=42) == paired_bootstrap([1, 0, -1], seed=42)
