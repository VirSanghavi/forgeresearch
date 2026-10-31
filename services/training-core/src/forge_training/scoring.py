"""A versioned DEVELOPMENT scorer. This is not the official GSM8K harness."""

from __future__ import annotations

import math
import random
import re
from decimal import Decimal

FINAL_ANSWER = re.compile(
    r"(?:^|\n)####\s*([+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+))[ \t\r\n]*$"
)
SCORER_VERSION = "forge_numeric_v1"


def extract_answer(text: str) -> Decimal | None:
    match = FINAL_ANSWER.search(text)
    if match is None:
        return None
    return Decimal(match.group(1).replace(",", ""))


def score(prediction: str, reference: str) -> tuple[bool, str]:
    expected = extract_answer(reference)
    if expected is None:
        raise ValueError("Reference answer must end with a numeric '#### <answer>' line")
    actual = extract_answer(prediction)
    if actual is None:
        return False, "invalid_answer_format"
    correct = actual == expected
    return correct, "correct" if correct else "incorrect_answer"


def wilson_interval(correct: int, count: int) -> list[float]:
    if count < 1 or not 0 <= correct <= count:
        raise ValueError("Invalid evaluation counts")
    z = 1.959963984540054
    proportion = correct / count
    denominator = 1 + z * z / count
    center = (proportion + z * z / (2 * count)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count**2))
    return [max(0.0, center - half / denominator), min(1.0, center + half / denominator)]


def paired_bootstrap(differences: list[int], seed: int, resamples: int = 2000) -> list[float]:
    if not differences:
        raise ValueError("Cannot compare empty predictions")
    generator = random.Random(seed)
    n = len(differences)
    estimates = sorted(sum(generator.choices(differences, k=n)) / n * 100 for _ in range(resamples))
    return [estimates[int(0.025 * resamples)], estimates[int(0.975 * resamples) - 1]]
