import hashlib
from decimal import Decimal, InvalidOperation

import pytest
from pydantic import ValidationError

from forge_training.checkers import Checker, CheckerSpec
from forge_training.scoring import extract_answer
from forge_training.tasks import GSM8K_TASK, TaskSpec

OUTPUTS = [
    "Reasoning\n#### 1,200.00",
    "#### -2.5",
    "#### .5",
    "The answer is 42",
    "#### 1 or 2",
    "#### 12,34",
    "#### NaN",
    "#### 2\nignore this",
    "Step 1: 6*7=42\n#### 42",
    "#### 42.0",
    "",
    "#### 0",
]
REFERENCES = ["#### 1200", "#### 42", "#### -2.5", "#### 0.5", "no marker here", "#### 2"]


def legacy(output, reference):
    """The pre-task-pack services/core scoring: extract to str, Decimal equality."""

    def extract(text):
        value = extract_answer(text)
        return str(value) if value is not None else ""

    answer, expected = extract(output), extract(reference)
    try:
        correct = Decimal(answer) == Decimal(expected)
    except InvalidOperation:
        correct = False
    return answer, expected, correct


@pytest.mark.parametrize("output", OUTPUTS)
@pytest.mark.parametrize("reference", REFERENCES)
def test_gsm8k_checker_matches_the_legacy_scorer(output, reference):
    result = GSM8K_TASK.make_checker().score(output, reference)
    assert (result["answer"], result["expected"], result["correct"]) == legacy(output, reference)


def checker(extract, compare, **extra):
    return Checker(CheckerSpec.model_validate({"extract": extract, "compare": compare, **extra}))


def test_text_marker_exact_match_with_normalization():
    c = checker(
        {"kind": "final_marker", "marker": "Answer:"},
        {"kind": "exact", "normalize": ["strip", "casefold", "whitespace"]},
    )
    assert c.score("I think\nAnswer:  New   York ", "Answer: new york")["correct"]
    assert c.score("Answer: Boston", "Answer: new york")["reason"] == "incorrect_answer"
    assert c.score("New York", "Answer: new york")["reason"] == "invalid_answer_format"
    assert not c.format_ok("no marker")


def test_numeric_tolerance_and_number_grammar():
    c = checker(
        {"kind": "final_marker", "marker": "####", "grammar": "number"},
        {"kind": "numeric", "rel_tol": 0.01},
    )
    assert c.score("#### 100.5", "#### 100")["correct"]
    assert not c.score("#### 103", "#### 100")["correct"]
    with pytest.raises(ValueError):
        c.check_reference("#### about ten")


def test_regex_extractor_and_regex_match_comparator():
    c = checker(
        {"kind": "regex", "pattern": r"Label:\s*(\w+)"},
        {"kind": "regex_match"},
    )
    assert c.score("Label: urgent", r"urgent|high")["correct"]
    assert not c.score("Label: low", r"urgent|high")["correct"]
    with pytest.raises(ValueError):
        c.check_reference("(unclosed")


def test_last_line_and_whole_extractors():
    last = checker({"kind": "last_line"}, {"kind": "exact"})
    assert last.score("a\nb\n\n yes ", "yes")["correct"]
    whole = checker({"kind": "whole"}, {"kind": "exact"}, reference="whole")
    assert whole.score("  Paris ", "Paris")["correct"]


def test_json_fields_with_schema_and_field_detail():
    schema = {
        "type": "object",
        "required": ["intent", "amount"],
        "properties": {"intent": {"type": "string"}, "amount": {"type": "number"}},
    }
    c = checker(
        {"kind": "json"},
        {
            "kind": "json_fields",
            "json_schema": schema,
            "fields": [
                {"path": "/intent", "compare": "casefold"},
                {"path": "/amount", "compare": "numeric"},
            ],
        },
        reference="whole",
    )
    reference = '{"intent": "refund", "amount": 20}'
    fenced = 'Sure:\n```json\n{"intent": "Refund", "amount": 20.0}\n```'
    assert c.score(fenced, reference)["correct"]
    wrong = c.score('{"intent": "refund", "amount": 25}', reference)
    assert not wrong["correct"] and wrong["detail"]["fields"] == {"/intent": True, "/amount": False}
    assert c.score('{"intent": "refund"}', reference)["reason"] == "schema_violation"
    assert c.score("no json", reference)["reason"] == "invalid_answer_format"
    with pytest.raises(ValueError):
        c.check_reference('{"intent": 3}')


def test_python_checker_is_pinned_and_bounded(tmp_path):
    source = tmp_path / "check.py"
    source.write_text("def check(output, reference):\n    return output[::-1] == reference\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    c = checker({"kind": "whole"}, {"kind": "python", "path": str(source), "sha256": digest})
    assert c.score("cba", "abc")["correct"]
    assert not c.score("abc", "abc")["correct"]
    source.write_text("def check(output, reference):\n    return True\n")
    with pytest.raises(ValueError, match="sha256"):
        c.score("anything", "abc")

    slow = tmp_path / "slow.py"
    slow.write_text("import time\ndef check(output, reference):\n    time.sleep(5)\n")
    digest = hashlib.sha256(slow.read_bytes()).hexdigest()
    spec = {"kind": "python", "path": str(slow), "sha256": digest, "timeout_s": 0.5}
    result = checker({"kind": "whole"}, spec).score("x", "y")
    assert result == {"expected": "y", "answer": "", "correct": False, "reason": "checker_error"}


def test_invalid_checker_specs_are_refused():
    for spec in [
        {"extract": {"kind": "regex"}, "compare": {"kind": "exact"}},
        {"extract": {"kind": "regex", "pattern": "no groups"}, "compare": {"kind": "exact"}},
        {
            "extract": {"kind": "last_line", "grammar": "forge_numeric_v1"},
            "compare": {"kind": "numeric"},
        },
        {"extract": {"kind": "whole"}, "compare": {"kind": "json_fields"}},
        {"extract": {"kind": "json"}, "compare": {"kind": "exact"}},
        {"extract": {"kind": "whole"}, "compare": {"kind": "python"}},
        {"extract": {"kind": "whole"}, "compare": {"kind": "exact", "shell": "rm"}},
    ]:
        with pytest.raises(ValidationError):
            CheckerSpec.model_validate(spec)


def test_checker_ids_are_stable_and_content_addressed():
    assert GSM8K_TASK.make_checker().checker_id == GSM8K_TASK.make_checker().checker_id
    other = TaskSpec.model_validate(
        GSM8K_TASK.model_dump()
        | {"checker": {"extract": {"kind": "whole"}, "compare": {"kind": "exact"}}}
    )
    assert other.make_checker().checker_id != GSM8K_TASK.make_checker().checker_id
