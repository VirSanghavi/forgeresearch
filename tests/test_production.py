import re

import pytest
import json

from services.core import production
from services.core.production import arithmetic_slips, campaign_rows, detect, scan

GOOD = "She has 48 / 2 = 24 cookies left, then 24 + 6 = 30.\n#### 30"


def trace(id_, output, prompt=None, **extra):
    return {"id": id_, "prompt": prompt or f"Question {id_}?", "output": output, **extra}


def jsonl(rows):
    return "\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows)


SLIP_CASES = [
    # (output, slip expected). Rows marked "real" are taken from Qwen2.5-0.5B production answers.
    (r"Dad = 4 \times 8 + 3 = 32 + 3 = 35 years", False),  # real: LaTeX and a correct chain
    (r"3 \times 66.67 - 250 = 199.01 - 250 = 316.6", True),  # real: 3 * 66.67 is 200.01
    ("316.67 - 250 = 66.67\n      \n\n3. **Next step", False),  # real: next line starts with "3."
    ("Each box holds 12 * 7 = 74 pens", True),
    ("So 7 * 8 = 54.", True),
    ("He walks (3 + 4) * 2 = 13 miles.", True),
    ("then 950 - 300 = 600", True),
    ("She has 48 / 2 = 24 cookies, then 24 + 6 = 30.", False),
    ("Each gets 10 / 3 = 3.33 dollars.", False),
    (r"\( 200 / 3 = 66.67 \)", False),
    ("Total: $1,200 / 4 = $300. Then 3 x 5 = 15.", False),
    ("He has 1,250 - 300 = 950 left", False),
    ("A 50% * 80 = 40 discount.", False),
    ("A 50% * 80 + 2 = 42 total.", False),
    ("so 6 * 4 + 1 = 25", False),
    ("Step 1.5 = 2 ways", False),
    ("x = 5, so 2x = 10", False),
    ("so 2^3 = 8", False),
    ("-5 + 3 = -2", False),
]


@pytest.mark.parametrize(("output", "expected"), SLIP_CASES)
def test_arithmetic_slips_are_caught_without_false_alarms(output, expected):
    assert bool(arithmetic_slips(output)) is expected


def test_slip_evidence_quotes_the_model_and_the_computed_value():
    assert arithmetic_slips("Each box holds 12 * 7 = 74 pens") == [
        {"equation": "12 * 7 = 74", "computed": "84"}
    ]
    assert arithmetic_slips(r"3 \times 66.67 - 250 = 199.01 - 250") == [
        {"equation": "3 * 66.67 - 250 = 199.01 - 250", "computed": "-49.99"}
    ]


def test_each_detector_fires_only_on_its_own_signal():
    kinds = lambda t: sorted({s["kind"] for s in detect(t)})  # noqa: E731
    assert kinds(trace("a", GOOD)) == []
    assert kinds(trace("b", "The answer is 30")) == ["missing_final_answer"]
    assert kinds(trace("c", GOOD, finish_reason="length")) == ["truncated"]
    assert kinds(trace("d", GOOD, feedback={"rating": -1})) == ["user_reported"]
    assert kinds(trace("e", GOOD, feedback={"rating": 1})) == []
    assert kinds(trace("f", "12 * 7 = 74\n#### 74")) == ["arithmetic_slip"]


def test_malformed_rows_are_skipped_not_fatal():
    text = jsonl(
        [trace("a", GOOD), "{not json", {"id": "x"}, trace("a", GOOD), [1, 2], trace("b", "no answer")]
    )
    report = scan(text)
    assert report["traces"] == 2
    assert [m["line"] for m in report["malformed"]] == [2, 3, 4, 5]
    assert report["flagged"] == 1


def test_sealed_test_questions_are_excluded_before_detection():
    text = jsonl([trace("a", "no answer", prompt="What is  SIX times seven?"), trace("b", "no answer")])
    report = scan(text, sealed_questions=["what is six times seven?"])
    assert report["excluded_sealed_overlap"] == ["a"]
    assert [f["trace"]["id"] for f in report["findings"]] == ["b"]


def test_audit_reports_precision_and_recall_against_labels():
    rows = [
        trace("hit", "12 * 7 = 74\n#### 74"),  # wrong and flagged
        trace("miss", "Clearly it is\n#### 99"),  # wrong, no detectable signal
        trace("false", "12 * 7 = 74 oops, 12 * 7 = 84\n#### 84"),  # right but flagged
        trace("ok", GOOD),  # right, not flagged
    ]
    labels = jsonl(
        [{"id": i, "answer": "#### 84" if i != "ok" else "#### 30"} for i in ("hit", "miss", "false", "ok")]
    )
    audit = scan(jsonl(rows), labels)["audit"]
    assert (audit["true_positives"], audit["missed"], audit["false_positives"]) == (1, 1, 1)
    assert audit["precision"] == 0.5 and audit["recall"] == 0.5


def test_campaign_rows_need_a_reference_and_split_stably():
    rows = [trace(f"t{i}", "no answer", prompt=f"How many apples in basket {i}?") for i in range(40)]
    rows.append(trace("fixed-by-user", "no answer", feedback={"correction": "It is 5\n#### 5"}))
    rows.append(trace("unlabeled", "12 * 7 = 74\n#### 74"))
    labels = jsonl([{"id": f"t{i}", "answer": "#### 3"} for i in range(40)])
    report = scan(jsonl(rows), labels)
    train, held_out = campaign_rows(report, labels)
    ids = {r["trace_id"] for r in train + held_out}
    # A user correction flags a failure but is never trusted as a training answer.
    assert "unlabeled" not in ids and "fixed-by-user" not in ids
    assert all(re.match(r"^prod-[0-9a-f]{12}$", r["id"]) for r in train + held_out)
    assert len(train) + len(held_out) == 40 and train and held_out
    assert not {r["question"] for r in train} & {r["question"] for r in held_out}
    again = campaign_rows(scan(jsonl(rows), labels), labels)
    assert [r["id"] for r in again[1]] == [r["id"] for r in held_out]


def test_too_many_traces_are_refused():
    production.MAX_TRACES, saved = 3, production.MAX_TRACES
    try:
        text = jsonl([trace(f"t{i}", GOOD) for i in range(5)])
        try:
            scan(text)
        except ValueError as error:
            assert "At most 3" in str(error)
        else:
            raise AssertionError("expected refusal")
    finally:
        production.MAX_TRACES = saved
