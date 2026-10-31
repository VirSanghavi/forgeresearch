import json
from collections import Counter

import pytest

from services.core import director, hard_examples, strategies


def row(i):
    return {"id": f"train-{i}", "question": f"q{i}", "answer": f"a <<1+{i}=x>>x\n#### {i}"}


def prediction(i, correct, finish="stop"):
    return {"id": f"train-{i}", "output": f"mine {i}\n#### {i}", "correct": correct, "finish_reason": finish}


def test_clean_reference_removes_only_calculator_annotations():
    text = "Cost is 4 * $20 = $<<4*20=80>>80.\n#### 80"
    assert hard_examples.clean_reference(text) == "Cost is 4 * $20 = $80.\n#### 80"


def test_build_targets_misses_with_references_and_hits_with_the_models_own_answer():
    rows = [row(i) for i in range(3)]
    built = hard_examples.build(
        rows, [prediction(0, False), prediction(1, True), prediction(2, True, "length")], 42
    )
    answers = {r["id"]: r["answer"] for r in built["rows"]}
    assert answers["train-0"] == "a x\n#### 0"
    assert answers["train-1"] == "mine 1\n#### 1"
    # A correct answer cut off at the token limit is not a behaviour worth holding in place.
    assert answers["train-2"] == "a x\n#### 2"
    assert (built["hard"], built["easy"], built["mined"]) == (2, 1, 3)
    assert built["hard_ids"] == ["train-0", "train-2"]


def test_build_gives_hard_rows_a_third_of_the_steps_and_never_repeats_easy_rows():
    rows = [row(i) for i in range(40)]
    predictions = [prediction(i, i >= 4) for i in range(40)]
    built = hard_examples.build(rows, predictions, 42)
    counts = Counter(r["id"] for r in built["rows"])
    assert all(counts[f"train-{i}"] == 1 for i in range(4, 40))
    assert sum(counts[f"train-{i}"] for i in range(4)) == 18
    assert all(counts[f"train-{i}"] >= 4 for i in range(4))
    assert built["rows"][0]["id"] in built["hard_ids"]


def test_build_refuses_when_there_is_nothing_to_mine_or_rows_are_misaligned():
    with pytest.raises(ValueError, match="nothing to mine"):
        hard_examples.build([row(0)], [prediction(0, True)], 42)
    with pytest.raises(ValueError, match="arrived for row"):
        hard_examples.build([row(0)], [prediction(1, False)], 42)


def test_mine_runs_the_base_model_in_chunks_and_reports_overall_progress():
    calls, reported = [], []

    class Backend:
        def load(self, adapter=None, model_id=None):
            calls.append(("load", adapter, model_id))
            return "model", "tokenizer"

        def evaluate(self, model, tokenizer, rows, progress, cancelled, split, pack):
            calls.append(("evaluate", len(rows), split))
            for i in range(len(rows)):
                progress(i + 1, len(rows), {})
            return {"predictions": [{"id": r["id"]} for r in rows]}

        def unload(self, model):
            calls.append(("unload", model))

    rows = [row(i) for i in range(hard_examples.CHUNK + 5)]
    found = hard_examples.mine(
        Backend(), {"model_id": "m"}, rows, lambda d, t, _p: reported.append((d, t)), lambda: False, None
    )
    assert [p["id"] for p in found] == [r["id"] for r in rows]
    assert calls[0] == ("load", None, "m") and calls[-1] == ("unload", "model")
    assert [c[1] for c in calls if c[0] == "evaluate"] == [hard_examples.CHUNK, 5]
    assert reported[-1] == (len(rows), len(rows))


def evidence(score=0.8):
    return {"score": score, "n": 2, "predictions": [{"correct": False}]}


def test_local_director_changes_the_second_hard_example_run_using_the_evidence(monkeypatch):
    for name in ("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    config = {"objective": "Improve math", "strategies": ["hard_sft"]}
    first = director.propose(config, evidence(), [])
    assert (first["strategy"], first["learning_rate"], first["lora_rank"]) == ("hard_sft", 0.0001, 16)
    dropped = director.propose(config, evidence(), [{"strategy": "hard_sft", "score": 0.7}])
    assert dropped["learning_rate"] == 0.00005
    flat = director.propose(config, evidence(), [{"strategy": "hard_sft", "score": 0.8}])
    assert flat["learning_rate"] == 0.0002


def test_hard_examples_are_allowed_on_modal_for_qa_tasks_only():
    assert strategies.STRATEGIES["hard_sft"]["backends"] == ("local", "modal")
    assert strategies.STRATEGIES["hard_sft"]["kinds"] == ("qa",)
    assert "hard_sft" in json.dumps(strategies.listing())
