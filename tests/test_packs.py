"""Task packs: GSM8K is the built-in pack, and routing it through packs changes nothing."""

import random
from decimal import Decimal, InvalidOperation

import pytest
from fastapi.testclient import TestClient

from forge_training.config import Example
from forge_training.data import verify_separation
from forge_training.scoring import extract_answer
from services.core import packs, protocol, store
from services.core.packs import gsm8k


class FakeSplit:
    """Stands in for a datasets split: len() and integer indexing, like the parquet loader."""

    def __init__(self, prefix, n):
        self.rows = [
            {
                "question": f"{prefix} question {i}: what is {i} plus {i}?",
                "answer": f"{i} + {i}\n#### {2 * i}",
            }
            for i in range(n)
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


@pytest.fixture
def fake_gsm8k(monkeypatch):
    splits = {"train": FakeSplit("train", 600), "test": FakeSplit("test", 200)}
    monkeypatch.setattr(gsm8k, "_parquet", lambda split, key: splits[split])
    monkeypatch.setattr(store, "campaigns", lambda: [])
    return splits


def legacy_benchmark_data(config, splits):
    """services/core/training.benchmark_data before task packs, verbatim apart from the loader."""
    ds = splits["train"]
    ids = list(range(len(ds)))
    random.Random(config["seed"]).shuffle(ids)
    dev_ids = ids[: config["dev_samples"]]
    train_ids = ids[config["dev_samples"] : config["dev_samples"] + 384]
    train = [dict(ds[i]) | {"id": f"train-{i}"} for i in train_ids]
    dev = [dict(ds[i]) | {"id": f"train-{i}"} for i in dev_ids]
    verify_separation([Example(**r) for r in train], [Example(**r) for r in dev])

    def sealed():
        test = splits["test"]
        available = list(range(len(test)))
        rng = random.Random(config["seed"] + 1)
        chosen = rng.sample(available, config["final_samples"])
        return [dict(test[i]) | {"id": f"test-{i}"} for i in chosen]

    return train, dev, sealed


@pytest.mark.parametrize("seed", [0, 42, 7])
def test_gsm8k_pack_splits_match_the_original_loader(fake_gsm8k, seed):
    config = {"seed": seed, "dev_samples": 24, "final_samples": 12}
    train, dev, sealed = packs.data(packs.get("gsm8k"), config)
    old_train, old_dev, old_sealed = legacy_benchmark_data(config, fake_gsm8k)
    assert train == old_train and dev == old_dev
    assert sealed() == old_sealed()


def test_sealed_exposure_counts_only_campaigns_on_the_same_pack(fake_gsm8k, monkeypatch):
    config = {"seed": 42, "dev_samples": 8, "final_samples": 12}
    *_, sealed = packs.data(packs.get(), config)
    first = sealed()
    exposed = {"config": {}, "sealed_ids": [r["id"] for r in first]}
    other_pack = {"config": {"task_pack": "support-tickets"}, "sealed_ids": ["test-0", "test-1"]}
    monkeypatch.setattr(store, "campaigns", lambda: [exposed, other_pack])
    again = sealed()
    assert not {r["id"] for r in again} & {r["id"] for r in first}


def legacy_score(output, reference):
    def extract(text):
        value = extract_answer(text)
        return str(value) if value is not None else ""

    answer, expected = extract(output), extract(reference)
    try:
        correct = Decimal(answer) == Decimal(expected)
    except InvalidOperation:
        correct = False
    return answer, expected, correct


def test_collect_scores_like_the_original_protocol():
    rows = [
        {"id": "a", "question": "q1", "answer": "work\n#### 42"},
        {"id": "b", "question": "q2", "answer": "#### 1,200"},
        {"id": "c", "question": "q3", "answer": "#### 7"},
    ]
    outputs = ["Reasoning\n#### 42", "#### 1200.0", "The answer is 7"]
    raw = [
        {"id": r["id"], "question": r["question"], "output": o, "finish_reason": "stop"}
        for r, o in zip(rows, outputs, strict=True)
    ]
    result = protocol.collect(rows, iter(raw), lambda *a: None, "development", None, 0.0, "cpu", seconds=1.0)
    for row, output, p in zip(rows, outputs, result["predictions"], strict=True):
        answer, expected, correct = legacy_score(output, row["answer"])
        assert list(p) == [
            "id",
            "question",
            "output",
            "answer",
            "expected",
            "correct",
            "finish_reason",
            "environment_id",
        ]
        assert (p["answer"], p["expected"], p["correct"]) == (answer, expected, correct)
    assert result["benchmark"] == "GSM8K subset"
    assert (
        result["scoring"] == "forge_numeric_v1, strict final #### numeric line (shared training-core scorer)"
    )
    assert result["prompt_version"] == "math_completion_v1"
    assert result["dataset_revision"] == protocol.DATASET_REVISION == gsm8k.DATASET_REVISION
    assert result["correct"] == 2


def test_gsm8k_pack_keeps_its_original_reward_and_regression_suite():
    pack = packs.get()
    assert pack.reward_version == "exact-answer-v1"
    assert [r.answer for r in pack.regression] == ["#### 42", "#### 9", "#### 88", "#### 41"]
    assert pack.capabilities.terse_world and pack.capabilities.grpo and pack.capabilities.lm_eval_final
    checker = pack.task.make_checker()
    for row in pack.regression:
        checker.check_reference(row.answer)
    with pytest.raises(KeyError):
        packs.get("no-such-pack")


def test_api_lists_packs_and_rejects_unknown_ones(monkeypatch):
    from services import api

    monkeypatch.setattr(api.store, "campaigns", lambda: [])
    client = TestClient(api.app, base_url="http://127.0.0.1:8000")
    listed = client.get("/packs").json()
    assert [p["id"] for p in listed] == ["gsm8k"]
    assert listed[0]["display"]["unit"] == "problems"
    response = client.post("/campaigns", json={"task_pack": "support-tickets"})
    assert response.status_code == 404
