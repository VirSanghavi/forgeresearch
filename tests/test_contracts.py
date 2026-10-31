from concurrent.futures import ThreadPoolExecutor
import pytest
from pydantic import ValidationError
from services.core.contracts import CampaignConfig, Action
from services.core.director import Proposal
from services.core import store
from forge_training.scoring import score


def test_campaign_rejects_unknown_fields_and_unsafe_limits():
    for extra in [
        {"shell": "echo surprise"},
        {"train_steps": 10000},
        {"max_seconds": -1},
        {"model_id": "unapproved/model"},
    ]:
        with pytest.raises(ValidationError):
            CampaignConfig(**extra)


def test_director_cannot_change_strategy_space_or_write_paths():
    base = {
        "strategy": "broad_sft",
        "learning_rate": 0.0001,
        "lora_rank": 8,
        "hypothesis": "Test measured arithmetic failures",
        "focus": "arithmetic",
    }
    assert Proposal(**base).strategy == "broad_sft"
    for update in [
        {"strategy": "shell"},
        {"learning_rate": 100},
        {"lora_rank": 4096},
        {"output_path": "/tmp/outside"},
    ]:
        with pytest.raises(ValidationError):
            Proposal(**(base | update))


def test_actions_are_strict_and_bounded():
    with pytest.raises(ValidationError):
        Action(episode_id="e", action_id="a", expected_version=-1, answer="3")
    with pytest.raises(ValidationError):
        Action(episode_id="e", action_id="a", expected_version=0, answer="3", reward=1)


def test_shared_scorer_avoids_last_number_false_positives():
    assert score("It may be 42, but I have no final answer", "#### 42") == (False, "invalid_answer_format")
    assert score("Reasoning.\n#### 1,200", "#### 1200")[0]
    assert not score("#### 12", "#### 1200")[0]


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA", tmp_path)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    monkeypatch.setattr(store, "ARTIFACTS", artifacts)
    return tmp_path


def test_one_bad_external_read_row_does_not_blank_collection(isolated_store):
    store.save({"id": "valid", "status": "completed"})
    with store.db() as c:
        c.execute("INSERT INTO campaigns VALUES(?,?)", ("bad", "{invalid"))
        for identity, payload in [("scalar", "42"), ("null", "null"), ("missing", '{"id":"incomplete"}')]:
            c.execute("INSERT INTO campaigns VALUES(?,?)", (identity, payload))
    assert [s["id"] for s in store.campaigns()] == ["valid"]


def test_durable_journal_keeps_concurrent_events(isolated_store):
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: store.event("test", "sample", str(i)), range(25)))
    events = store.events("test")
    assert len(events) == 25
    assert len({e["id"] for e in events}) == 25
    assert len((store.ARTIFACTS / "test/events.jsonl").read_text().splitlines()) == 25


def test_evidence_before_subscription_is_recoverable(isolated_store):
    first = store.event("c", "completed", "Completed before observer attached")
    assert store.events("c")[0]["id"] == first["id"]
    assert store.events("c", first["timestamp"]) == []
