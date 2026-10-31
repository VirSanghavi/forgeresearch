from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from world_stub import StubWorld

from forge_training.artifacts import directory_sha256, file_sha256, write_json
from forge_training.config import ExperimentConfig, Task
from forge_training.data import load_tasks, write_dataset
from forge_training.fixtures import make_grpo_fixture
from forge_training.grpo import NO_ANSWER, answer_action, group_advantages
from forge_training.worker import run_experiment, verify_run


def references(directory: Path) -> dict[str, str]:
    lines = (directory / "data/train.jsonl").read_text().splitlines()
    return {json.loads(line)["id"]: json.loads(line)["answer"] for line in lines}


def run_with(directory: Path, stub: StubWorld, monkeypatch) -> dict:
    for name, value in stub.environment.items():
        monkeypatch.setenv(name, value)
    return run_experiment(directory / "experiment.json")


@pytest.fixture(scope="module")
def grpo_run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("grpo") / "fixture"
    make_grpo_fixture(directory)
    with StubWorld(references(directory)) as stub, pytest.MonkeyPatch.context() as patch:
        result = run_with(directory, stub, patch)
        requests = list(stub.requests)
    return directory, result, requests


@pytest.mark.parametrize(
    ("completion", "answer"),
    [
        ("Work.\n#### 1,234", "1234"),
        ("#### -0.50", "-0.50"),
        ("#### 6", "6"),
        ("The answer is 6.", NO_ANSWER),
        ("#### 6 because", NO_ANSWER),
        ("#### " + "9" * 70, NO_ANSWER),
    ],
)
def test_actions_follow_the_development_scorers_final_line(completion, answer):
    assert answer_action(completion) == {"type": "submit_answer", "answer": answer}


@pytest.mark.parametrize("scale", ["group", "batch", "none"])
def test_group_advantages_match_the_pinned_trainer_formula(scale):
    import torch

    rewards = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]
    tensor = torch.tensor(rewards)
    grouped = tensor.view(-1, 4)
    centered = tensor - grouped.mean(dim=1).repeat_interleave(4)
    std = {"group": grouped.std(dim=1).repeat_interleave(4), "batch": tensor.std()}
    expected = centered if scale == "none" else centered / (std[scale] + 1e-4)
    assert group_advantages(rewards, 4, scale) == pytest.approx(expected.tolist(), abs=1e-6)


def test_grpo_settings_are_required_and_bounded(config):
    data = config.model_dump()
    with pytest.raises(ValueError, match="GRPO settings"):
        ExperimentConfig.model_validate({**data, "training": {"method": "grpo"}})
    with pytest.raises(ValueError, match="GRPO settings"):
        ExperimentConfig.model_validate({**data, "grpo": {}})
    grpo = {"method": "grpo", "batch_size": 3, "gradient_accumulation_steps": 1}
    with pytest.raises(ValueError, match="multiple of num_generations"):
        ExperimentConfig.model_validate({**data, "training": grpo, "grpo": {}})
    grpo = {"method": "grpo", "batch_size": 16, "gradient_accumulation_steps": 64}
    with pytest.raises(ValueError, match="At most 1000"):
        ExperimentConfig.model_validate({**data, "training": grpo, "grpo": {}})
    grpo = {"method": "grpo", "batch_size": 4, "max_steps": 10}
    with pytest.raises(ValueError, match="max_rollouts"):
        ExperimentConfig.model_validate({**data, "training": grpo, "grpo": {"max_rollouts": 100}})


def test_task_records_cannot_carry_reference_answers(config, tmp_path):
    from forge_training.config import DatasetSpec

    with pytest.raises(ValueError, match="Expected task records"):
        load_tasks(config.train)
    write_dataset(
        tmp_path,
        "tasks",
        [Task(id="t1", question="What is 1 + 1?")],
        source="unit-fixture",
        revision="v1",
        source_split="synthetic",
        license_name="CC0-1.0",
        purpose="train",
    )
    spec = DatasetSpec(
        records=str(tmp_path / "tasks.jsonl"), manifest=str(tmp_path / "tasks.manifest.json")
    )
    assert [task.id for task in load_tasks(spec)[0]] == ["t1"]
    records = tmp_path / "tasks.jsonl"
    records.write_text('{"id":"t1","question":"What is 1 + 1?","answer":"#### 2"}\n')
    manifest = json.loads((tmp_path / "tasks.manifest.json").read_text())
    (tmp_path / "tasks.manifest.json").unlink()
    write_json(
        tmp_path / "tasks.manifest.json", manifest | {"content_sha256": file_sha256(records)}
    )
    with pytest.raises(ValueError, match="answer"):
        load_tasks(spec)


@pytest.mark.integration
def test_world_rewards_drive_verified_updates_with_lineage(grpo_run):
    directory, result, requests = grpo_run
    output = directory / "run"
    assert result["status"] == "completed" and result["training_method"] == "grpo"
    assert not result["promotion_authorized"]
    verified = verify_run(output)["rollouts"]
    assert (verified["updates"], verified["rollouts"], verified["snapshots"]) == (6, 48, 7)
    training = json.loads((output / "training.json").read_text())
    assert training["reward_source"] == "terse_world_transitions"
    assert training["trainer"] == {"library": "trl", "version": "0.24.0", "class": "GRPOTrainer"}
    # One scoring request per update, each from the policy the previous update produced.
    assert len(requests) == 6
    policy = training["lineage"]["initial_policy_sha256"]
    for update, request in zip(training["updates"], requests, strict=True):
        assert json.loads(request)["policySha256"] == policy == update["rollout_policy_sha256"]
        assert update["policy_sha256_after"] != policy
        policy = update["policy_sha256_after"]
    assert policy == training["lineage"]["final_policy_sha256"]
    # Rewards varied within groups, so advantages were nonzero and drove the updates.
    rollouts = [json.loads(line) for line in (output / "rollouts.jsonl").read_text().splitlines()]
    assert any(record["advantage"] != 0 for record in rollouts)
    assert 0 < sum(record["reward"] for record in rollouts) < len(rollouts)
    # The trainer never saw a reference answer.
    staged = (output / "inputs/train.jsonl").read_text()
    assert "answer" not in staged and "####" not in staged
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert sum(event["type"] == "ROLLOUTS_SCORED" for event in events) == 6


@pytest.mark.integration
def test_rewards_without_signal_cannot_produce_a_checkpoint(tmp_path, monkeypatch):
    """Control: identical rewards give zero advantages, so the optimizer changes nothing."""
    directory = tmp_path / "fixture"
    make_grpo_fixture(directory)
    with StubWorld(references(directory), constant=0) as stub:
        with pytest.raises(RuntimeError, match="Worker subprocess failed"):
            run_with(directory, stub, monkeypatch)
    output = directory / "run"
    failure = json.loads((output / "failure.json").read_text())
    assert failure["status"] == "failed"
    assert "did not change trainable weights" in (output / "training.log").read_text()
    assert not (output / "checkpoint").exists() and not (output / "result.json").exists()


@pytest.mark.integration
def test_interrupted_scoring_retries_the_identical_rollouts(tmp_path, monkeypatch):
    directory = tmp_path / "fixture"
    make_grpo_fixture(directory)
    with StubWorld(references(directory)) as stub:
        stub.drops, stub.failures = 1, 2
        result = run_with(directory, stub, monkeypatch)
        requests = list(stub.requests)
    assert result["status"] == "completed"
    # The first update was sent four times, byte for byte, before it was scored.
    assert len(requests) == 6 + 3 and len(set(requests[:4])) == 1
    assert verify_run(directory / "run")["rollouts"]["updates"] == 6


@pytest.mark.integration
def test_a_reward_altered_after_training_fails_verification(grpo_run, tmp_path):
    directory, _, _ = grpo_run
    copy = tmp_path / "run"
    shutil.copytree(directory / "run", copy)
    path = copy / "rollouts.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    target = next(record for record in records if record["reward"] == 0)
    target["reward"] = target["transition"]["reward"] = 1
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
    # A writer that can rewrite every file, including the manifest, is still caught.
    manifest = json.loads((copy / "manifest.json").read_text())
    manifest["files"]["rollouts.jsonl"] = file_sha256(path)
    (copy / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="advantages do not follow|summary differs"):
        verify_run(copy)


@pytest.mark.integration
def test_grpo_without_the_world_gateway_fails_closed(tmp_path, monkeypatch):
    directory = tmp_path / "fixture"
    make_grpo_fixture(directory)
    monkeypatch.delenv("FORGE_WORLD_GATEWAY_URL", raising=False)
    monkeypatch.delenv("FORGE_WORLD_GATEWAY_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="Worker subprocess failed"):
        run_experiment(directory / "experiment.json")
    assert "world gateway" in (directory / "run/training.log").read_text()


def continued(parent_fixture: Path, directory: Path, **changes) -> Path:
    """A GRPO fixture that continues the parent run's checkpoint on the same base model."""
    make_grpo_fixture(directory)
    path = directory / "experiment.json"
    config = json.loads(path.read_text())
    parent = parent_fixture / "run" / "checkpoint"
    config["model"]["model_id"] = str(parent_fixture / "base-model")
    config["grpo"]["init_adapter"] = {"path": str(parent), "sha256": directory_sha256(parent)}
    for section, values in changes.items():
        config[section].update(values)
    path.unlink()
    write_json(path, config)
    return path


@pytest.mark.integration
def test_grpo_continues_a_parent_adapter_and_is_measured_against_it(
    grpo_run, tmp_path, monkeypatch
):
    parent_fixture, _, _ = grpo_run
    child = tmp_path / "child"
    continued(parent_fixture, child)
    with StubWorld(references(child)) as stub:
        result = run_with(child, stub, monkeypatch)
    parent = parent_fixture / "run" / "checkpoint"
    output = child / "run"
    # The baseline is the starting policy, so the comparison isolates this run's change.
    baseline = json.loads((output / "baseline.json").read_text())
    assert baseline["checkpoint_sha256"] == directory_sha256(parent)
    assert result["starting_checkpoint_sha256"] == directory_sha256(parent)
    assert directory_sha256(output / "inputs/parent-adapter") == directory_sha256(parent)
    training = json.loads((output / "training.json").read_text())
    parent_training = json.loads((parent_fixture / "run/training.json").read_text())
    assert (
        training["lineage"]["initial_policy_sha256"]
        == parent_training["lineage"]["final_policy_sha256"]
    )
    assert verify_run(output)["rollouts"]["updates"] == 6


@pytest.mark.integration
def test_a_mismatched_parent_adapter_is_refused(grpo_run, tmp_path, monkeypatch):
    parent_fixture, _, _ = grpo_run
    wrong_hash = tmp_path / "hash"
    path = continued(parent_fixture, wrong_hash)
    config = json.loads(path.read_text())
    config["grpo"]["init_adapter"]["sha256"] = "0" * 64
    path.unlink()
    write_json(path, config)
    with pytest.raises(ValueError, match="Starting adapter checksum mismatch"):
        run_experiment(path)
    other_rank = tmp_path / "rank"
    continued(parent_fixture, other_rank, training={"lora_rank": 8, "lora_alpha": 16})
    with StubWorld(references(other_rank)) as stub:
        with pytest.raises(RuntimeError, match="Worker subprocess failed"):
            run_with(other_rank, stub, monkeypatch)
    assert "LoRA settings differ" in (other_rank / "run/training.log").read_text()
