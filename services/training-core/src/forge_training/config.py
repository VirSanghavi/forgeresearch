"""Strict write contracts. Paths are resolved relative to the configuration file."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SHA256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class ModelSpec(Contract):
    source: Literal["hub", "local"] = "hub"
    model_status: Literal["base", "instruct", "fixture"] = "instruct"
    model_id: Annotated[str, Field(min_length=1)]
    revision: str

    @model_validator(mode="after")
    def pinned(self) -> ModelSpec:
        expected = r"[0-9a-f]{40}" if self.source == "hub" else r"[0-9a-f]{64}"
        if not re.fullmatch(expected, self.revision):
            raise ValueError("Use a pinned Hub commit or a local directory SHA-256, never main.")
        if self.source == "hub" and not re.fullmatch(r"[\w.-]+/[\w.-]+", self.model_id):
            raise ValueError("Hub model_id must be an explicit organization/model identifier")
        return self


class DatasetSpec(Contract):
    records: str
    manifest: str


class DatasetManifest(Contract):
    schema_version: Literal[1] = 1
    dataset_id: Identifier
    purpose: Literal["train", "development"]
    source: Annotated[str, Field(min_length=1)]
    source_revision: Annotated[str, Field(min_length=1)]
    source_split: Literal["train", "synthetic"]
    license: Annotated[str, Field(min_length=1)]
    content_sha256: SHA256
    num_examples: Annotated[int, Field(ge=1)]
    # "task" records carry no reference answer; reward-driven training reads only these.
    record_kind: Literal["example", "task"] = "example"


class Example(Contract):
    id: Identifier
    question: Annotated[str, Field(min_length=1, max_length=32000)]
    answer: Annotated[str, Field(min_length=1, max_length=32000)]


class Task(Contract):
    """A reward-driven training prompt. Its answer stays with the trusted task service."""

    id: Identifier
    question: Annotated[str, Field(min_length=1, max_length=32000)]


class TrainingSpec(Contract):
    method: Literal["sft_lora", "grpo"] = "sft_lora"
    learning_rate: Annotated[float, Field(ge=1e-7, le=1e-2)] = 1e-4
    max_steps: Annotated[int, Field(ge=1, le=10000)] = 100
    batch_size: Annotated[int, Field(ge=1, le=16)] = 1
    gradient_accumulation_steps: Annotated[int, Field(ge=1, le=64)] = 4
    max_sequence_length: Annotated[int, Field(ge=32, le=4096)] = 512
    lora_rank: Annotated[int, Field(ge=1, le=128)] = 8
    lora_alpha: Annotated[int, Field(ge=1, le=256)] = 16
    target_modules: Annotated[list[str], Field(min_length=1)] = ["q_proj", "v_proj"]
    max_seconds: Annotated[int, Field(ge=1, le=86400)] = 1800
    max_training_tokens: Annotated[int, Field(ge=1, le=100000000)] = 1000000
    completion_only_logits: bool = False


class EvaluationSpec(Contract):
    scorer: Literal["forge_numeric_v1"] = "forge_numeric_v1"
    max_new_tokens: Annotated[int, Field(ge=1, le=1024)] = 256
    max_prompt_tokens: Annotated[int, Field(ge=16, le=4096)] = 1024
    max_seconds: Annotated[int, Field(ge=1, le=86400)] = 1800
    # Greedy generation in left-padded batches of this size. Part of the frozen protocol:
    # padding can change near-tie decisions, so baseline and candidates must share it.
    batch_size: Annotated[int, Field(ge=1, le=64)] = 1


class AdapterRef(Contract):
    """An existing LoRA adapter directory, identified by its content hash."""

    path: str
    sha256: SHA256


class GRPOSpec(Contract):
    """Group-relative policy optimization. Rewards come only from Terse world transitions.

    `training.batch_size` completions form one loss micro-batch; an update generates
    `batch_size * gradient_accumulation_steps` completions, `num_generations` per prompt.
    """

    num_generations: Annotated[int, Field(ge=2, le=16)] = 4
    max_completion_tokens: Annotated[int, Field(ge=1, le=1024)] = 256
    max_prompt_tokens: Annotated[int, Field(ge=16, le=4096)] = 1024
    temperature: Annotated[float, Field(gt=0, le=2)] = 1.0
    top_p: Annotated[float, Field(gt=0, le=1)] = 1.0
    beta: Annotated[float, Field(ge=0, le=1)] = 0.0
    epsilon: Annotated[float, Field(gt=0, le=1)] = 0.2
    loss_type: Literal["grpo", "dr_grpo", "dapo", "bnpo"] = "dapo"
    scale_rewards: Literal["group", "batch", "none"] = "group"
    reward_version: Literal["exact-numeric-v1"] = "exact-numeric-v1"
    action_parser: Literal["final_answer_v1"] = "final_answer_v1"
    # Bounds remote world transitions: one per completion, each an actor command.
    max_rollouts: Annotated[int, Field(ge=1, le=100000)] = 4096
    # Save the adapter after every Nth update (0: only the initial and final states).
    snapshot_every: Annotated[int, Field(ge=0, le=10000)] = 1
    gradient_checkpointing: bool = False
    request_timeout_seconds: Annotated[int, Field(ge=1, le=600)] = 120
    # Continue from this adapter (for example an approved SFT checkpoint) instead of a fresh
    # LoRA. The run's baseline is then this starting policy, not the untrained model.
    init_adapter: AdapterRef | None = None


class ExperimentConfig(Contract):
    schema_version: Literal[1] = 1
    campaign_id: Identifier
    experiment_id: Identifier
    model: ModelSpec
    train: DatasetSpec
    development: DatasetSpec
    training: TrainingSpec = TrainingSpec()
    grpo: GRPOSpec | None = None
    evaluation: EvaluationSpec = EvaluationSpec()
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)] = 42
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    precision: Literal["float32", "bfloat16"] = "float32"
    attention_implementation: Literal["eager", "sdpa"] = "eager"
    device_memory_limit_mb: Annotated[int, Field(ge=256, le=131072)] | None = None
    max_model_parameters: Annotated[int, Field(ge=1, le=3000000000)] = 600000000
    output_dir: str

    @model_validator(mode="after")
    def method_settings(self) -> ExperimentConfig:
        if (self.training.method == "grpo") != (self.grpo is not None):
            raise ValueError("GRPO settings are required for, and only allowed with, method grpo")
        if self.grpo is not None:
            per_update = self.training.batch_size * self.training.gradient_accumulation_steps
            # Rollout IDs carry a three-digit index within the update.
            if per_update > 1000:
                raise ValueError("At most 1000 completions per GRPO update")
            if per_update % self.grpo.num_generations:
                raise ValueError(
                    "batch_size x gradient_accumulation_steps must be a multiple of num_generations"
                )
            if self.training.max_steps * per_update > self.grpo.max_rollouts:
                raise ValueError("Planned rollouts exceed grpo.max_rollouts")
        return self


def load_config(path: Path) -> ExperimentConfig:
    config = ExperimentConfig.model_validate_json(path.read_text())
    base = path.resolve().parent

    def resolve(value: str) -> str:
        return str((base / value).resolve())

    data = config.model_dump()
    data["output_dir"] = resolve(config.output_dir)
    for name in ("train", "development"):
        data[name] = {key: resolve(value) for key, value in data[name].items()}
    if config.model.source == "local":
        data["model"]["model_id"] = resolve(config.model.model_id)
    if config.grpo is not None and config.grpo.init_adapter is not None:
        data["grpo"]["init_adapter"]["path"] = resolve(config.grpo.init_adapter.path)
    return ExperimentConfig.model_validate(data)


def starting_checkpoint(config: ExperimentConfig) -> str | None:
    """Hash of the adapter training starts from; None means the untrained base model."""
    adapter = config.grpo.init_adapter if config.grpo is not None else None
    return adapter.sha256 if adapter is not None else None
