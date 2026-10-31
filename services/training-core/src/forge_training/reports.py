"""Strict evaluation read/write contracts and independently recomputed scores."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .config import SHA256, Contract, DatasetManifest, EvaluationSpec, ModelSpec
from .scoring import score, wilson_interval


class SamplePrediction(Contract):
    id: str
    question: str
    prompt_sha256: SHA256
    reference_answer: str
    prediction: str
    correct: bool
    diagnosis: Literal["correct", "incorrect_answer", "invalid_answer_format"]
    generated_tokens: Annotated[int, Field(ge=0)]
    hit_generation_limit: bool

    @model_validator(mode="after")
    def validate_score(self) -> SamplePrediction:
        correct, diagnosis = score(self.prediction, f"#### {self.reference_answer}")
        if self.correct != correct or self.diagnosis != diagnosis:
            raise ValueError("Stored sample score does not match its prediction and reference")
        return self


class Protocol(Contract):
    evaluation: EvaluationSpec
    seed: int
    device: Literal["cpu", "cuda", "mps"]
    precision: Literal["float32", "bfloat16"]
    attention_implementation: Literal["eager", "sdpa"] = "eager"
    device_memory_limit_mb: Annotated[int, Field(ge=256, le=131072)] | None = None
    prompt_version: Literal["math_completion_v1"]
    instruction: str
    do_sample: Literal[False]
    num_beams: Literal[1]
    runtime: dict[str, str | None]


class EvaluationReport(Contract):
    schema_version: Literal[1]
    campaign_id: str
    experiment_id: str
    started_at: str
    completed_at: str
    evaluation_role: Literal["development"]
    benchmark_claim: Literal["custom development diagnostic; not an official GSM8K score"]
    model: ModelSpec
    checkpoint_sha256: SHA256 | None
    dataset: DatasetManifest
    protocol: Protocol
    scorer: Literal["forge_numeric_v1"]
    num_examples: Annotated[int, Field(ge=1)]
    num_correct: Annotated[int, Field(ge=0)]
    accuracy: Annotated[float, Field(ge=0, le=1)]
    accuracy_wilson_95: Annotated[list[float], Field(min_length=2, max_length=2)]
    elapsed_seconds: Annotated[float, Field(ge=0)]
    samples: Annotated[list[SamplePrediction], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_summary(self) -> EvaluationReport:
        if self.dataset.purpose != "development":
            raise ValueError("Research evaluation must use development data")
        if self.num_examples != len(self.samples) or self.num_examples != self.dataset.num_examples:
            raise ValueError("Evaluation count does not match sample or dataset count")
        if len({row.id for row in self.samples}) != self.num_examples:
            raise ValueError("Evaluation samples contain duplicate IDs")
        correct = sum(row.correct for row in self.samples)
        if self.num_correct != correct or self.accuracy != correct / self.num_examples:
            raise ValueError("Aggregate score does not match sample-level evidence")
        if self.accuracy_wilson_95 != wilson_interval(correct, self.num_examples):
            raise ValueError("Confidence interval does not match evaluation counts")
        for row in self.samples:
            if row.generated_tokens > self.protocol.evaluation.max_new_tokens:
                raise ValueError("Generated tokens exceed the declared decoding limit")
            if row.hit_generation_limit != (
                row.generated_tokens == self.protocol.evaluation.max_new_tokens
            ):
                raise ValueError("Generation limit flag does not match token count")
        return self
