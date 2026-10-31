"""A task pack: one task's data, prompt, checker, splits, regression suite and labels."""

import hashlib
import json
from typing import Annotated, Literal

from forge_training.tasks import TaskSpec
from pydantic import BaseModel, ConfigDict, Field

PackId = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]{1,63}$")]


class Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Source(Spec):
    kind: Literal["hf_parquet", "upload"]
    repo: str | None = None
    revision: str = Field(min_length=1)
    license: str = Field(min_length=1)
    # The operator's statement that this data may be used for training. Required for uploads.
    data_terms: str | None = None


class Display(Spec):
    name: str = Field(min_length=3, max_length=80)
    objective: str = Field(min_length=10, max_length=500)
    benchmark_label: str = Field(min_length=1, max_length=120)
    scoring_label: str = Field(min_length=1, max_length=200)
    unit: str = Field(default="examples", min_length=1, max_length=24)


class Example(Spec):
    id: str = Field(min_length=1, max_length=96)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    rejected: str | None = None


class Capabilities(Spec):
    # Development answers are rewarded by Terse RLWorldActor, which compares numbers.
    terse_world: bool
    grpo: bool
    dpo: bool
    lm_eval_final: bool
    remote: bool


class TaskPack(Spec):
    id: PackId
    kind: Literal["qa", "extraction", "preference"]
    task: TaskSpec
    source: Source
    display: Display
    regression: tuple[Example, ...] = Field(min_length=1)
    # The pack whose prompt and checker score the regression rows. None: this pack.
    regression_pack: str | None = None
    regression_title: str = Field(default="Regression review", min_length=1, max_length=80)
    regression_label: str = Field(min_length=1, max_length=200)
    detectors: tuple[str, ...] = ()
    focus_tags: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    capabilities: Capabilities

    @property
    def sha256(self) -> str:
        body = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(body.encode()).hexdigest()

    @property
    def reward_version(self) -> str:
        # The Terse world reward kept its original name; every other pack names its checker.
        if self.capabilities.terse_world:
            return "exact-answer-v1"
        return f"{self.task.make_checker().checker_id}@{self.sha256[:12]}"
