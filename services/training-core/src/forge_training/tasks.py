"""A task: how a question is posed to the model and how its answer is checked."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from .checkers import Checker, CheckerSpec, Comparator, Extractor
from .modeling import PROMPT_INSTRUCTION, PROMPT_VERSION

Name = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_.-]{0,63}$")]


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    task_id: Name
    instruction: str = Field(min_length=1, max_length=4000)
    # Applied when a dataset is registered: rows are stored already rendered as question/answer.
    input_template: str = Field(default="{question}", min_length=1, max_length=4000)
    target_template: str = Field(default="{answer}", min_length=1, max_length=4000)
    prompt_version: Name
    checker: CheckerSpec
    max_new_tokens: int | None = Field(default=None, ge=1, le=4096)

    def make_checker(self) -> Checker:
        return Checker(self.checker)


# The original Forge task: GSM8K-style math with a strict final '#### <number>' line.
GSM8K_TASK = TaskSpec(
    task_id="gsm8k",
    instruction=PROMPT_INSTRUCTION,
    prompt_version=PROMPT_VERSION,
    checker=CheckerSpec(
        extract=Extractor(kind="final_marker", marker="####", grammar="forge_numeric_v1"),
        compare=Comparator(kind="numeric"),
    ),
)
