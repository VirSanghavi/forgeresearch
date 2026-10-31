"""Strict workflow contracts. Manifests contain secret names, never secret values."""

from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Name = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")]


def relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or "\x00" in value:
        raise ValueError("Paths must be relative, without traversal or backslashes")
    if str(path) in (".", ".forge") or path.parts[0] == ".forge":
        raise ValueError("The .forge namespace is reserved")
    return str(path)


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Input(Contract):
    path: str
    source: str
    task: Name | None = None

    _paths = field_validator("path", "source")(relative_path)


class Sandbox(Contract):
    template: str | None = None
    snapshot: str | None = None
    snapshot_on_success: bool = False
    auto_delete_seconds: int = Field(default=3600, ge=60, le=86400)

    @model_validator(mode="after")
    def one_origin(self):
        if self.template and self.snapshot:
            raise ValueError("Choose a template or snapshot, not both")
        return self


class Check(Contract):
    artifact: str
    pointer: str = ""
    operator: Literal["eq", "ne", "ge", "le", "gt", "lt"] = "eq"
    value: str | bool | int | float | None

    _path = field_validator("artifact")(relative_path)

    @field_validator("pointer")
    @classmethod
    def json_pointer(cls, value):
        if value and not value.startswith("/"):
            raise ValueError("JSON pointers must be empty or start with /")
        return value


class Task(Contract):
    id: Name
    command: list[str] = Field(min_length=1, max_length=256)
    backend: Literal["local", "superserve"] = "local"
    needs: list[Name] = Field(default_factory=list)
    inputs: list[Input] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    checks: list[Check] = Field(default_factory=list)
    env: list[Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]] = Field(default_factory=list)
    timeout_seconds: int = Field(default=300, ge=1, le=86400)
    max_output_bytes: int = Field(default=16 * 1024 * 1024, ge=1024, le=2**30)
    max_artifact_bytes: int = Field(default=256 * 1024 * 1024, ge=1, le=2**31)
    attempts: int = Field(default=1, ge=1, le=5)
    retry_safe: bool = False
    retry_delay_seconds: float = Field(default=1, ge=0, le=300)
    resources: list[Name] = Field(default_factory=list)
    sandbox: Sandbox = Field(default_factory=Sandbox)

    @field_validator("command")
    @classmethod
    def command_is_valid(cls, value):
        if not value[0] or any("\x00" in arg for arg in value):
            raise ValueError("Command must have an executable and no NUL bytes")
        return value

    @field_validator("outputs")
    @classmethod
    def output_paths(cls, value):
        return [relative_path(p) for p in value]

    @model_validator(mode="after")
    def safe_retry(self):
        if self.attempts > 1 and not self.retry_safe:
            raise ValueError("Multiple attempts require retry_safe=true")
        if any(check.artifact not in self.outputs for check in self.checks):
            raise ValueError("Checks must reference a declared output")
        for values in (self.needs, self.outputs, self.env, self.resources, [i.path for i in self.inputs]):
            if len(values) != len(set(values)):
                raise ValueError("Duplicate dependencies, paths, environment names or resources")
        return self


class Workflow(Contract):
    version: Literal[1] = 1
    name: Name
    max_parallel: int = Field(default=2, ge=1, le=16)
    max_seconds: int = Field(default=3600, ge=1, le=604800)
    fail_fast: bool = False
    tasks: list[Task] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def graph(self):
        tasks = {task.id: task for task in self.tasks}
        if len(tasks) != len(self.tasks):
            raise ValueError("Task IDs must be unique")
        visited = set()
        pending = set(tasks)
        for task in self.tasks:
            if set(task.needs) - tasks.keys():
                raise ValueError(f"Unknown dependencies for {task.id}")
            for item in task.inputs:
                if item.task and (item.task not in task.needs or item.source not in tasks[item.task].outputs):
                    raise ValueError(f"Input {item.path} must name a declared output of a direct dependency")
        while pending:
            ready = {key for key in pending if set(tasks[key].needs) <= visited}
            if not ready:
                raise ValueError("Workflow dependencies contain a cycle")
            visited.update(ready)
            pending.difference_update(ready)
        return self
