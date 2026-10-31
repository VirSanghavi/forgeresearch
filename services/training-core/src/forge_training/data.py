"""Verified JSONL input and research-only train/development partitioning."""

from __future__ import annotations

import random
import re
import unicodedata
from pathlib import Path

from .artifacts import file_sha256, write_json
from .config import DatasetManifest, DatasetSpec, Example, Task


def question_key(question: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", question).casefold().split())


def _load(spec: DatasetSpec, purpose: str, kind: str) -> tuple[list, DatasetManifest]:
    path = Path(spec.records)
    manifest = DatasetManifest.model_validate_json(Path(spec.manifest).read_text())
    if manifest.purpose != purpose:
        raise ValueError(f"Expected {purpose} data, got {manifest.purpose}")
    if manifest.record_kind != kind:
        raise ValueError(f"Expected {kind} records, got {manifest.record_kind}")
    if file_sha256(path) != manifest.content_sha256:
        raise ValueError(f"Dataset checksum mismatch: {path}")
    record = Example if kind == "example" else Task
    rows = []
    # JSON strings may contain Unicode separators (including real GSM8K rows).
    # str.splitlines() would split those inside strings; JSONL boundaries are LF.
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL record at {path}:{index}")
            # Strict records: a task line that carries an answer field is rejected.
            rows.append(record.model_validate_json(line))
    if len(rows) != manifest.num_examples:
        raise ValueError("Dataset count does not match manifest")
    if len({row.id for row in rows}) != len(rows):
        raise ValueError("Duplicate example IDs within dataset")
    if len({question_key(row.question) for row in rows}) != len(rows):
        raise ValueError("Duplicate normalized questions within dataset")
    if any(not question_key(row.question) for row in rows):
        raise ValueError("Whitespace-only questions or answers are not allowed")
    if kind == "example" and any(not row.answer.strip() for row in rows):
        raise ValueError("Whitespace-only questions or answers are not allowed")
    return rows, manifest


def load_dataset(spec: DatasetSpec, purpose: str) -> tuple[list[Example], DatasetManifest]:
    return _load(spec, purpose, "example")


def load_tasks(spec: DatasetSpec) -> tuple[list[Task], DatasetManifest]:
    """Answer-free training prompts for reward-driven optimization."""
    return _load(spec, "train", "task")


def verify_separation(train: list[Example | Task], development: list[Example]) -> None:
    if {row.id for row in train} & {row.id for row in development}:
        raise ValueError("Train/development example IDs overlap")
    if {question_key(row.question) for row in train} & {
        question_key(row.question) for row in development
    }:
        raise ValueError("Train/development questions overlap")


def write_dataset(
    output: Path,
    name: str,
    rows: list[Example] | list[Task],
    *,
    source: str,
    revision: str,
    source_split: str,
    license_name: str,
    purpose: str | None = None,
) -> None:
    path = output / f"{name}.jsonl"
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(row.model_dump_json() + "\n")
    manifest = DatasetManifest(
        dataset_id=name,
        purpose=purpose or name,
        source=source,
        source_revision=revision,
        source_split=source_split,
        license=license_name,
        content_sha256=file_sha256(path),
        num_examples=len(rows),
        record_kind="task" if rows and isinstance(rows[0], Task) else "example",
    )
    write_json(output / f"{name}.manifest.json", manifest.model_dump())


def prepare_gsm8k(output: Path, revision: str, seed: int, train_size: int, dev_size: int) -> None:
    """Never requests or loads the official test split."""
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("GSM8K revision must be a pinned Hub commit")
    if train_size < 1 or dev_size < 1:
        raise ValueError("Both partition sizes must be positive")
    # A named dataset with split="train" can still download/prepare ALL splits.
    # Fetch only the pinned train file and give the parquet builder only that file.
    train_file = hf_hub_download(
        repo_id="openai/gsm8k",
        repo_type="dataset",
        revision=revision,
        filename="main/train-00000-of-00001.parquet",
    )
    dataset = load_dataset("parquet", data_files={"train": train_file}, split="train")
    if train_size + dev_size > len(dataset):
        raise ValueError("Requested partition sizes exceed official train split")
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    output.mkdir(parents=True, exist_ok=False)
    for name, selected in (
        ("train", indices[:train_size]),
        ("development", indices[train_size : train_size + dev_size]),
    ):
        rows = [Example(id=f"gsm8k-train-{i}", **dict(dataset[i])) for i in selected]
        write_dataset(
            output,
            name,
            rows,
            source="openai/gsm8k",
            revision=revision,
            source_split="train",
            license_name="MIT",
        )
    write_json(
        output / "partition.json",
        {
            "source": "openai/gsm8k",
            "revision": revision,
            "source_split": "train",
            "source_file": "main/train-00000-of-00001.parquet",
            "seed": seed,
            "train_indices": indices[:train_size],
            "development_indices": indices[train_size : train_size + dev_size],
            "official_test_loaded": False,
        },
    )
