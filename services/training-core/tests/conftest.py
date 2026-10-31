from __future__ import annotations

from pathlib import Path

import pytest

from forge_training.artifacts import write_json
from forge_training.config import DatasetSpec, Example, ExperimentConfig
from forge_training.data import write_dataset


@pytest.fixture
def config(tmp_path: Path) -> ExperimentConfig:
    for purpose, number in (("train", 2), ("development", 3)):
        write_dataset(
            tmp_path,
            purpose,
            [
                Example(
                    id=f"{purpose}-1",
                    question=f"What is {number} + 1?",
                    answer=f"#### {number + 1}",
                )
            ],
            source="unit-fixture",
            revision="v1",
            source_split="synthetic",
            license_name="CC0-1.0",
        )
    return ExperimentConfig(
        campaign_id="unit-campaign",
        experiment_id="unit-experiment",
        model={"model_id": "Qwen/Qwen2.5-0.5B-Instruct", "revision": "a" * 40},
        train=DatasetSpec(
            records=str(tmp_path / "train.jsonl"), manifest=str(tmp_path / "train.manifest.json")
        ),
        development=DatasetSpec(
            records=str(tmp_path / "development.jsonl"),
            manifest=str(tmp_path / "development.manifest.json"),
        ),
        output_dir=str(tmp_path / "run"),
    )


@pytest.fixture
def config_file(config: ExperimentConfig, tmp_path: Path) -> Path:
    path = tmp_path / "experiment.json"
    write_json(path, config.model_dump())
    return path
