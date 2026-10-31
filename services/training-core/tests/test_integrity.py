from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from forge_training.artifacts import directory_sha256
from forge_training.config import DatasetManifest, Example, ExperimentConfig, load_config
from forge_training.data import load_dataset, prepare_gsm8k, verify_separation
from forge_training.worker import validate_inputs


@pytest.mark.parametrize(
    "change",
    [
        {"unapproved_shell_command": "echo unsafe"},
        {"training": {"max_steps": True}},
        {"training": {"learning_rate": float("nan")}},
        {"model": {"model_id": "Qwen/Qwen2.5-0.5B-Instruct", "revision": "main"}},
    ],
)
def test_config_rejects_unapproved_or_unbounded_values(config, change):
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(config.model_dump() | change)


def test_relative_paths_use_config_location(config, config_file, tmp_path):
    body = json.loads(config_file.read_text())
    body["output_dir"] = "new-run"
    config_file.write_text(json.dumps(body))
    assert load_config(config_file).output_dir == str(tmp_path / "new-run")


def test_checksum_detects_dataset_edits(config):
    path = Path(config.train.records)
    path.write_text(path.read_text().replace("#### 3", "#### 999"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_dataset(config.train, "train")


def test_no_sealed_test_manifest_is_accepted(config):
    body = json.loads(Path(config.development.manifest).read_text())
    body["source_split"] = "test"
    with pytest.raises(ValidationError):
        DatasetManifest.model_validate(body)


def test_normalized_question_leakage_is_rejected():
    train = [Example(id="a", question="What is ２ + 1?", answer="#### 3")]
    dev = [Example(id="b", question="WHAT  is 2 + 1?", answer="#### 3")]
    with pytest.raises(ValueError, match="questions overlap"):
        verify_separation(train, dev)


def test_duplicate_cross_partition_ids_are_rejected():
    with pytest.raises(ValueError, match="IDs overlap"):
        verify_separation(
            [Example(id="same", question="one?", answer="#### 1")],
            [Example(id="same", question="two?", answer="#### 2")],
        )


def test_input_and_output_directories_cannot_overlap(config):
    body = config.model_dump() | {"output_dir": str(Path(config.train.records).parent)}
    with pytest.raises(ValueError, match="contain input"):
        validate_inputs(ExperimentConfig.model_validate(body))


def test_directory_hash_covers_names_contents_and_rejects_symlinks(tmp_path):
    (tmp_path / "weights").write_text("actual-weights")
    original = directory_sha256(tmp_path)
    (tmp_path / "weights").rename(tmp_path / "renamed-weights")
    assert directory_sha256(tmp_path) != original
    (tmp_path / "link").symlink_to(tmp_path / "renamed-weights")
    with pytest.raises(ValueError, match="Symlinks"):
        directory_sha256(tmp_path)


def test_gsm8k_requests_only_the_pinned_training_file(monkeypatch, tmp_path):
    import datasets
    import huggingface_hub

    downloaded = []
    loaded = []

    def download(**kwargs):
        downloaded.append(kwargs)
        return "/explicit-train.parquet"

    def load(*args, **kwargs):
        loaded.append((args, kwargs))
        return [{"question": f"What is {i} + 1?", "answer": f"#### {i + 1}"} for i in range(6)]

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(datasets, "load_dataset", load)
    prepare_gsm8k(tmp_path / "prepared", "a" * 40, 42, 2, 2)
    assert downloaded == [
        {
            "repo_id": "openai/gsm8k",
            "repo_type": "dataset",
            "revision": "a" * 40,
            "filename": "main/train-00000-of-00001.parquet",
        }
    ]
    assert loaded == [
        (("parquet",), {"data_files": {"train": "/explicit-train.parquet"}, "split": "train"})
    ]
    partition = json.loads((tmp_path / "prepared/partition.json").read_text())
    assert not partition["official_test_loaded"]
    assert set(partition["train_indices"]).isdisjoint(partition["development_indices"])


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
def test_jsonl_preserves_unicode_line_separators(tmp_path, separator):
    from forge_training.config import DatasetSpec
    from forge_training.data import write_dataset

    rows = [Example(id="unicode", question=f"First line{separator}second line?", answer="#### 2")]
    write_dataset(
        tmp_path,
        "train",
        rows,
        source="fixture",
        revision="v1",
        source_split="synthetic",
        license_name="CC0",
    )
    spec = DatasetSpec(
        records=str(tmp_path / "train.jsonl"), manifest=str(tmp_path / "train.manifest.json")
    )
    loaded, _ = load_dataset(spec, "train")
    assert loaded == rows
