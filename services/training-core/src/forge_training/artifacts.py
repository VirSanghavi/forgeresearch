"""Content identities and exclusive artifact writes; no implicit overwrite."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_manifest(path: Path) -> dict[str, str]:
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"Expected a real artifact directory: {path}")
    files = {}
    for entry in sorted(path.rglob("*")):
        if entry.is_symlink():
            raise ValueError(f"Symlinks are not allowed in artifact directories: {entry}")
        if entry.is_file():
            files[entry.relative_to(path).as_posix()] = file_sha256(entry)
    if not files:
        raise ValueError(f"Empty artifact directory: {path}")
    return files


def manifest_sha256(files: dict[str, str]) -> str:
    data = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def directory_sha256(path: Path) -> str:
    return manifest_sha256(directory_manifest(path))


def write_json(path: Path, data: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def runtime_versions() -> dict[str, str | None]:
    versions = {}
    for name in (
        "forge-training-core",
        "torch",
        "transformers",
        "peft",
        "tokenizers",
        "numpy",
        "accelerate",
        "datasets",
        "trl",
        "pydantic",
        "huggingface-hub",
        "safetensors",
    ):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    versions["forge_training_source_sha256"] = manifest_sha256(
        {path.name: file_sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    )
    return versions


class EventLog:
    """Single-writer worker log. The orchestrator must reconcile with persisted state."""

    def __init__(self, path: Path, campaign_id: str, experiment_id: str, producer: str = "worker"):
        self.path = path
        self.campaign_id = campaign_id
        self.experiment_id = experiment_id
        self.producer = producer
        self.sequence = 0

    def emit(self, kind: str, payload: dict) -> None:
        self.sequence += 1
        event = {
            "schemaVersion": 1,
            "eventId": f"{self.experiment_id}:{self.producer}:{self.sequence}",
            "campaignId": self.campaign_id,
            "experimentId": self.experiment_id,
            "sequence": self.sequence,
            "producer": self.producer,
            "timestamp": utc_now(),
            "type": kind,
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
