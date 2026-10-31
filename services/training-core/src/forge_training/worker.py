"""Bounded subprocess orchestration; no campaign budgets or promotion permissions."""

from __future__ import annotations

import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from .artifacts import (
    EventLog,
    directory_manifest,
    directory_sha256,
    file_sha256,
    manifest_sha256,
    runtime_versions,
    utc_now,
    write_json,
)
from .config import ExperimentConfig, load_config, starting_checkpoint
from .data import load_dataset, load_tasks, verify_separation
from .evaluation import compare, evaluation_protocol
from .reports import EvaluationReport
from .scoring import extract_answer

# Credentials for the orchestrator's world gateway; only the GRPO training phase gets them.
GATEWAY_ENVIRONMENT = ("FORGE_WORLD_GATEWAY_URL", "FORGE_WORLD_GATEWAY_TOKEN")


def validate_inputs(config: ExperimentConfig, *, check_output: bool = True) -> dict:
    # Reward-driven training reads answer-free tasks; supervised training reads examples.
    grpo = config.training.method == "grpo"
    train, train_manifest = (
        load_tasks(config.train) if grpo else load_dataset(config.train, "train")
    )
    development, dev_manifest = load_dataset(config.development, "development")
    verify_separation(train, development)
    referenced = development if grpo else train + development
    if any(extract_answer(row.answer) is None for row in referenced):
        raise ValueError("All references must end with a numeric '#### <answer>' line")
    output = Path(config.output_dir)
    inputs = [
        Path(spec_path)
        for spec in (config.train, config.development)
        for spec_path in (spec.records, spec.manifest)
    ]
    if config.model.source == "local":
        inputs.append(Path(config.model.model_id))
        if output.is_relative_to(Path(config.model.model_id)):
            raise ValueError("Output may not be written inside the local base-model directory")
    adapter = config.grpo.init_adapter if config.grpo is not None else None
    if adapter is not None:
        # The starting adapter is an input like the data: checked by hash, never written to.
        if directory_sha256(Path(adapter.path)) != adapter.sha256:
            raise ValueError("Starting adapter checksum mismatch")
        inputs.append(Path(adapter.path))
        if output.is_relative_to(Path(adapter.path)):
            raise ValueError("Output may not be written inside the starting adapter")
    if check_output and any(input_path.is_relative_to(output) for input_path in inputs):
        raise ValueError("Output directory may not contain input artifacts")
    return {"train": train_manifest.model_dump(), "development": dev_manifest.model_dump()}


def run_phase(arguments: list[str], timeout: int, log: Path, *, world_access: bool = False) -> None:
    """A parent deadline terminates a whole process group, including model loading."""
    environment = {
        name: value
        for name, value in os.environ.items()
        if world_access or name not in GATEWAY_ENVIRONMENT
    }
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(
            [sys.executable, "-m", "forge_training", *arguments],
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=environment,
        )
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            # Also cancel the child on Ctrl-C or termination of the calling operation.
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if code:
        raise RuntimeError(f"Worker subprocess failed (exit {code}); inspect {log}")


def adopt_frozen_baseline(
    config: ExperimentConfig, source: Path, expected_sha256: str | None, destination: Path
) -> None:
    """Reuse the campaign's untrained evaluation only when it matches this run exactly."""
    if expected_sha256 is None or file_sha256(source) != expected_sha256:
        raise ValueError("Frozen baseline checksum mismatch or missing expected checksum")
    report = EvaluationReport.model_validate_json(source.read_text()).model_dump()
    rows, manifest = load_dataset(config.development, "development")
    if (
        report["checkpoint_sha256"] != starting_checkpoint(config)
        or report["campaign_id"] != config.campaign_id
        or report["model"] != config.model.model_dump()
        or report["dataset"] != manifest.model_dump()
        or report["protocol"] != evaluation_protocol(config)
    ):
        raise ValueError("Frozen baseline differs from this run's model, data or protocol")
    for row, sample in zip(rows, report["samples"], strict=True):
        if (row.id, row.question, str(extract_answer(row.answer))) != (
            sample["id"],
            sample["question"],
            sample["reference_answer"],
        ):
            raise ValueError("Frozen baseline samples differ from the staged development data")
    shutil.copyfile(source, destination)


def run_experiment(
    config_path: Path,
    frozen_baseline: Path | None = None,
    frozen_baseline_sha256: str | None = None,
) -> dict:
    config = load_config(config_path)
    datasets = validate_inputs(config)
    output = Path(config.output_dir)
    # Creation is the claim on a run directory; another worker cannot reuse it.
    output.mkdir(parents=True, exist_ok=False)
    events = EventLog(output / "events.jsonl", config.campaign_id, config.experiment_id)
    started = time.monotonic()
    try:
        runtime = runtime_versions()
        write_json(
            output / "environment.json",
            {
                "python": sys.version,
                "platform": platform.platform(),
                "runtime": runtime,
            },
        )
        # Freeze exact input bytes before separate processes read them. Source files
        # may be owned by another agent and must not change the effective experiment.
        staged = output / "inputs"
        staged.mkdir()
        effective = config.model_dump()
        replay = config.model_dump()
        for purpose in ("train", "development"):
            source = getattr(config, purpose)
            for key, suffix in (("records", "jsonl"), ("manifest", "manifest.json")):
                destination = staged / f"{purpose}.{suffix}"
                shutil.copyfile(getattr(source, key), destination)
                effective[purpose][key] = str(destination)
                replay[purpose][key] = f"inputs/{purpose}.{suffix}"
        start = starting_checkpoint(config)
        if start is not None:
            shutil.copytree(config.grpo.init_adapter.path, staged / "parent-adapter")
            effective["grpo"]["init_adapter"]["path"] = str(staged / "parent-adapter")
            replay["grpo"]["init_adapter"]["path"] = "inputs/parent-adapter"
        config = ExperimentConfig.model_validate(effective)
        validate_inputs(config, check_output=False)
        replay["output_dir"] = "."
        write_json(output / "replay-config.json", replay)
        write_json(output / "resolved-config.json", config.model_dump())
        events.emit("EXPERIMENT_STARTED", {"runtime": runtime, "datasets": datasets})
        frozen = str(output / "resolved-config.json")
        roles: tuple[str, ...] = ("baseline", "training", "candidate")
        baseline_source: dict = {"kind": "in_run"}
        if frozen_baseline is not None:
            # The protocol includes this package's source hash and runtime versions, so a
            # reused report was produced by identical code on the identical inputs.
            adopt_frozen_baseline(
                config, frozen_baseline, frozen_baseline_sha256, output / "baseline.json"
            )
            baseline_source = {"kind": "frozen", "sha256": frozen_baseline_sha256}
            events.emit(
                "EVALUATION_REUSED",
                {
                    "role": "baseline",
                    "evaluationRole": "development",
                    "reportSha256": frozen_baseline_sha256,
                },
            )
            roles = ("training", "candidate")
        for role in roles:
            if runtime_versions() != runtime:
                raise ValueError("Worker source or runtime changed between experiment phases")
            if role == "training":
                run_phase(
                    ["_train", "--config", frozen],
                    config.training.max_seconds,
                    output / "training.log",
                    world_access=config.training.method == "grpo",
                )
            else:
                events.emit("EVALUATION_STARTED", {"role": role, "evaluationRole": "development"})
                args = ["evaluate", "--config", frozen, "--output", str(output / f"{role}.json")]
                if role == "baseline" and start is not None:
                    # The baseline is the policy training starts from.
                    args += [
                        "--checkpoint",
                        config.grpo.init_adapter.path,
                        "--expected-hash",
                        start,
                    ]
                if role == "candidate":
                    training = json.loads((output / "training.json").read_text())
                    args += [
                        "--checkpoint",
                        str(output / "checkpoint"),
                        "--expected-hash",
                        training["checkpoint_sha256"],
                    ]
                run_phase(args, config.evaluation.max_seconds, output / f"{role}.log")
                report = json.loads((output / f"{role}.json").read_text())
                events.emit(
                    "EVALUATION_COMPLETED",
                    {
                        "role": role,
                        "accuracy": report["accuracy"],
                        "numExamples": report["num_examples"],
                        "evaluationRole": "development",
                        "reportSha256": file_sha256(output / f"{role}.json"),
                    },
                )
            if runtime_versions() != runtime:
                raise ValueError("Worker source or runtime changed during an experiment phase")
        baseline = json.loads((output / "baseline.json").read_text())
        candidate = json.loads((output / "candidate.json").read_text())
        comparison = compare(
            baseline,
            candidate,
            config.seed,
            shared_baseline=baseline_source["kind"] == "frozen",
            starting_checkpoint=start,
        )
        write_json(output / "comparison.json", comparison)
        summary = {
            "schema_version": 1,
            "campaign_id": config.campaign_id,
            "experiment_id": config.experiment_id,
            "status": "completed",
            "completed_at": utc_now(),
            "elapsed_seconds": time.monotonic() - started,
            "comparison": comparison,
            "checkpoint_sha256": training["checkpoint_sha256"],
            "weights_changed": training["weights_changed"],
            "training_method": config.training.method,
            "starting_checkpoint_sha256": start,
            "candidate_evaluated_in_fresh_process": True,
            "baseline_source": baseline_source,
            "cost_usd": None,
            "cost_status": "not_measured_by_local_worker",
            "governance_status": "unreviewed",
            "promotion_authorized": False,
        }
        write_json(output / "result.json", summary)
        events.emit(
            "EXPERIMENT_COMPLETED",
            {
                "resultSha256": file_sha256(output / "result.json"),
                "improvementObserved": comparison["improvement_observed"],
                "promotionAuthorized": False,
            },
        )
        artifacts = {
            path.relative_to(output).as_posix(): file_sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        }
        write_json(
            output / "manifest.json",
            {
                "schema_version": 1,
                "campaign_id": config.campaign_id,
                "experiment_id": config.experiment_id,
                "created_at": utc_now(),
                "files": artifacts,
            },
        )
        return summary
    except BaseException as error:
        events.emit(
            "EXPERIMENT_FAILED",
            {
                "errorType": type(error).__name__,
                "message": str(error),
                "elapsedSeconds": time.monotonic() - started,
            },
        )
        write_json(
            output / "failure.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "message": str(error),
                "completed_at": utc_now(),
            },
        )
        raise


def verify_run(output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_text())
    expected_files = manifest["files"]
    actual_files = {
        path.relative_to(output).as_posix(): path
        for path in output.rglob("*")
        if path.is_file() and path != output / "manifest.json"
    }
    if expected_files.keys() != actual_files.keys():
        raise ValueError("Artifact files were added or removed after the manifest was created")
    for relative, digest in expected_files.items():
        path = output / relative
        if not path.resolve().is_relative_to(output.resolve()) or path.is_symlink():
            raise ValueError("Artifact path is outside the run directory")
        if file_sha256(path) != digest:
            raise ValueError(f"Artifact checksum mismatch: {relative}")
    config = load_config(output / "resolved-config.json")
    baseline = json.loads((output / "baseline.json").read_text())
    candidate = json.loads((output / "candidate.json").read_text())
    result = json.loads((output / "result.json").read_text())
    source = result.get("baseline_source", {"kind": "in_run"})
    shared = source["kind"] == "frozen"
    if shared and file_sha256(output / "baseline.json") != source.get("sha256"):
        raise ValueError("Reused baseline differs from the frozen report the run recorded")
    start = starting_checkpoint(config)
    if start is not None and directory_sha256(output / "inputs/parent-adapter") != start:
        raise ValueError("Staged starting adapter differs from the configured one")
    recomputed = compare(
        baseline, candidate, config.seed, shared_baseline=shared, starting_checkpoint=start
    )
    if recomputed != json.loads((output / "comparison.json").read_text()):
        raise ValueError("Stored comparison does not match sample-level evidence")
    training = json.loads((output / "training.json").read_text())
    checkpoint_files = directory_manifest(output / "checkpoint")
    checkpoint_hash = manifest_sha256(checkpoint_files)
    if training["checkpoint_files"] != checkpoint_files:
        raise ValueError("Training metadata does not match saved checkpoint files")
    if any(
        checkpoint_hash != record["checkpoint_sha256"] for record in (training, candidate, result)
    ):
        raise ValueError("Checkpoint identities differ across training, evaluation, and result")
    if training["weights_changed"] is not True or (
        training["initial_trainable_sha256"] == training["final_trainable_sha256"]
    ):
        raise ValueError("Training metadata does not demonstrate changed weights")
    if training.get("method", "sft_lora") != config.training.method:
        raise ValueError("Training metadata method differs from the configuration")
    grpo = None
    if config.training.method == "grpo":
        from .grpo import verify_rollouts

        grpo = verify_rollouts(output, config, training)
    if result["comparison"] != recomputed or result["status"] != "completed":
        raise ValueError("Result does not match completed evaluation evidence")
    if result["promotion_authorized"] is not False or result["governance_status"] != "unreviewed":
        raise ValueError("A training worker cannot grant governance approval")
    for record in (
        (manifest, result, candidate) if shared else (manifest, result, baseline, candidate)
    ):
        if (record["campaign_id"], record["experiment_id"]) != (
            config.campaign_id,
            config.experiment_id,
        ):
            raise ValueError(
                "Artifact campaign or experiment identities do not match configuration"
            )
    if baseline["campaign_id"] != config.campaign_id:
        raise ValueError("Baseline belongs to another campaign")
    if baseline["model"] != config.model.model_dump():
        raise ValueError("Evaluation model differs from the frozen configuration")
    if (output / "replay-config.json").exists():
        replay = load_config(output / "replay-config.json")
        rows, dataset = load_dataset(replay.development, "development")
        # Phase 6 added record_kind with a backwards-compatible default. Normalize
        # the old manifest in memory; never rewrite the preserved evidence bytes.
        from .config import DatasetManifest

        if dataset.model_dump() != DatasetManifest.model_validate(baseline["dataset"]).model_dump():
            raise ValueError("Evaluated dataset differs from input snapshot")
        for row, sample in zip(rows, baseline["samples"], strict=True):
            if (row.id, row.question, str(extract_answer(row.answer))) != (
                sample["id"],
                sample["question"],
                sample["reference_answer"],
            ):
                raise ValueError("Evaluation samples differ from preserved input snapshot")
    return {
        "status": "verified",
        "num_artifacts": len(expected_files),
        "manifest_sha256": file_sha256(output / "manifest.json"),
        "rollouts": grpo,
        "note": "Hashes detect changes relative to this manifest; they are not a signature.",
    }
