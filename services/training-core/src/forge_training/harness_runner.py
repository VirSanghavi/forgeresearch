"""Append-only harness artifacts, fresh-process paired runs, and evidence verification."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from .artifacts import (
    directory_manifest,
    directory_sha256,
    file_sha256,
    runtime_versions,
    write_json,
)
from .config import load_config
from .data import load_dataset, load_tasks, verify_separation
from .harness import harness_identity, task_path
from .harness_analysis import compare_harness, summarize_ablation
from .harness_config import AblationSpec, HarnessProtocol, ResourceLimits, ablation_protocols
from .harness_resources import output_bytes, preflight_resources, supervised


def seal_bundle(root: Path) -> dict:
    manifest = {"schema_version": 1, "files": directory_manifest(root)}
    write_json(root / "bundle.json", manifest)
    return {"path": str(root.resolve()), "bundle_sha256": file_sha256(root / "bundle.json")}


def verify_bundle(root: Path, expected: str | None = None) -> dict:
    if expected and file_sha256(root / "bundle.json") != expected:
        raise ValueError("Bundle identity differs from authorized hash")
    manifest = json.loads((root / "bundle.json").read_text())
    actual = directory_manifest(root)
    actual.pop("bundle.json")
    if manifest["files"] != actual:
        raise ValueError("Bundle files were changed, added, or removed")
    if (root / "comparison.json").exists():
        baseline = json.loads((root / "baseline.json").read_text())
        candidate = json.loads((root / "candidate.json").read_text())
        if compare_harness(baseline, candidate) != json.loads(
            (root / "comparison.json").read_text()
        ):
            raise ValueError("Comparison differs from sample evidence")
        for side, report in (("baseline", baseline), ("candidate", candidate)):
            if report.get("raw_sha256") != file_sha256(root / f"{side}.raw.json"):
                raise ValueError("Raw harness evidence checksum mismatch")
    if (root / "ablation.json").exists():
        if summarize_ablation(ablation_entries(root)) != json.loads(
            (root / "summary.json").read_text()
        ):
            raise ValueError("Ablation summary differs from variant evidence")
    return {
        "status": "verified",
        "files": len(actual),
        "bundle_sha256": file_sha256(root / "bundle.json"),
    }


def new_directory(root: Path, limits: ResourceLimits):
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite {root}")
    preflight_resources(root.parent, limits, count_output=False)
    root.mkdir(mode=0o700)


def copy_adapter(source: Path | None, root: Path, name: str, limits: ResourceLimits):
    if source is None:
        return None
    digest = directory_sha256(source)
    if output_bytes(root) + output_bytes(source) > limits.max_output_mib * 1024**2:
        raise ValueError("Adapter copy would exceed artifact size budget")
    shutil.copytree(source, root / name)
    if directory_sha256(root / name) != digest:
        raise ValueError("Adapter changed while copying")
    return {"path": name, "sha256": digest}


def write_rows(path: Path, rows: list[dict]):
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_plan(
    root: Path,
    *,
    role: str,
    model: dict,
    rows: list[dict],
    fewshot: list[dict],
    dataset: dict,
    baseline: Path | None,
    candidate: Path,
    protocol: HarnessProtocol,
    limits: ResourceLimits,
) -> dict:
    write_rows(root / "evaluation.jsonl", rows)
    write_rows(root / "fewshot.jsonl", fewshot)
    plan = {
        "schema_version": 1,
        "role": role,
        "model": model,
        "protocol": protocol.model_dump(),
        "limits": limits.model_dump(),
        "dataset": dataset,
        "runtime": {**runtime_versions(), "lm_eval": harness_identity()},
        "task_sha256": file_sha256(task_path()),
        "baseline": copy_adapter(baseline, root, "baseline-adapter", limits),
        "candidate": copy_adapter(candidate, root, "candidate-adapter", limits),
        "inputs": {n: file_sha256(root / n) for n in ("evaluation.jsonl", "fewshot.jsonl")},
    }
    if plan["baseline"] and plan["baseline"]["sha256"] == plan["candidate"]["sha256"]:
        raise ValueError("Candidate equals the starting policy")
    write_json(root / "plan.json", plan)
    return plan


def run_plan(root: Path, limits: ResourceLimits) -> dict:
    plan_hash = file_sha256(root / "plan.json")
    try:
        resources = {}
        for side in ("baseline", "candidate"):
            resources[side] = supervised(
                [
                    sys.executable,
                    "-m",
                    "forge_training.harness_cli",
                    "_evaluate",
                    "--plan",
                    str(root / "plan.json"),
                    "--side",
                    side,
                ],
                root,
                side,
                limits,
            )
        if file_sha256(root / "plan.json") != plan_hash:
            raise ValueError("Evaluation plan changed during execution")
        plan = json.loads((root / "plan.json").read_text())
        for name, digest in plan["inputs"].items():
            if file_sha256(root / name) != digest:
                raise ValueError("Evaluation data changed during execution")
        reports = [json.loads((root / f"{s}.json").read_text()) for s in ("baseline", "candidate")]
        comparison = compare_harness(*reports)
        write_json(root / "comparison.json", comparison)
        write_json(root / "resources.json", resources)
        write_json(
            root / "result.json",
            {
                "status": "completed",
                "role": plan["role"],
                "plan_sha256": plan_hash,
                "official_test_loaded": plan["role"] == "final",
                "final_improvement_supported": comparison["final_improvement_supported"],
            },
        )
        return {"status": "completed", **seal_bundle(root), "comparison": comparison}
    except BaseException as error:
        write_json(
            root / "failure.json",
            {"status": "failed", "error": str(error), "partial_results_are_not_scores": True},
        )
        raise


def development_run(
    config_path: Path,
    checkpoint: Path,
    expected_hash: str,
    output: Path,
    protocol: HarnessProtocol,
    limits: ResourceLimits,
) -> dict:
    config = load_config(config_path)
    if directory_sha256(checkpoint) != expected_hash:
        raise ValueError("Candidate checksum mismatch")
    rows, manifest = load_dataset(config.development, "development")
    train, _ = (
        load_tasks(config.train)
        if config.training.method == "grpo"
        else load_dataset(config.train, "train")
    )
    verify_separation(train, rows)
    if protocol.num_fewshot and (
        config.training.method == "grpo" or len(train) < protocol.num_fewshot
    ):
        raise ValueError("Few-shot evaluation requires enough answer-bearing training examples")
    start = config.grpo.init_adapter if config.grpo else None
    if start and directory_sha256(Path(start.path)) != start.sha256:
        raise ValueError("Starting adapter checksum mismatch")
    new_directory(output, limits)
    prepare_plan(
        output,
        role="development",
        model=config.model.model_dump(),
        rows=[r.model_dump() for r in rows],
        fewshot=[r.model_dump() for r in train] if protocol.num_fewshot else [],
        dataset=manifest.model_dump(),
        baseline=Path(start.path) if start else None,
        candidate=checkpoint,
        protocol=protocol,
        limits=limits,
    )
    return run_plan(output, limits)


def audit_legacy(run: Path, output: Path) -> dict:
    """Re-score preserved generations with official task filters, without loading a model."""
    from lm_eval.api.instance import Instance

    from .evaluation import compare
    from .harness import task_for
    from .worker import verify_run

    verify_run(run)
    baseline, candidate = [
        json.loads((run / f"{s}.json").read_text()) for s in ("baseline", "candidate")
    ]
    compare(
        baseline,
        candidate,
        baseline["protocol"]["seed"],
        shared_baseline=True,
        starting_checkpoint=baseline["checkpoint_sha256"],
    )
    new_directory(output, ResourceLimits())
    normalized = []
    for side, report in (("baseline", baseline), ("candidate", candidate)):
        rows = [
            {"id": s["id"], "question": s["question"], "answer": f"#### {s['reference_answer']}"}
            for s in report["samples"]
        ]
        task = task_for(rows, [], HarnessProtocol(num_fewshot=0))
        instances = []
        for i, (row, sample) in enumerate(zip(rows, report["samples"], strict=True)):
            inst = Instance(
                request_type="generate_until",
                doc=row,
                arguments=(row["question"], {}),
                idx=0,
                metadata=("gsm8k", i, 1),
            )
            inst.resps = [sample["prediction"]]
            instances.append(inst)
        task._instances = instances
        task.apply_filters()
        samples = []
        for inst, sample in zip(instances, report["samples"], strict=True):
            metrics = {
                k: bool(task.process_results(inst.doc, [v])["exact_match"])
                for k, v in inst.filtered_resps.items()
            }
            samples.append(
                {
                    "id": sample["id"],
                    "question": sample["question"],
                    "target": inst.doc["answer"],
                    "prompt_sha256": sample["prompt_sha256"],
                    "prediction": sample["prediction"],
                    "strict": metrics["strict-match"],
                    "flexible": metrics["flexible-extract"],
                    "hit_limit": sample["hit_generation_limit"],
                }
            )
        write_json(output / f"{side}.raw.json", report)
        record = {
            "role": "development",
            "model": report["model"],
            "dataset": report["dataset"],
            "protocol": {
                **report["protocol"],
                "analysis": "preserved generations rescored; NOT a harness generation run",
            },
            "runtime": {"lm_eval": harness_identity()},
            "task_sha256": file_sha256(task_path()),
            "checkpoint_sha256": report["checkpoint_sha256"],
            "samples": samples,
            "raw_sha256": file_sha256(output / f"{side}.raw.json"),
        }
        write_json(output / f"{side}.json", record)
        normalized.append(record)
    comparison = compare_harness(*normalized)
    write_json(output / "comparison.json", comparison)
    write_json(
        output / "result.json",
        {
            "status": "completed",
            "role": "development",
            "model_loaded": False,
            "official_test_loaded": False,
        },
    )
    return {**seal_bundle(output), "comparison": comparison}


def ablation_entries(root: Path) -> list[dict]:
    plan = json.loads((root / "ablation.json").read_text())
    entries = []
    for variant in plan["variants"]:
        directory = root / f"variant-{variant['name']}"
        verify_bundle(directory)
        baseline = json.loads((directory / "baseline.json").read_text())
        if baseline["protocol"] != variant["protocol"]:
            raise ValueError(f"Variant {variant['name']} ran a different protocol")
        entries.append(
            {
                "name": variant["name"],
                "protocol": variant["protocol"],
                "comparison": json.loads((directory / "comparison.json").read_text()),
            }
        )
    return entries


def ablation_run(
    config_path: Path,
    checkpoint: Path,
    expected_hash: str,
    output: Path,
    base: HarnessProtocol,
    spec: AblationSpec,
    limits: ResourceLimits,
) -> dict:
    """Run predeclared prompt/length variants of one development pair, then summarize.

    The variant list is written before any generation so it cannot be chosen after
    seeing results. Each variant is an ordinary sealed development bundle.
    """
    protocols = ablation_protocols(base, spec)
    new_directory(output, limits)
    write_json(
        output / "ablation.json",
        {
            "schema_version": 1,
            "base_protocol": base.model_dump(),
            "candidate_sha256": expected_hash,
            "variants": [{"name": n, "protocol": p.model_dump()} for n, p in protocols],
        },
    )
    declared = file_sha256(output / "ablation.json")
    try:
        for name, protocol in protocols:
            development_run(
                config_path,
                checkpoint,
                expected_hash,
                output / f"variant-{name}",
                protocol,
                limits,
            )
        if file_sha256(output / "ablation.json") != declared:
            raise ValueError("Ablation plan changed during execution")
        summary = summarize_ablation(ablation_entries(output))
        write_json(output / "summary.json", summary)
        return {"status": "completed", **seal_bundle(output), "summary": summary}
    except BaseException as error:
        write_json(
            output / "failure.json",
            {"status": "failed", "error": str(error), "partial_results_are_not_scores": True},
        )
        raise
