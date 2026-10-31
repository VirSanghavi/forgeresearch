"""Freeze only a development-selected, governance-approved, promoted checkpoint."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .artifacts import directory_sha256, file_sha256, runtime_versions, utc_now, write_json
from .config import DatasetSpec, load_config
from .data import load_dataset, verify_separation
from .harness import harness_identity, task_path
from .harness_config import GSM8K_REVISION, HarnessProtocol, ResourceLimits
from .harness_runner import copy_adapter, new_directory, seal_bundle, verify_bundle, write_rows
from .worker import verify_run


def freeze_release(
    campaign: Path,
    output: Path,
    protocol: HarnessProtocol,
    limits: ResourceLimits,
    harness_evidence: Path,
):
    report_path = campaign / "research-report.json"
    report = json.loads(report_path.read_text())
    promoted = report.get("bestCheckpointPromoted")
    if report.get("campaignStatus") != "stopped" or not promoted:
        raise ValueError("Freeze requires a stopped campaign with a promoted checkpoint")
    matches = [r for r in report["governance"] if r["experimentId"] == promoted["experimentId"]]
    if (
        len(matches) != 1
        or matches[0]["decision"] != "approved"
        or matches[0]["promotion"] != "PROMOTED"
    ):
        raise ValueError("Promoted checkpoint lacks unique governance approval")
    review = matches[0]
    # IDs are file names, never paths supplied by a report.
    for name in (review["label"], review["experimentId"]):
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Invalid campaign artifact identifier")
    experiment = campaign / "experiments" / review["label"]
    run = experiment / "run"
    verified = verify_run(run)
    config = load_config(run / "resolved-config.json")
    if (
        config.experiment_id != promoted["experimentId"]
        or config.campaign_id != report["campaignId"]
    ):
        raise ValueError("Promotion identifies a different run")
    digest = directory_sha256(run / "checkpoint")
    if digest != promoted["checkpointSha256"]:
        raise ValueError("Promoted checkpoint bytes changed")
    verify_bundle(harness_evidence)
    harness_candidate = json.loads((harness_evidence / "candidate.json").read_text())
    harness_baseline = json.loads((harness_evidence / "baseline.json").read_text())
    harness_comparison = json.loads((harness_evidence / "comparison.json").read_text())
    initial = config.grpo.init_adapter if config.grpo else None
    if (
        harness_candidate["role"] != "development"
        or harness_candidate["checkpoint_sha256"] != digest
        or harness_baseline["checkpoint_sha256"] != (initial.sha256 if initial else None)
        or harness_candidate["model"] != config.model.model_dump()
        or harness_candidate["protocol"] != protocol.model_dump()
    ):
        raise ValueError(
            "Release needs a paired development harness run with the same model, "
            "adapters, and protocol"
        )
    if (
        config.model.model_status != "fixture"
        and not harness_comparison["development_release_gate"]["passed"]
    ):
        raise ValueError(
            "Development harness release gate failed: strict gain and no flexible "
            "accuracy drop required"
        )
    evidence_path = campaign / "governance" / f"{review['experimentId']}.json"
    if file_sha256(evidence_path) != review["evidenceSha256"]:
        raise ValueError("Governance evidence checksum mismatch")
    evidence = json.loads(evidence_path.read_text())
    required = {
        "checksumVerified",
        "dataAllowed",
        "evaluationReproducible",
        "objectiveAligned",
        "regressionsPassed",
    }
    if (
        evidence["experimentId"] != config.experiment_id
        or evidence["decision"] != "approved"
        or evidence["reasonCodes"]
        or set(evidence["checks"]) != required
        or any(v is not True for v in evidence["checks"].values())
        or any(v != digest for v in evidence["facts"]["checkpoint"].values())
    ):
        raise ValueError("Governance evidence does not approve these checkpoint bytes")
    evaluation_hash = evidence["facts"]["evaluationReport"]["recomputed"]
    evaluations = [
        p for p in (experiment / "evaluator").glob("*.json") if file_sha256(p) == evaluation_hash
    ]
    if len(evaluations) != 1:
        raise ValueError("Reviewed evaluation report is missing or changed")
    reviewed = json.loads(evaluations[0].read_text())
    if reviewed["evaluation_role"] != "development" or reviewed["checkpoint_sha256"] != digest:
        raise ValueError("Selection must use a development evaluation of this checkpoint")
    regression_path = campaign / "governance" / f"{review['label']}-regression.json"
    regression_baseline = campaign / "governance/regression-baseline.json"
    regression = evidence["facts"]["regressionSuite"]
    if file_sha256(regression_path) != regression["candidateReportSha256"]:
        raise ValueError("Regression evidence changed")
    from .evaluation import compare

    regression_before = json.loads(regression_baseline.read_text())
    regression_after = json.loads(regression_path.read_text())
    comparison = compare(
        regression_before,
        regression_after,
        config.seed,
        shared_baseline=True,
        starting_checkpoint=regression_before["checkpoint_sha256"],
    )
    if (
        regression_after["checkpoint_sha256"] != digest
        or comparison["candidate_accuracy"] < comparison["baseline_accuracy"]
        or regression_before["num_correct"] != regression["baselineCorrect"]
        or regression_after["num_correct"] != regression["candidateCorrect"]
    ):
        raise ValueError("Held-out regression evidence does not support release")
    development, dev_manifest = load_dataset(
        DatasetSpec(
            records=str(run / "inputs/development.jsonl"),
            manifest=str(run / "inputs/development.manifest.json"),
        ),
        "development",
    )
    # Reuse the campaign's original answer-bearing allowed training partition for few-shot
    # contexts, including when the selected run used answer-free GRPO task projections.
    source = campaign / "data/gsm8k"
    if not source.exists():
        source = run / "inputs"
    train, train_manifest = load_dataset(
        DatasetSpec(
            records=str(source / "train.jsonl"), manifest=str(source / "train.manifest.json")
        ),
        "train",
    )
    verify_separation(train, development)
    from .data import question_key
    from .harness import digest_json

    if harness_candidate["dataset"]["content_sha256"] != dev_manifest.content_sha256:
        raise ValueError("Harness used a different development partition")
    if config.model.model_status != "fixture" and any(
        m.source != "openai/gsm8k"
        or m.source_revision != GSM8K_REVISION
        or m.source_split != "train"
        for m in (train_manifest, dev_manifest)
    ):
        raise ValueError("Release data must come from the pinned official training partition")
    if protocol.num_fewshot and len(train) < protocol.num_fewshot:
        raise ValueError("Not enough approved few-shot examples")
    new_directory(output, limits)
    adapter = copy_adapter(run / "checkpoint", output, "checkpoint", limits)
    start = config.grpo.init_adapter if config.grpo else None
    baseline = copy_adapter(
        run / "inputs/parent-adapter" if start else None, output, "baseline-adapter", limits
    )
    if start and baseline["sha256"] != start.sha256:
        raise ValueError("Starting policy differs from the reviewed baseline")
    for source_path, name in (
        (report_path, "campaign.json"),
        (evidence_path, "governance.json"),
        (evaluations[0], "development-evaluation.json"),
        (run / "manifest.json", "training-manifest.json"),
        (regression_path, "regression-candidate.json"),
        (regression_baseline, "regression-baseline.json"),
    ):
        shutil.copyfile(source_path, output / name)
    write_rows(output / "development.jsonl", [r.model_dump() for r in development])
    write_rows(
        output / "fewshot.jsonl", [r.model_dump() for r in train] if protocol.num_fewshot else []
    )
    write_json(
        output / "research-question-hashes.json",
        sorted(
            {
                digest_json(question_key(q))
                for q in (
                    [r.question for r in train + development]
                    + [s["question"] for s in regression_after["samples"]]
                )
            }
        ),
    )
    for name in ("baseline.json", "candidate.json", "comparison.json", "bundle.json"):
        shutil.copyfile(harness_evidence / name, output / f"harness-{name}")
    write_json(
        output / "release.json",
        {
            "schema_version": 1,
            "frozen_at": utc_now(),
            "campaign_id": config.campaign_id,
            "experiment_id": config.experiment_id,
            "model": config.model.model_dump(),
            "candidate": adapter,
            "baseline": baseline,
            "protocol": protocol.model_dump(),
            "limits": limits.model_dump(),
            "runtime": {**runtime_versions(), "lm_eval": harness_identity()},
            "task_sha256": file_sha256(task_path()),
            "training_verification": verified,
            "final_dataset": {
                "source": "openai/gsm8k",
                "revision": GSM8K_REVISION,
                "file": "main/test-00000-of-00001.parquet",
                "num_examples": 1319,
            },
            "official_test_loaded": False,
            "final_evaluation_completed": False,
            "boundary": (
                "Use a separately owned evaluator account/container for adversarial "
                "isolation; local hashes are not independent attestation."
            ),
        },
    )
    (output / "REPRODUCE.md").write_text(
        "# Frozen Forge release\n\n"
        "Install the matching Forge source with `pip install -e '.[train,harness]'`. "
        "Use `forge-harness verify --bundle <release> --expected-hash <trusted hash>`. "
        "The baseline model is referenced by pinned revision and is not duplicated here. "
        "The adapter is in `checkpoint/`; PEFT loads it on the recorded base model.\n\n"
        "A separately owned evaluator provisions the pinned model and final dataset into "
        "its Hugging Face cache, initializes an authority, authorizes this bundle hash, "
        "and runs `forge-final evaluate`. Evaluation is offline and refuses cache misses. "
        "One attempt consumes the campaign grant even if it crashes. Rehearsal uses only "
        "development data: `forge-final rehearse --release <release> --output <new-dir>`.\n\n"
        "Event-rule acceptance, physical evaluator isolation, clean-machine reproduction, "
        "and a faithful two-minute recording remain external readiness gates.\n"
    )
    return {"status": "frozen", "official_test_loaded": False, **seal_bundle(output)}
