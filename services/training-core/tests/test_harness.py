from __future__ import annotations

import copy
import json
import os
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytest.importorskip("lm_eval")

from forge_training.artifacts import directory_sha256, file_sha256, write_json
from forge_training.config import load_config
from forge_training.final_cli import authorize, consume, init_authority
from forge_training.fixtures import make_smoke_fixture
from forge_training.harness_analysis import compare_harness, paired_metric
from forge_training.harness_config import HarnessProtocol, ResourceLimits
from forge_training.harness_resources import GIB, supervised, violation
from forge_training.harness_runner import (
    audit_legacy,
    development_run,
    seal_bundle,
    verify_bundle,
)
from forge_training.worker import run_experiment, verify_run


def paired_reports(n=16):
    common = {
        "protocol": {"seed": 42},
        "model": {"id": "fixture"},
        "dataset": {"sha": "a"},
        "runtime": {},
        "task_sha256": "a" * 64,
        "role": "development",
        "checkpoint_sha256": None,
        "samples": [
            {
                "id": str(i),
                "question": str(i),
                "target": "1",
                "prompt_sha256": str(i),
                "strict": False,
                "flexible": True,
                "hit_limit": False,
            }
            for i in range(n)
        ],
    }
    after = copy.deepcopy(common)
    after["checkpoint_sha256"] = "b" * 64
    for row in after["samples"]:
        row["strict"] = True
    return common, after


def test_format_learning_does_not_become_a_final_claim():
    before, after = paired_reports()
    report = compare_harness(before, after)
    assert report["metrics"]["strict"]["wins"] == 16
    assert report["metrics"]["flexible"]["delta_percentage_points"] == 0
    assert len(report["strict_gains_already_flexible_correct"]) == 16
    assert not report["final_improvement_supported"]


def test_secondary_regression_blocks_release_and_final_claim():
    before, after = paired_reports(1319)
    before["role"] = after["role"] = "final"
    after["samples"][0]["flexible"] = False
    report = compare_harness(before, after)
    assert not report["development_release_gate"]["passed"]
    assert not report["final_improvement_supported"]
    assert report["recommendations"] == []  # No research feedback from sealed results.


@pytest.mark.parametrize(
    "field", ["runtime", "dataset", "protocol", "task_sha256", "model", "role"]
)
def test_mismatched_protocols_refused(field):
    before, after = paired_reports()
    after[field] = "changed"
    with pytest.raises(ValueError, match=field):
        compare_harness(before, after)


def test_reordered_missing_duplicate_or_nonboolean_samples_refused():
    before, after = paired_reports()
    after["samples"].reverse()
    with pytest.raises(ValueError, match="Unpaired"):
        compare_harness(before, after)
    before, after = paired_reports()
    after["samples"].pop()
    with pytest.raises(ValueError, match="counts"):
        compare_harness(before, after)
    before, after = paired_reports()
    before["samples"][1]["id"] = before["samples"][0]["id"]
    with pytest.raises(ValueError, match="Duplicate"):
        compare_harness(before, after)
    before, after = paired_reports()
    after["samples"][0]["strict"] = 1
    with pytest.raises(ValueError, match="booleans"):
        compare_harness(before, after)


def test_zero_discordance_is_not_significant_and_tiny_sample_has_uncertainty():
    result = paired_metric([False] * 2, [False] * 2, 42)
    assert result["mcnemar_exact_one_sided_p"] == 1
    assert result["baseline_wilson_95"][1] > 0.6
    assert paired_metric([False] * 5, [True] * 5, 42)["mcnemar_exact_one_sided_p"] == 1 / 32


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("rss_bytes", 6 * GIB, "process memory"),
        ("available_bytes", GIB, "available memory"),
        ("swap_bytes", GIB, "swap growth"),
        ("disk_free_bytes", GIB, "disk reserve"),
        ("output_bytes", 300 * 1024**2, "output size"),
    ],
)
def test_resource_caps(field, value, reason):
    sample = {
        "rss_bytes": 0,
        "available_bytes": 4 * GIB,
        "swap_bytes": 0,
        "disk_free_bytes": 20 * GIB,
        "output_bytes": 0,
    }
    sample[field] = value
    assert reason in violation(sample, 0, 0, ResourceLimits())


def test_supervisor_stops_own_job_and_preserves_partial_files(tmp_path):
    limits = ResourceLimits(max_seconds_per_model=1, min_available_gib=0.25, min_disk_free_gib=1.0)
    marker = tmp_path / "preserved.txt"
    marker.write_text("existing evidence")
    with pytest.raises(RuntimeError, match="time limit"):
        supervised(
            [sys.executable, "-c", "import time; time.sleep(20)"], tmp_path, "deadline", limits
        )
    assert marker.read_text() == "existing evidence"
    assert (tmp_path / "deadline.resources.jsonl").exists()


def authority_fixture(tmp_path):
    authority, release = tmp_path / "authority", tmp_path / "release"
    init_authority(authority)
    release.mkdir()
    write_json(
        release / "release.json",
        {"campaign_id": "campaign-one", "model": {"model_status": "instruct"}},
    )
    seal_bundle(release)
    issued = authorize(authority, release)
    return authority, release, json.loads(Path(issued["grant"]).read_text())


def test_one_attempt_across_concurrent_evaluators_and_restart(tmp_path):
    authority, release, grant = authority_fixture(tmp_path)

    def attempt(_):
        try:
            consume(authority, grant, release)
            return "consumed"
        except ValueError:
            return "refused"

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(attempt, range(4)))
    assert results.count("consumed") == 1
    with pytest.raises(ValueError, match="already consumed"):
        consume(authority, grant, release)
    with pytest.raises(sqlite3.IntegrityError):
        authorize(authority, release)


def test_forged_or_mutated_release_does_not_consume_grant(tmp_path):
    authority, release, grant = authority_fixture(tmp_path)
    with pytest.raises(ValueError, match="signature"):
        consume(authority, {**grant, "signature": "0" * 64}, release)
    (release / "extra.txt").write_text("tamper")
    with pytest.raises(ValueError, match="changed, added"):
        consume(authority, grant, release)
    with sqlite3.connect(authority / "registry.sqlite") as conn:
        assert conn.execute("SELECT status FROM grants").fetchone()[0] == "authorized"


def test_public_authority_and_fixture_authorization_refused(tmp_path):
    authority, release, _ = authority_fixture(tmp_path)
    os.chmod(authority, 0o755)
    with pytest.raises(ValueError, match="private"):
        authorize(authority, release)
    fixture = tmp_path / "fixture-release"
    fixture.mkdir()
    write_json(
        fixture / "release.json", {"campaign_id": "fixture", "model": {"model_status": "fixture"}}
    )
    seal_bundle(fixture)
    os.chmod(authority, 0o700)
    with pytest.raises(ValueError, match="Fixture"):
        authorize(authority, fixture)


@pytest.fixture(scope="module")
def harness_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("harness")
    config_path = make_smoke_fixture(root / "fixture")
    result = run_experiment(config_path)
    protocol = HarnessProtocol(num_fewshot=0, max_new_tokens=8, max_prompt_tokens=128)
    limits = ResourceLimits(max_seconds_per_model=90, min_available_gib=0.25, min_disk_free_gib=1.0)
    output = root / "harness"
    # Test holds the same real host-wide exclusion as the CLI.
    from forge_training.locking import HELD, acquire_model_lock

    lock = acquire_model_lock("harness-test")
    try:
        evidence = development_run(
            config_path,
            root / "fixture/run/checkpoint",
            result["checkpoint_sha256"],
            output,
            protocol,
            limits,
        )
    finally:
        if lock:
            lock.close()
        os.environ.pop(HELD, None)
    return root, config_path, output, evidence


@pytest.mark.integration
def test_real_harness_generation_scores_and_adapter_reload(harness_run):
    root, config_path, output, evidence = harness_run
    assert evidence["status"] == "completed"
    assert verify_bundle(output)["status"] == "verified"
    baseline, candidate = [
        json.loads((output / f"{s}.json").read_text()) for s in ("baseline", "candidate")
    ]
    assert len(baseline["samples"]) == len(candidate["samples"]) == 2
    assert baseline["checkpoint_sha256"] is None
    assert candidate["checkpoint_sha256"] == directory_sha256(root / "fixture/run/checkpoint")
    assert baseline["protocol"]["batch_size"] == 1
    assert not evidence["comparison"]["final_improvement_supported"]
    assert not json.loads((output / "result.json").read_text())["official_test_loaded"]
    assert load_config(config_path).model.model_status == "fixture"


@pytest.mark.integration
def test_no_overwrite_and_tampering_detected(harness_run, tmp_path):
    import shutil

    _, config_path, output, _ = harness_run
    before = file_sha256(output / "bundle.json")
    with pytest.raises(FileExistsError):
        development_run(
            config_path,
            output / "candidate-adapter",
            directory_sha256(output / "candidate-adapter"),
            output,
            HarnessProtocol(num_fewshot=0),
            ResourceLimits(),
        )
    assert file_sha256(output / "bundle.json") == before
    shutil.copytree(output, tmp_path / "copy")
    (tmp_path / "copy/candidate-adapter/adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        verify_bundle(tmp_path / "copy")


@pytest.mark.integration
def test_legacy_audit_does_not_modify_old_evidence(harness_run, tmp_path):
    root, _, _, _ = harness_run
    run = root / "fixture/run"
    before = directory_sha256(run)
    result = audit_legacy(run, tmp_path / "audit")
    assert result["comparison"]["role"] == "development"
    assert directory_sha256(run) == before
    assert verify_bundle(tmp_path / "audit")["status"] == "verified"
    assert verify_run(run)["status"] == "verified"


def make_campaign(harness_run, tmp_path):
    """Labeled fixture governance evidence; never a production approval."""
    import shutil

    root, config_path, _, _ = harness_run
    config = load_config(config_path)
    campaign = tmp_path / "campaign"
    experiment = campaign / "experiments/selected"
    experiment.mkdir(parents=True)
    shutil.copytree(root / "fixture/run", experiment / "run")
    (experiment / "evaluator").mkdir()
    shutil.copyfile(experiment / "run/candidate.json", experiment / "evaluator/report.json")
    governance = campaign / "governance"
    governance.mkdir()
    shutil.copyfile(experiment / "run/candidate.json", governance / "selected-regression.json")
    shutil.copyfile(experiment / "run/baseline.json", governance / "regression-baseline.json")
    before = json.loads((experiment / "run/baseline.json").read_text())
    after = json.loads((experiment / "run/candidate.json").read_text())
    checkpoint = directory_sha256(experiment / "run/checkpoint")
    evidence = governance / f"{config.experiment_id}.json"
    write_json(
        evidence,
        {
            "experimentId": config.experiment_id,
            "decision": "approved",
            "reasonCodes": [],
            "checks": dict.fromkeys(
                [
                    "checksumVerified",
                    "dataAllowed",
                    "evaluationReproducible",
                    "objectiveAligned",
                    "regressionsPassed",
                ],
                True,
            ),
            "facts": {
                "checkpoint": dict.fromkeys(
                    ["recomputed", "recordedByExperiment", "recordedByRegistry"], checkpoint
                ),
                "evaluationReport": {
                    "recomputed": file_sha256(experiment / "evaluator/report.json")
                },
                "regressionSuite": {
                    "baselineCorrect": before["num_correct"],
                    "candidateCorrect": after["num_correct"],
                    "candidateReportSha256": file_sha256(governance / "selected-regression.json"),
                },
            },
        },
    )
    write_json(
        campaign / "research-report.json",
        {
            "campaignId": config.campaign_id,
            "campaignStatus": "stopped",
            "bestCheckpointPromoted": {
                "experimentId": config.experiment_id,
                "checkpointSha256": checkpoint,
            },
            "governance": [
                {
                    "experimentId": config.experiment_id,
                    "label": "selected",
                    "decision": "approved",
                    "promotion": "PROMOTED",
                    "evidenceSha256": file_sha256(evidence),
                }
            ],
        },
    )
    return campaign


@pytest.mark.integration
def test_freeze_rehearse_and_portable_adapter_bundle(harness_run, tmp_path):
    from forge_training.final_cli import evaluate_release
    from forge_training.locking import HELD, acquire_model_lock
    from forge_training.release import freeze_release

    campaign = make_campaign(harness_run, tmp_path)
    _, _, harness, _ = harness_run
    protocol = HarnessProtocol.model_validate(
        json.loads((harness / "plan.json").read_text())["protocol"]
    )
    limits = ResourceLimits(max_seconds_per_model=90, min_available_gib=0.25, min_disk_free_gib=1.0)
    release = tmp_path / "release"
    result = freeze_release(campaign, release, protocol, limits, harness)
    assert result["status"] == "frozen"
    assert verify_bundle(release)["status"] == "verified"
    assert (release / "checkpoint/adapter_model.safetensors").exists()
    lock = acquire_model_lock("harness-rehearsal-test")
    try:
        rehearsed = evaluate_release(release, tmp_path / "rehearsal")
    finally:
        if lock:
            lock.close()
        os.environ.pop(HELD, None)
    assert rehearsed["comparison"]["role"] == "rehearsal"
    assert not rehearsed["comparison"]["final_improvement_supported"]
    assert verify_bundle(tmp_path / "rehearsal")["status"] == "verified"


@pytest.mark.integration
def test_veto_or_wrong_protocol_cannot_freeze(harness_run, tmp_path):
    from forge_training.release import freeze_release

    campaign = make_campaign(harness_run, tmp_path)
    _, _, harness, _ = harness_run
    with pytest.raises(ValueError, match="same model, adapters, and protocol"):
        freeze_release(campaign, tmp_path / "bad", HarnessProtocol(), ResourceLimits(), harness)
    report_path = campaign / "research-report.json"
    report = json.loads(report_path.read_text())
    report["governance"][0]["decision"] = "rejected"
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="approval"):
        freeze_release(campaign, tmp_path / "vetoed", HarnessProtocol(), ResourceLimits(), harness)


@pytest.mark.integration
def test_harness_never_silently_truncates(harness_run, tmp_path):
    root, config_path, _, _ = harness_run
    checkpoint = root / "fixture/run/checkpoint"
    # Deliberately too much prompt+generation for the fixture's 256-token context.
    protocol = HarnessProtocol(num_fewshot=0, max_new_tokens=255, max_prompt_tokens=32)
    limits = ResourceLimits(max_seconds_per_model=90, min_available_gib=0.25, min_disk_free_gib=1.0)
    from forge_training.locking import HELD, acquire_model_lock

    lock = acquire_model_lock("harness-truncation-test")
    try:
        with pytest.raises(RuntimeError, match="baseline failed"):
            development_run(
                config_path,
                checkpoint,
                directory_sha256(checkpoint),
                tmp_path / "long",
                protocol,
                limits,
            )
    finally:
        if lock:
            lock.close()
        os.environ.pop(HELD, None)
    assert "refusing silent truncation" in (tmp_path / "long/baseline.log").read_text()
    assert not (tmp_path / "long/comparison.json").exists()
    assert (tmp_path / "long/failure.json").exists()


def test_final_data_is_read_only_after_durable_consumption_and_failure_spends_grant(
    tmp_path, monkeypatch
):
    from forge_training import final_cli
    from forge_training.artifacts import runtime_versions
    from forge_training.harness import harness_identity, task_path

    authority, release = tmp_path / "authority", tmp_path / "release"
    init_authority(authority)
    release.mkdir()
    write_json(
        release / "release.json",
        {
            "campaign_id": "one-shot",
            "model": {"model_status": "instruct"},
            "protocol": HarnessProtocol().model_dump(),
            "limits": ResourceLimits(min_available_gib=0.25, min_disk_free_gib=1.0).model_dump(),
            "runtime": {**runtime_versions(), "lm_eval": harness_identity()},
            "task_sha256": file_sha256(task_path()),
        },
    )
    seal_bundle(release)
    grant = json.loads(Path(authorize(authority, release)["grant"]).read_text())
    reads = []

    def controlled_reader(_):
        with sqlite3.connect(authority / "registry.sqlite") as conn:
            assert conn.execute("SELECT status FROM grants").fetchone()[0] == "started"
        reads.append(True)
        raise OSError("FAULT: final evaluator failed after grant consumption")

    monkeypatch.setattr(final_cli, "read_sealed_rows", controlled_reader)
    with pytest.raises(ValueError, match="signature"):
        final_cli.evaluate_release(
            release, tmp_path / "forged", authority, {**grant, "signature": "0" * 64}
        )
    assert reads == []
    with pytest.raises(OSError, match="FAULT"):
        final_cli.evaluate_release(release, tmp_path / "attempt", authority, grant)
    assert reads == [True]
    with sqlite3.connect(authority / "registry.sqlite") as conn:
        assert conn.execute("SELECT status FROM grants").fetchone()[0] == "failed"
    with pytest.raises(ValueError, match="already consumed"):
        final_cli.evaluate_release(release, tmp_path / "retry", authority, grant)
    assert reads == [True]


@pytest.mark.integration
def test_legacy_manifest_default_is_normalized_without_rewriting(harness_run, tmp_path):
    import shutil

    from forge_training.artifacts import directory_manifest

    root, _, _, _ = harness_run
    run = tmp_path / "legacy"
    shutil.copytree(root / "fixture/run", run)
    for side in ("baseline", "candidate"):
        path = run / f"{side}.json"
        report = json.loads(path.read_text())
        report["dataset"].pop("record_kind")
        path.write_text(json.dumps(report))
    manifest = json.loads((run / "manifest.json").read_text())
    files = directory_manifest(run)
    files.pop("manifest.json")
    manifest["files"] = files
    (run / "manifest.json").write_text(json.dumps(manifest))
    before = directory_sha256(run)
    assert verify_run(run)["status"] == "verified"
    assert directory_sha256(run) == before


def ablation_entry(name, max_new_tokens, baseline, candidate):
    return {
        "name": name,
        "protocol": HarnessProtocol(max_new_tokens=max_new_tokens).model_dump(),
        "comparison": compare_harness(baseline, candidate),
    }


def test_ablation_variants_are_bounded_distinct_and_cannot_change_seed_or_model():
    from pydantic import ValidationError

    from forge_training.harness_config import AblationSpec, ablation_protocols

    base = HarnessProtocol(num_fewshot=0)
    spec = AblationSpec.model_validate(
        {"variants": [{"name": "reference"}, {"name": "long", "max_new_tokens": 1024}]}
    )
    protocols = ablation_protocols(base, spec)
    assert [p.max_new_tokens for _, p in protocols] == [512, 1024]
    assert all(p.seed == base.seed and p.device == base.device for _, p in protocols)
    for bad in (
        {"variants": [{"name": "only"}]},
        {"variants": [{"name": "a"}, {"name": "b", "seed": 7}]},
        {"variants": [{"name": "a"}, {"name": "B"}]},
        {"variants": [{"name": "a"}, {"name": "b", "max_new_tokens": 4096}]},
    ):
        with pytest.raises(ValidationError):
            AblationSpec.model_validate(bad)
    for duplicate in (
        {"variants": [{"name": "a"}, {"name": "a", "max_new_tokens": 64}]},
        {"variants": [{"name": "a"}, {"name": "b", "max_new_tokens": 512}]},
    ):
        with pytest.raises(ValueError, match="unique|distinct"):
            ablation_protocols(base, AblationSpec.model_validate(duplicate))


def test_ablation_verdict_separates_robust_fragile_and_length_artifacts():
    from forge_training.harness_analysis import summarize_ablation

    baseline, candidate = paired_reports()
    # Strict gains with flexible already correct: release gate passes in both variants.
    robust = summarize_ablation(
        [
            ablation_entry("a", 512, baseline, candidate),
            ablation_entry("b", 1024, baseline, candidate),
        ]
    )
    assert robust["verdict"] == "robust"
    assert robust["stable_strict_gain_ids"] == [str(i) for i in sorted(range(16), key=str)]
    assert robust["gain_without_truncation"]

    # Under a short cap the baseline is truncated and the candidate "gains"; at a long
    # cap both policies finish and the gain disappears, with a flexible regression.
    capped_base = copy.deepcopy(baseline)
    for row in capped_base["samples"]:
        row["hit_limit"] = True
    long_candidate = copy.deepcopy(candidate)
    for row in long_candidate["samples"]:
        row["strict"] = False
    long_candidate["samples"][0]["flexible"] = False
    fragile = summarize_ablation(
        [
            ablation_entry("short", 64, capped_base, candidate),
            ablation_entry("long", 1024, baseline, long_candidate),
        ]
    )
    assert fragile["verdict"] == "fragile"
    assert fragile["uncapped_variants"] == ["long"]
    assert not fragile["gain_without_truncation"]
    assert fragile["stable_strict_gain_ids"] == []

    none = summarize_ablation(
        [
            ablation_entry("x", 64, baseline, long_candidate),
            ablation_entry("y", 128, baseline, long_candidate),
        ]
    )
    assert none["verdict"] == "no_gain"
    with pytest.raises(ValueError, match="at least two"):
        summarize_ablation([ablation_entry("x", 64, baseline, candidate)])
    final = copy.deepcopy(baseline), copy.deepcopy(candidate)
    for report in final:
        report["role"] = "final"
    with pytest.raises(ValueError, match="development data only"):
        summarize_ablation([ablation_entry("x", 64, *final), ablation_entry("y", 128, *final)])


@pytest.mark.integration
def test_real_ablation_runs_predeclared_variants_and_detects_tampering(harness_run, tmp_path):
    import shutil

    from forge_training.harness_config import AblationSpec
    from forge_training.harness_runner import ablation_run
    from forge_training.locking import HELD, acquire_model_lock

    root, config_path, _, _ = harness_run
    checkpoint = root / "fixture/run/checkpoint"
    spec = AblationSpec.model_validate(
        {"variants": [{"name": "short", "max_new_tokens": 4}, {"name": "long"}]}
    )
    limits = ResourceLimits(max_seconds_per_model=90, min_available_gib=0.25, min_disk_free_gib=1.0)
    output = tmp_path / "ablation"
    lock = acquire_model_lock("harness-ablation-test")
    try:
        result = ablation_run(
            config_path,
            checkpoint,
            directory_sha256(checkpoint),
            output,
            HarnessProtocol(num_fewshot=0, max_new_tokens=8, max_prompt_tokens=128),
            spec,
            limits,
        )
    finally:
        if lock:
            lock.close()
        os.environ.pop(HELD, None)
    summary = result["summary"]
    assert [v["name"] for v in summary["variants"]] == ["short", "long"]
    assert [v["protocol"]["max_new_tokens"] for v in summary["variants"]] == [4, 8]
    assert summary["num_examples"] == 2
    assert summary["verdict"] in {"robust", "fragile", "no_gain"}
    assert verify_bundle(output)["status"] == "verified"
    for name in ("short", "long"):
        assert verify_bundle(output / f"variant-{name}")["status"] == "verified"
        assert not json.loads((output / f"variant-{name}/result.json").read_text())[
            "official_test_loaded"
        ]

    # A summary edited to look better, even with a resealed manifest, is caught.
    shutil.copytree(output, tmp_path / "copy")
    forged = json.loads((tmp_path / "copy/summary.json").read_text())
    forged["verdict"] = "robust" if forged["verdict"] != "robust" else "no_gain"
    for name in ("summary.json", "bundle.json"):
        (tmp_path / "copy" / name).unlink()
    write_json(tmp_path / "copy/summary.json", forged)
    seal_bundle(tmp_path / "copy")
    with pytest.raises(ValueError, match="Ablation summary differs"):
        verify_bundle(tmp_path / "copy")
    with pytest.raises(FileExistsError):
        ablation_run(
            config_path,
            checkpoint,
            directory_sha256(checkpoint),
            output,
            HarnessProtocol(num_fewshot=0),
            spec,
            limits,
        )
