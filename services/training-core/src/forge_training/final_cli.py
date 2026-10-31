"""Evaluator-owned, one-attempt final access. No final command exists in forge-train.

The authority directory MUST be owned by a separate OS principal/container for a
hostile-worker boundary. Same-UID mode is a local rehearsal, not a security sandbox.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import secrets
import signal
import sqlite3
import sys
from pathlib import Path

from .artifacts import file_sha256, runtime_versions, utc_now, write_json
from .harness import digest_json, harness_identity, task_path
from .harness_config import HarnessProtocol, ResourceLimits
from .harness_runner import new_directory, prepare_plan, run_plan, verify_bundle


def init_authority(path: Path) -> dict:
    path.mkdir(mode=0o700, parents=False, exist_ok=False)
    with (path / "key").open("xb") as stream:
        os.chmod(path / "key", 0o600)
        stream.write(secrets.token_bytes(32))
        stream.flush()
        os.fsync(stream.fileno())
    with sqlite3.connect(path / "registry.sqlite") as connection:
        connection.execute(
            "CREATE TABLE grants (campaign TEXT PRIMARY KEY, release_hash TEXT NOT "
            "NULL, status TEXT NOT NULL, updated_at TEXT NOT NULL, result TEXT)"
        )
    os.chmod(path / "registry.sqlite", 0o600)
    return {
        "status": "initialized",
        "authority": str(path),
        "requires_separate_os_owner_for_isolation": True,
    }


def authority_key(path: Path) -> bytes:
    for entry in (path, path / "key", path / "registry.sqlite"):
        if entry.is_symlink() or entry.stat().st_uid != os.getuid() or entry.stat().st_mode & 0o077:
            raise ValueError("Authority must be private, owned by the evaluator, and not symlinked")
    key = (path / "key").read_bytes()
    if len(key) != 32:
        raise ValueError("Invalid evaluator key")
    return key


def authorize(authority: Path, release: Path) -> dict:
    key = authority_key(authority)
    verified = verify_bundle(release)
    frozen = json.loads((release / "release.json").read_text())
    if frozen["model"]["model_status"] == "fixture":
        raise ValueError("Fixture models may rehearse but cannot consume final test access")
    body = {"campaign_id": frozen["campaign_id"], "release_sha256": verified["bundle_sha256"]}
    signature = hmac.new(key, digest_json(body).encode(), hashlib.sha256).hexdigest()
    with sqlite3.connect(authority / "registry.sqlite") as connection:
        connection.execute(
            "INSERT INTO grants VALUES (?, ?, 'authorized', ?, NULL)",
            (body["campaign_id"], body["release_sha256"], utc_now()),
        )
    grant = {**body, "signature": signature}
    grant_path = authority / f"{digest_json(body)}.grant.json"
    write_json(grant_path, grant)
    return {"status": "authorized", "grant": str(grant_path), **body}


def consume(authority: Path, grant: dict, release: Path) -> dict:
    key = authority_key(authority)
    if set(grant) != {"campaign_id", "release_sha256", "signature"}:
        raise ValueError("Invalid final grant")
    body = {k: grant[k] for k in ("campaign_id", "release_sha256")}
    expected = hmac.new(key, digest_json(body).encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, grant["signature"]):
        raise ValueError("Final grant signature mismatch")
    verify_bundle(release, grant["release_sha256"])
    frozen = json.loads((release / "release.json").read_text())
    if frozen["campaign_id"] != grant["campaign_id"]:
        raise ValueError("Grant belongs to another campaign")
    with sqlite3.connect(authority / "registry.sqlite", timeout=10) as connection:
        connection.execute("BEGIN IMMEDIATE")
        updated = connection.execute(
            "UPDATE grants SET status='started', updated_at=? WHERE campaign=? AND "
            "release_hash=? AND status='authorized'",
            (utc_now(), grant["campaign_id"], grant["release_sha256"]),
        )
        if updated.rowcount != 1:
            raise ValueError("Final evaluation grant already consumed or not authorized")
    return frozen


def read_sealed_rows(frozen: dict) -> tuple[list[dict], dict]:
    # Called ONLY after consume() durably commits. Fetch just this pinned parquet,
    # never a named dataset loader that might prepare both train and test partitions.
    import pyarrow.parquet as parquet
    from huggingface_hub import hf_hub_download

    from .config import Example
    from .harness_config import GSM8K_REVISION

    spec = frozen["final_dataset"]
    expected = {
        "source": "openai/gsm8k",
        "revision": GSM8K_REVISION,
        "file": "main/test-00000-of-00001.parquet",
        "num_examples": 1319,
    }
    if spec != expected:
        raise ValueError("Final evaluation requires the full pinned GSM8K test partition")
    path = Path(
        hf_hub_download(
            repo_id=spec["source"],
            repo_type="dataset",
            revision=spec["revision"],
            filename=spec["file"],
            local_files_only=True,
        )
    )
    data = parquet.read_table(path).to_pylist()
    if len(data) != 1319:
        raise ValueError("Final data must contain all 1319 examples; subsets are forbidden")
    rows = [Example(id=f"gsm8k-test-{i}", **r).model_dump() for i, r in enumerate(data)]
    return rows, {**spec, "source_split": "test", "parquet_sha256": file_sha256(path)}


def evaluate_release(
    release: Path, output: Path, authority: Path | None = None, grant: dict | None = None
) -> dict:
    verify_bundle(release)
    frozen = json.loads((release / "release.json").read_text())
    if frozen["runtime"] != {**runtime_versions(), "lm_eval": harness_identity()} or frozen[
        "task_sha256"
    ] != file_sha256(task_path()):
        raise ValueError("Evaluator runtime differs from frozen release")
    protocol = HarnessProtocol.model_validate(frozen["protocol"])
    limits = ResourceLimits.model_validate(frozen["limits"])
    new_directory(output, limits)
    if (authority is None) != (grant is None):
        raise ValueError("Both evaluator authority and grant are required")
    consumed = False
    try:
        if authority is not None:
            if frozen["model"]["model_status"] == "fixture":
                raise ValueError("Fixture models cannot access final data")
            frozen = consume(authority, grant, release)
            consumed = True
            rows, dataset = read_sealed_rows(frozen)
            role = "final"
        else:
            rows = [json.loads(s) for s in (release / "development.jsonl").read_text().splitlines()]
            dataset = {
                "source": "frozen-development-rehearsal",
                "sha256": file_sha256(release / "development.jsonl"),
            }
            role = "rehearsal"
        fewshot = [json.loads(s) for s in (release / "fewshot.jsonl").read_text().splitlines()]
        from .data import question_key

        if len({question_key(r["question"]) for r in rows}) != len(rows):
            raise ValueError("Duplicate evaluation questions")
        if {question_key(r["question"]) for r in rows} & {
            question_key(r["question"]) for r in fewshot
        }:
            raise ValueError("Few-shot examples overlap evaluation questions")
        if role == "final":
            exposed = set(json.loads((release / "research-question-hashes.json").read_text()))
            if exposed & {digest_json(question_key(r["question"])) for r in rows}:
                raise ValueError("Final questions overlap exposed research data")
        prepare_plan(
            output,
            role=role,
            model=frozen["model"],
            rows=rows,
            fewshot=fewshot,
            dataset=dataset,
            baseline=release / frozen["baseline"]["path"] if frozen["baseline"] else None,
            candidate=release / frozen["candidate"]["path"],
            protocol=protocol,
            limits=limits,
        )
        write_json(
            output / "release-reference.json",
            {
                "bundle_sha256": file_sha256(release / "bundle.json"),
                "campaign_id": frozen["campaign_id"],
                "role": role,
            },
        )
        result = run_plan(output, limits)
        if consumed:
            with sqlite3.connect(authority / "registry.sqlite") as connection:
                connection.execute(
                    "UPDATE grants SET status='completed', result=?, updated_at=? WHERE campaign=?",
                    (result["bundle_sha256"], utc_now(), frozen["campaign_id"]),
                )
        return result
    except BaseException as error:
        if not (output / "failure.json").exists():
            write_json(
                output / "failure.json",
                {"status": "failed", "error": str(error), "final_grant_consumed": consumed},
            )
        if consumed:
            with sqlite3.connect(authority / "registry.sqlite") as connection:
                connection.execute(
                    "UPDATE grants SET status='failed', result=?, updated_at=? WHERE campaign=?",
                    (str(error), utc_now(), frozen["campaign_id"]),
                )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(prog="forge-final")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init-authority")
    init.add_argument("--authority", type=Path, required=True)
    authorize_cmd = commands.add_parser("authorize")
    authorize_cmd.add_argument("--authority", type=Path, required=True)
    authorize_cmd.add_argument("--release", type=Path, required=True)
    for name in ("evaluate", "rehearse"):
        command = commands.add_parser(name)
        command.add_argument("--release", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "evaluate":
            command.add_argument("--authority", type=Path, required=True)
            command.add_argument("--grant", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "init-authority":
            result = init_authority(args.authority.resolve())
        elif args.command == "authorize":
            result = authorize(args.authority.resolve(), args.release.resolve())
        else:
            from .locking import HELD, acquire_model_lock

            def terminate(_signal, _frame):
                raise KeyboardInterrupt("Final evaluator terminated")

            signal.signal(signal.SIGTERM, terminate)
            os.environ.pop(HELD, None)
            if os.environ.get("FORGE_MODEL_LOCK") == "off":
                raise ValueError("Final evaluator requires the host model-job lock")
            lock = acquire_model_lock("harness-final")
            try:
                result = evaluate_release(
                    args.release.resolve(),
                    args.output.resolve(),
                    args.authority.resolve() if args.command == "evaluate" else None,
                    json.loads(args.grant.read_text()) if args.command == "evaluate" else None,
                )
            finally:
                if lock:
                    lock.close()
                os.environ.pop(HELD, None)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print(
            json.dumps(
                {"status": "failed", "error_type": type(error).__name__, "message": str(error)}
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
