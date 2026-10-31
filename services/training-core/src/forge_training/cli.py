"""Worker CLI. No command in this package accepts a sealed final dataset."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

# Commands that load model weights; the host runs at most one at a time.
MODEL_COMMANDS = frozenset({"run", "_train", "evaluate", "smoke"})


def main() -> int:
    parser = argparse.ArgumentParser(prog="forge-train")
    commands = parser.add_subparsers(dest="command", required=True)
    schema = commands.add_parser("schema")
    schema.add_argument("--kind", choices=("experiment", "evaluation"), default="experiment")
    for name in ("validate", "run", "_train", "protocol"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        if name == "run":
            command.add_argument("--frozen-baseline", type=Path)
            command.add_argument("--frozen-baseline-sha256")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--config", type=Path, required=True)
    evaluation.add_argument("--checkpoint", type=Path)
    evaluation.add_argument("--expected-hash")
    evaluation.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--run-dir", type=Path, required=True)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--output", type=Path, required=True)
    prepare = commands.add_parser("prepare-gsm8k")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--revision", required=True)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--train-size", type=int, default=256)
    prepare.add_argument("--dev-size", type=int, default=64)
    args = parser.parse_args()

    def terminate(_signal, _frame):
        raise KeyboardInterrupt("Worker received a termination signal")

    signal.signal(signal.SIGTERM, terminate)
    try:
        from .locking import acquire_model_lock

        # Held until this process exits; waiting happens before any phase deadline starts.
        lock = acquire_model_lock(args.command) if args.command in MODEL_COMMANDS else None
        if args.command == "schema":
            from .config import ExperimentConfig
            from .reports import EvaluationReport

            contract = ExperimentConfig if args.kind == "experiment" else EvaluationReport
            result = contract.model_json_schema()
        elif args.command == "prepare-gsm8k":
            from .data import prepare_gsm8k

            prepare_gsm8k(args.output, args.revision, args.seed, args.train_size, args.dev_size)
            result = {
                "status": "prepared",
                "output": str(args.output.resolve()),
                "official_test_loaded": False,
            }
        elif args.command == "smoke":
            from .fixtures import make_smoke_fixture
            from .worker import run_experiment, verify_run

            config_path = make_smoke_fixture(args.output)
            result = run_experiment(config_path)
            result["verification"] = verify_run(args.output / "run")
            result["evidence_scope"] = "offline random tiny model; mechanics only"
        elif args.command == "verify":
            from .worker import verify_run

            result = verify_run(args.run_dir)
        else:
            from .artifacts import EventLog, write_json
            from .config import load_config
            from .worker import run_experiment, validate_inputs

            config = load_config(args.config)
            if args.command == "validate":
                result = {
                    "status": "valid",
                    "datasets": validate_inputs(config),
                    "config": config.model_dump(),
                }
            elif args.command == "run":
                if bool(args.frozen_baseline) != bool(args.frozen_baseline_sha256):
                    raise ValueError(
                        "--frozen-baseline and --frozen-baseline-sha256 must be provided together"
                    )
                result = run_experiment(
                    args.config, args.frozen_baseline, args.frozen_baseline_sha256
                )
            elif args.command == "protocol":
                from .data import load_dataset
                from .evaluation import evaluation_protocol

                _, manifest = load_dataset(config.development, "development")
                result = {
                    "campaign_id": config.campaign_id,
                    "experiment_id": config.experiment_id,
                    "model": config.model.model_dump(),
                    "dataset": manifest.model_dump(),
                    "protocol": evaluation_protocol(config),
                }
            elif args.command == "evaluate":
                from .evaluation import evaluate

                if bool(args.checkpoint) != bool(args.expected_hash):
                    raise ValueError("--checkpoint and --expected-hash must be provided together")
                if args.output.exists():
                    raise FileExistsError(f"Refusing to overwrite {args.output}")
                result = evaluate(config, args.checkpoint, args.expected_hash)
                write_json(args.output, result)
            else:
                validate_inputs(config, check_output=False)
                events = EventLog(
                    Path(config.output_dir) / "events.jsonl",
                    config.campaign_id,
                    config.experiment_id,
                    producer="optimizer",
                )
                if config.training.method == "grpo":
                    from .data import load_tasks
                    from .grpo import train_grpo

                    tasks, _ = load_tasks(config.train)
                    result = train_grpo(config, tasks, Path(config.output_dir), events)
                else:
                    from .data import load_dataset
                    from .training import train

                    rows, _ = load_dataset(config.train, "train")
                    result = train(config, rows, Path(config.output_dir), events)
        print(json.dumps(result, indent=2, allow_nan=False))
        del lock
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print(
            json.dumps(
                {"status": "failed", "error_type": type(error).__name__, "message": str(error)}
            ),
            file=sys.stderr,
        )
        return 1
