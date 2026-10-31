"""Research-side harness CLI: development data only; final access lives in forge-final."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path

from .harness_config import AblationSpec, HarnessProtocol, ResourceLimits


def read_settings(path: Path | None, cls):
    return cls.model_validate_json(path.read_text()) if path else cls()


def main() -> int:
    parser = argparse.ArgumentParser(prog="forge-harness")
    commands = parser.add_subparsers(dest="command", required=True)
    dev = commands.add_parser("development")
    ablation = commands.add_parser("ablation")
    for command in (dev, ablation):
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--expected-hash", required=True)
    ablation.add_argument("--variants", type=Path, required=True)
    for command in (dev, ablation, commands.add_parser("freeze")):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--protocol", type=Path)
        command.add_argument("--limits", type=Path)
        if command not in (dev, ablation):
            command.add_argument("--campaign", type=Path, required=True)
            command.add_argument("--harness-evidence", type=Path, required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--run-dir", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--bundle", type=Path, required=True)
    verify.add_argument("--expected-hash")
    child = commands.add_parser("_evaluate")
    child.add_argument("--plan", type=Path, required=True)
    child.add_argument("--side", choices=("baseline", "candidate"), required=True)
    args = parser.parse_args()
    try:
        from .harness_runner import ablation_run, audit_legacy, development_run, verify_bundle

        def terminate(_signal, _frame):
            raise KeyboardInterrupt("Harness terminated")

        signal.signal(signal.SIGTERM, terminate)
        if args.command == "_evaluate":
            # Internal worker requires a parent holding the host model lock.
            if os.environ.get("FORGE_MODEL_LOCK_HELD") != "1":
                raise ValueError("Use the supervised harness command")
            from .harness import evaluate_side

            result = evaluate_side(args.plan.resolve(), args.side)
        elif args.command == "verify":
            result = verify_bundle(args.bundle, args.expected_hash)
        elif args.command == "audit":
            result = audit_legacy(args.run_dir, args.output.resolve())
        else:
            protocol = read_settings(args.protocol, HarnessProtocol)
            limits = read_settings(args.limits, ResourceLimits)
            if args.command == "freeze":
                from .release import freeze_release

                result = freeze_release(
                    args.campaign.resolve(),
                    args.output.resolve(),
                    protocol,
                    limits,
                    args.harness_evidence.resolve(),
                )
            else:
                from .locking import HELD, acquire_model_lock

                # A caller cannot accidentally bypass exclusion via the worker env switch.
                os.environ.pop(HELD, None)
                if os.environ.get("FORGE_MODEL_LOCK") == "off":
                    raise ValueError("Harness requires the host model-job lock")
                lock = acquire_model_lock(f"harness-{args.command}")
                try:
                    if args.command == "ablation":
                        result = ablation_run(
                            args.config,
                            args.checkpoint,
                            args.expected_hash,
                            args.output.resolve(),
                            protocol,
                            read_settings(args.variants, AblationSpec),
                            limits,
                        )
                    else:
                        result = development_run(
                            args.config,
                            args.checkpoint,
                            args.expected_hash,
                            args.output.resolve(),
                            protocol,
                            limits,
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
