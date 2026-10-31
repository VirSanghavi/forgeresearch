"""python -m services.harness: validate, run, resume, inspect, cancel and verify."""

import argparse
import importlib.util
import json
import os
import signal
import sys
from pathlib import Path

from services.harness.runtime import Harness, inspect_run, verify_run
from services.harness.schema import Workflow


def main(argv=None):
    parser = argparse.ArgumentParser(description="Forge durable workflow harness")
    commands = parser.add_subparsers(dest="action", required=True)
    validate = commands.add_parser("validate", help="Validate a workflow without executing it")
    validate.add_argument("manifest", type=Path)
    run = commands.add_parser("run", help="Execute a frozen workflow")
    run.add_argument("manifest", type=Path)
    run.add_argument("--root", type=Path, default=Path.cwd(), help="Root for declared source inputs")
    run.add_argument("--output", type=Path, required=True)
    resume = commands.add_parser("resume", help="Resume from the frozen manifest and verified artifacts")
    resume.add_argument("output", type=Path)
    resume.add_argument("--retry-interrupted", action="store_true")
    for name in ("status", "verify", "cancel", "events"):
        command = commands.add_parser(name)
        command.add_argument("output", type=Path)
    commands.add_parser("doctor", help="Report provider prerequisites without provisioning compute")
    args = parser.parse_args(argv)
    try:
        if args.action == "doctor":
            print(
                json.dumps(
                    {
                        "local": {"available": os.name == "posix", "python": sys.executable},
                        "superserve": {
                            "sdk_installed": importlib.util.find_spec("superserve") is not None,
                            "api_key_configured": bool(os.environ.get("SUPERSERVE_API_KEY")),
                            "python_template_required": True,
                        },
                    },
                    indent=2,
                )
            )
            return 0
        if args.action in ("validate", "run"):
            plan = Workflow.model_validate_json(args.manifest.read_text())
            if args.action == "validate":
                print(json.dumps({"valid": True, "name": plan.name, "tasks": len(plan.tasks)}))
                return 0
        elif args.action == "verify":
            print(json.dumps(verify_run(args.output)))
            return 0
        else:
            state = inspect_run(args.output)
            if args.action == "cancel":
                (args.output / "CANCEL").touch()
                print(json.dumps({"status": "cancellation_requested"}))
                return 0
            if args.action == "status":
                print(
                    json.dumps(
                        {
                            "status": state["status"],
                            "name": state["plan"]["name"],
                            "tasks": {
                                key: {"status": value["status"], "attempts": len(value["attempts"])}
                                for key, value in state["tasks"].items()
                            },
                        },
                        indent=2,
                    )
                )
                return 0
            if args.action == "events":
                for event in state["events"]:
                    print(json.dumps(event))
                return 0
            plan = Workflow.model_validate(state["plan"])
            args.root = Path(state["root"])
        harness = Harness(
            plan, args.root, args.output, on_event=lambda event: print(json.dumps(event), flush=True)
        )
        previous = {}
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, lambda *_: harness.stop.set())
        try:
            result = harness.run(
                resume=args.action == "resume", retry_interrupted=getattr(args, "retry_interrupted", False)
            )
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
        return 0 if result["status"] == "succeeded" else 1
    except (ValueError, OSError, RuntimeError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
