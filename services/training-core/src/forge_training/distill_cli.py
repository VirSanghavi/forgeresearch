"""forge-distill: probe a teacher, build a verified teacher dataset, or re-verify one."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .distill import load_config, probe, run, verify


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="forge-distill", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("probe", "run"):
        commands.add_parser(name).add_argument("config", type=Path)
    commands.add_parser("verify").add_argument("output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            result = verify(args.output)
        elif args.command == "probe":
            result = probe(load_config(args.config))
        else:
            result = run(
                load_config(args.config),
                on_progress=lambda p: print(json.dumps({"progress": p}), flush=True),
            )
    except (ValueError, RuntimeError, OSError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
