"""python -m services.core.overseer token create|revoke, or cycle: manage ingest and run a check."""

import argparse
import json

from dotenv import load_dotenv

from services.core import store


def main(argv=None):
    load_dotenv(store.ROOT / ".env")
    parser = argparse.ArgumentParser(prog="python -m services.core.overseer")
    commands = parser.add_subparsers(dest="command", required=True)
    token = commands.add_parser("token").add_subparsers(dest="action", required=True)
    create = token.add_parser("create", help="Issue an ingest token for one deployment and task pack")
    create.add_argument("--source", required=True)
    create.add_argument("--pack", required=True)
    create.add_argument("--daily-cap", type=int, default=10_000)
    token.add_parser("revoke").add_argument("id")
    commands.add_parser("cycle", help="Pull, check, judge and raise alerts once")
    args = parser.parse_args(argv)
    from services.core import packs
    from services.core.overseer import pull, worker

    if args.command == "cycle":
        result = worker.cycle()
    elif args.action == "create":
        packs.get(args.pack)
        result = pull.create_token(args.source, args.pack, args.daily_cap)
        result["note"] = "Shown once. Send it as 'Authorization: Bearer <token>' to the ingest function."
    else:
        result = pull.revoke_token(args.id)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
