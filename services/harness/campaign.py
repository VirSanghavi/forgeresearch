"""Opt-in durable supervision of the existing Forge campaign worker."""

import os
import json
import signal
import sys

from services.core import store
from services.harness.runtime import Harness
from services.harness.schema import Workflow


def campaign_workflow(campaign_id, config, root, environment):
    # Existing campaigns intentionally inherit their launch environment, including configured
    # actor/relay/model providers. Only variable NAMES enter the frozen workflow.
    return Workflow.model_validate(
        {
            "name": campaign_id,
            "max_seconds": config["max_seconds"],
            "max_parallel": 1,
            "tasks": [
                {
                    "id": "campaign",
                    "timeout_seconds": config["max_seconds"],
                    "resources": ["model"],
                    "command": ["{python}", "campaign_entry.py", str(root), campaign_id],
                    "inputs": [{"path": "campaign_entry.py", "source": "services/harness/campaign_entry.py"}],
                    "env": sorted(environment),
                    "max_output_bytes": 64 * 1024 * 1024,
                }
            ],
        }
    )


def main(campaign_id):
    snapshot = store.get(campaign_id)
    if snapshot is None:
        raise ValueError("Campaign does not exist")
    folder = store.ARTIFACTS / campaign_id / "workflow"
    workflow = campaign_workflow(campaign_id, snapshot["config"], store.ROOT, os.environ)
    harness = Harness(workflow, store.ROOT, folder)
    signal.signal(signal.SIGTERM, lambda *_: harness.stop.set())
    signal.signal(signal.SIGINT, lambda *_: harness.stop.set())
    result = harness.run()
    if result["status"] != "succeeded":
        latest = store.get(campaign_id)
        if latest["status"] not in {"completed", "cancelled", "failed"}:
            from services.core.engine import actor

            message = "Durable worker stopped; inspect workflow evidence before starting new research."
            status = "cancelled" if result["status"] == "cancelled" else "failed"
            store.patch(campaign_id, lambda state: state.update(status=status, error=message))
            store.event(campaign_id, "error", message, "trainer")
            try:
                authority = actor(campaign_id)
                snapshot = json.loads(authority.snapshot())
                for key, reservation in snapshot["reservations"].items():
                    if reservation["status"] == "reserved":
                        authority.settle(key, reservation["seconds"])
                authority.transition(status, message)
            except Exception:
                store.event(
                    campaign_id, "error", "Actor reconciliation required after worker stop.", "trainer"
                )
    return 0 if result["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
