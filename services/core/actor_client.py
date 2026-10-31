"""Actor calls that survive a runtime crash without repeating work.

Every logical call gets one Terse request ID, reused on every retry. The runtime records each
committed invocation with its request ID (in the same atomic commit as the state change) and
answers a replay with the recorded completion instead of executing again; this survives SIGKILL
and restart. So an ambiguous outcome ("outcome_unknown": the response was lost, the runtime
died mid-call) is retried safely: either the first attempt never committed and the retry runs
once, or it committed and the retry returns its recorded result. The actors' own idempotency
keys (reservation/settlement/experiment/step IDs) are a second, independent layer.

The SDK's public Client.invoke mints a new request ID per call and only reuses it internally
for failures known to precede execution. StableClient exposes the same request path with a
caller-chosen ID (durable-actors 0.14.0, pinned in pyproject.toml).
"""

import json
import os
import time
import uuid
from durable_actors.client import ActorInvocationError, ActorProtocolError, Client
from services.actors.generated import actors

# Ambiguous or not-executed outcomes. Application failures (validation, budget) are final.
RETRYABLE = {"outcome_unknown", "unavailable"}


class StableClient(Client):
    def invoke_with_request_id(self, actor_name, actor_id, method, args, request_id):
        return self._invoke(actor_name, actor_id, method, args, request_id)


def journal(record, folder=None):
    """Append a retry record for the crash-recovery view. Never fails the call."""
    try:
        from services.core import store

        path = (folder or store.DATA / "chaos") / "retries.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass


class ReconciledTransport:
    def __init__(self, client=None, retry_seconds=None, journal_folder=None):
        self.client = client or StableClient()
        # The API answers people, so it fails fast; the campaign worker raises this to outlast
        # a runtime restart, including Terse's owner-lease fence (about 30 s after a SIGKILL).
        self.retry_seconds = (
            retry_seconds
            if retry_seconds is not None
            else float(os.environ.get("FORGE_ACTOR_RETRY_SECONDS", "4"))
        )
        self.journal_folder = journal_folder
        self.role = "client"

    def invoke(self, actor_name, actor_id, method, args):
        request_id = str(uuid.uuid4())
        started = time.monotonic()
        errors = []
        delay = 0.1
        while True:
            try:
                result = self.client.invoke_with_request_id(actor_name, actor_id, method, args, request_id)
            except (ActorInvocationError, ActorProtocolError) as error:
                code = getattr(error, "code", "protocol_error")
                if isinstance(error, ActorInvocationError) and code not in RETRYABLE:
                    raise
                errors.append(code)
                if time.monotonic() - started + delay > self.retry_seconds:
                    self._record(actor_name, actor_id, method, request_id, started, errors, "gave_up")
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 1.0)
                continue
            if errors:
                self._record(actor_name, actor_id, method, request_id, started, errors, "completed")
            return result

    def _record(self, actor_name, actor_id, method, request_id, started, errors, outcome):
        journal(
            {
                "at": time.time(),
                "pid": os.getpid(),
                "role": self.role,
                "actor": actor_name,
                "actor_id": actor_id,
                "method": method,
                "request_id": request_id,
                "attempts": len(errors) + (outcome == "completed"),
                "errors": sorted(set(errors)),
                "ambiguous": "outcome_unknown" in errors or "protocol_error" in errors,
                "waited_seconds": round(time.monotonic() - started, 3),
                "outcome": outcome,
            },
            self.journal_folder,
        )

    def __getattr__(self, name):
        return getattr(self.client, name)


transport = ReconciledTransport()


def get_actor(name, identity):
    return getattr(actors, name).get(identity, transport)
