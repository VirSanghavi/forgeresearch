"""Hephaestus's durable mind, as the rest of the bot sees it: a thin client for its Terse actor.

The actor decides; this only encodes calls and says plainly when Terse can't be reached. Writes wait long
enough to outlast a runtime restart, and each call keeps one request ID across retries, so a call that
already committed isn't run again (services/core/actor_client.py).
"""

import json
import os

from durable_actors.client import ActorInvocationError, ActorProtocolError

from services.actors.generated import actors
from services.core.actor_client import RETRYABLE, ReconciledTransport, StableClient

# Terse fences its data for about 30 seconds after a SIGKILL. Reads answer a person, so they fail fast.
WRITE_RETRY_SECONDS = 60
READ_RETRY_SECONDS = 3
MAX_RESULT = 8000
_transports = {}


class Unavailable(Exception):
    """Terse isn't answering, so nothing was recorded and nothing should be done."""


def _actor_id():
    return os.environ.get("FORGE_HEPHAESTUS_ACTOR") or "hephaestus"


def _transport(patient):
    url = os.environ.get("DURABLE_ACTORS_CONTROL_PLANE_URL", "http://127.0.0.1:7100")
    key = (url, patient)
    if key not in _transports:
        found = ReconciledTransport(StableClient(url), WRITE_RETRY_SECONDS if patient else READ_RETRY_SECONDS)
        found.role = "hephaestus"
        _transports[key] = found
    return _transports[key]


def _call(method, *arguments, patient=True):
    actor = actors.HephaestusActor.get(_actor_id(), _transport(patient))
    try:
        return getattr(actor, method)(*arguments)
    except ActorProtocolError as error:
        raise Unavailable(type(error).__name__) from error
    except ActorInvocationError as error:
        if error.code in RETRYABLE:
            raise Unavailable(error.code) from error
        raise ValueError(str(error)) from error
    except OSError as error:
        raise Unavailable(type(error).__name__) from error
    except Exception as error:
        # The client raises its HTTP library's errors when the runtime is gone.
        if type(error).__module__.startswith("httpx"):
            raise Unavailable(type(error).__name__) from error
        raise


def _read(method, *arguments):
    return json.loads(_call(method, *arguments, patient=False))


def _write(method, *arguments):
    return json.loads(_call(method, *arguments))


class Ledger:
    """One conversation's view of the mind. Limits, the audit chain and notes are shared by all."""

    def __init__(self, conversation="lab"):
        self.conversation = conversation

    # Conversations and turns.

    def cursor(self):
        return _call("cursor", self.conversation, patient=False)

    def advance(self, position):
        return _call("advance", self.conversation, position)

    def begin_turn(self, source, kind, role, text, position=-1, show=False):
        return _write("begin_turn", self.conversation, str(source), kind, role, text, position, show)

    def finish_turn(self, turn_id, reply, status="done", announce=False):
        return _write("finish_turn", turn_id, status, reply or "", announce)

    def open_turns(self, before):
        return _read("open_turns", self.conversation, before)

    def resume_turn(self, turn_id):
        return _call("resume_turn", turn_id)

    def memory(self, limit):
        return _read("memory", self.conversation, limit)

    def activity(self):
        return _read("activity", self.conversation)

    # The action journal.

    def begin_action(self, key, turn_id, tool, arguments):
        return _write("begin_action", key, turn_id, tool, json.dumps(arguments, sort_keys=True))

    def finish_action(self, key, status, result=None):
        kept = json.dumps(result)
        return _write(
            "finish_action", key, status, kept if len(kept) <= MAX_RESULT else '{"kept": "too long to keep"}'
        )

    # Limits.

    def take(self, name, limit, amount=1, window="day"):
        return _write("take", name, float(limit), float(amount), window)["ok"]

    def give_back(self, name, amount=1):
        return _call("give_back", name, float(amount))

    def used(self, name):
        return _call("used", name, patient=False)

    def begin_model_call(self, call_limit, token_limit):
        return _write("begin_model_call", float(call_limit), float(token_limit))

    def charge(self, model, usage):
        return _call("charge", model, json.dumps(usage or {}))

    def usage(self, days=7):
        return _read("usage", days)

    # Approvals.

    def hold(self, code_hash, tool, arguments, reason, seconds):
        return _write(
            "hold", self.conversation, code_hash, tool, json.dumps(arguments), reason, float(seconds)
        )

    def claim_approval(self, code_hash, decline=False):
        return _write("claim_approval", self.conversation, code_hash, decline)

    def waiting_approvals(self):
        return _read("waiting_approvals", self.conversation)

    # The audit chain.

    def audit(self, entry):
        return _write("audit", json.dumps(entry))

    def read_audit(self, limit):
        return _read("read_audit", limit)

    def verify_audit(self):
        return _read("verify_audit")

    # Campaigns it watches.

    def watch(self, campaign, mission=None):
        return _write("watch", self.conversation, campaign, int(mission or 0))

    def watching(self):
        return _read("watching", self.conversation)

    def set_watch_state(self, campaign, state):
        return _write("set_watch_state", campaign, state)

    def report_ended(self, campaign, text):
        return _write("report_ended", campaign, text)

    def forget_watch(self, campaign):
        return _write("forget_watch", campaign)

    def was_started(self, campaign):
        return _call("was_started", campaign)

    # The chat itself.

    def post(self, text):
        return _call("post", self.conversation, "hephaestus", text)

    def undelivered(self):
        return _read("undelivered", self.conversation)

    def mark_delivered(self, message_id, delivered=True):
        return _write("mark_delivered", message_id, delivered)

    def transcript(self, limit):
        return _read("transcript", self.conversation, limit)

    # Notes, missions and check-ins.

    def remember(self, text):
        return _call("remember", text)

    def forget(self, note_id):
        return _write("forget", int(note_id))

    def notes(self):
        return _read("notes")

    def begin_mission(self, goal, max_campaigns, max_wakes):
        return _write("begin_mission", self.conversation, goal, int(max_campaigns), int(max_wakes))

    def active_mission(self):
        return _read("active_mission", self.conversation)

    def missions(self, limit=5):
        return _read("missions", self.conversation, limit)

    def claim_wake(self, mission_id, event, gap_seconds):
        return _write("claim_wake", mission_id, event, float(gap_seconds))

    def end_mission(self, mission_id, status, outcome):
        return _write("end_mission", mission_id, status, outcome)

    def schedule(self, note, seconds):
        return _call("schedule", self.conversation, note, float(seconds))

    def claim_due(self):
        return _write("claim_due", self.conversation)

    def scheduled(self):
        return _read("scheduled", self.conversation)

    # Small facts.

    def set_fact(self, name, value):
        return _write("set_fact", name, value or "")

    def get_fact(self, name):
        return _call("get_fact", name, patient=False) or None
