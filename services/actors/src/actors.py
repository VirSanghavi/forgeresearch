"""Authoritative Terse state. Model optimization runs outside actors."""

import hashlib
import json
import time
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, cast
from durable_actors import Actor, SqliteValue, persisted
from durable_actors.client import Client


def encode(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class ResearchCampaignActor(Actor):
    state: str = persisted("")

    def initialize(self, config_json: str) -> str:
        if self.state and json.loads(self.state)["config"] != json.loads(config_json):
            raise ValueError("Campaign configuration is immutable")
        if not self.state:
            self.state = encode(
                {
                    "config": json.loads(config_json),
                    "status": "initialized",
                    "reservations": {},
                    "consumed_seconds": 0,
                    "best_checkpoint": None,
                    "selection_frozen": False,
                    "final_claimed": False,
                    "decisions": [],
                    "version": 0,
                }
            )
        return self.state

    def snapshot(self) -> str:
        return self.state

    def transition(self, status: str, decision: str) -> str:
        s = json.loads(self.state)
        if s["status"] == status and s["decisions"] and s["decisions"][-1]["reason"] == decision:
            return self.state
        allowed = {
            "baseline",
            "diagnosing",
            "researching",
            "training",
            "evaluating",
            "reviewing",
            "finalizing",
            "completed",
            "failed",
            "cancelled",
        }
        if status not in allowed or s["status"] in {"completed", "cancelled"}:
            raise ValueError("Invalid campaign transition")
        s["status"] = status
        s["version"] += 1
        s["decisions"] = [*s["decisions"], {"version": s["version"], "status": status, "reason": decision}]
        self.state = encode(s)
        return self.state

    def reserve(self, experiment_id: str, seconds: int) -> str:
        s = json.loads(self.state)
        if experiment_id in s["reservations"]:
            if s["reservations"][experiment_id]["seconds"] != seconds:
                raise ValueError("Idempotency conflict")
            return self.state
        if seconds <= 0 or s["selection_frozen"] or s["status"] in {"cancelled", "completed", "failed"}:
            raise ValueError("Campaign cannot reserve compute")
        reserved = sum(r["seconds"] for r in s["reservations"].values() if r["status"] == "reserved")
        if reserved + s["consumed_seconds"] + seconds > s["config"]["max_seconds"]:
            raise ValueError("Compute budget exhausted")
        s["reservations"][experiment_id] = {
            "seconds": seconds,
            "status": "reserved",
            "reserved_at": time.time(),
        }
        self.state = encode(s)
        return self.state

    def settle(self, experiment_id: str, seconds: float) -> str:
        s = json.loads(self.state)
        r = s["reservations"].get(experiment_id)
        if r is None:
            raise ValueError("Reservation required")
        if r["status"] == "settled":
            if r["actual_seconds"] != seconds:
                raise ValueError("Compute settlement is immutable")
            return self.state
        if seconds < 0 or seconds > r["seconds"]:
            raise ValueError("Usage outside reservation")
        r["status"] = "settled"
        r["actual_seconds"] = seconds
        s["consumed_seconds"] += seconds
        self.state = encode(s)
        return self.state

    def freeze_selection(self, checkpoint_id: str) -> str:
        s = json.loads(self.state)
        if s["selection_frozen"] and s["best_checkpoint"] != checkpoint_id:
            raise ValueError("Checkpoint selection is immutable")
        s["best_checkpoint"] = checkpoint_id
        s["selection_frozen"] = True
        self.state = encode(s)
        return self.state

    def claim_final(self) -> bool:
        s = json.loads(self.state)
        if not s["selection_frozen"] or s["final_claimed"]:
            return False
        s["final_claimed"] = True
        self.state = encode(s)
        return True


class ExperimentActor(Actor):
    state: str = persisted("")

    def initialize(self, spec_json: str) -> str:
        if self.state and json.loads(self.state)["spec"] != json.loads(spec_json):
            raise ValueError("Experiment configuration is immutable")
        if not self.state:
            self.state = encode(
                {"spec": json.loads(spec_json), "status": "proposed", "metrics": [], "result": None}
            )
        return self.state

    def update(self, status: str, payload_json: str) -> str:
        s = json.loads(self.state)
        if status == s["status"]:
            if s["result"] != json.loads(payload_json):
                raise ValueError("Experiment update conflicts with an existing receipt")
            return self.state
        transitions = {
            "proposed": ["approved", "rejected"],
            "approved": ["training", "failed", "cancelled"],
            "training": ["evaluating", "failed", "cancelled"],
            "evaluating": ["completed", "failed", "cancelled"],
        }
        if status != s["status"] and status not in transitions.get(s["status"], []):
            raise ValueError("Invalid experiment transition")
        s["status"] = status
        s["result"] = json.loads(payload_json)
        self.state = encode(s)
        return self.state

    def metric(self, metric_json: str) -> str:
        s = json.loads(self.state)
        m = json.loads(metric_json)
        if s["status"] != "training":
            raise ValueError("Experiment is not training")
        if not any(x["step"] == m["step"] for x in s["metrics"]):
            s["metrics"] = [*s["metrics"], m]
        elif next(x for x in s["metrics"] if x["step"] == m["step"]) != m:
            raise ValueError("Conflicting metric for an existing optimizer step")
        self.state = encode(s)
        return self.state

    def snapshot(self) -> str:
        return self.state


class GovernanceActor(Actor):
    policy: str = persisted("")
    reviews: str = persisted("{}")
    promoted: str = persisted("")

    def initialize(self, policy_json: str) -> str:
        policy = encode(json.loads(policy_json))
        if self.policy and self.policy != policy:
            raise ValueError("Governance policy is immutable")
        self.policy = policy
        return self.policy

    def review(self, evidence_json: str) -> str:
        e, p = json.loads(evidence_json), json.loads(self.policy)
        reasons = []
        if e.get("reward_version") != p["reward_version"]:
            reasons.append("REWARD_FUNCTION_MODIFIED")
        if e.get("sealed_data_access"):
            reasons.append("SEALED_DATA_ACCESS")
        if not e.get("provenance_valid", False):
            reasons.append("MISSING_PROVENANCE")
        if e.get("phase") == "checkpoint":
            if not e.get("weights_changed"):
                reasons.append("NO_WEIGHT_UPDATE")
            if not e.get("independent_reload"):
                reasons.append("CHECKPOINT_NOT_RELOADED")
            if not e.get("artifact_verified") or len(e.get("sha256", "")) != 64:
                reasons.append("ARTIFACT_INTEGRITY_FAILURE")
            if e.get("regression_drop", 1) > p["max_regression_drop"]:
                reasons.append("REGRESSION_THRESHOLD_EXCEEDED")
        # Policies from before task packs carry no pack hash, so their reviews replay unchanged.
        if "task_pack_sha256" in p and e.get("task_pack_sha256") != p["task_pack_sha256"]:
            reasons.append("TASK_PACK_MODIFIED")
        synthetic = e.get("synthetic")
        if synthetic is not None:
            if not synthetic.get("provenance_verified"):
                reasons.append("SYNTHETIC_PROVENANCE_MISSING")
            if not synthetic.get("teacher_terms_present"):
                reasons.append("TEACHER_TERMS_MISSING")
            if not synthetic.get("checker_matches_pack"):
                reasons.append("SYNTHETIC_CHECKER_MISMATCH")
            if not synthetic.get("separation_verified"):
                reasons.append("SYNTHETIC_SEPARATION_FAILED")
            if synthetic.get("verification") != "reference" and not p.get("allow_unverified_synthetic"):
                reasons.append("SYNTHETIC_UNVERIFIED")
        result = {
            "id": e["id"],
            "decision": "rejected" if reasons else "approved",
            "reason_codes": reasons,
            "evidence": e,
            "policy_version": p.get("policy_version", "forge-policy-v1"),
        }
        reviews = json.loads(self.reviews)
        if e["id"] in reviews and reviews[e["id"]] != result:
            raise ValueError("Review evidence is immutable")
        reviews[e["id"]] = result
        self.reviews = encode(reviews)
        return encode(result)

    def promote(self, checkpoint_id: str, sha256: str) -> str:
        review = json.loads(self.reviews).get(checkpoint_id)
        if not review or review["decision"] != "approved" or review["evidence"].get("phase") != "checkpoint":
            raise ValueError("Checkpoint promotion denied: independent approval required")
        if review["evidence"].get("sha256") != sha256:
            raise ValueError("Checkpoint promotion denied: checksum mismatch")
        # Acyclic authority read: governance -> registry. A review arriving before
        # the files is preserved, but cannot authorize a premature promotion.
        with Client() as client:
            registry = json.loads(client.invoke("ArtifactRegistryActor", self.id, "snapshot", []))
        evidence = review["evidence"]
        checkpoint = registry.get(evidence.get("checkpoint_artifact_id", ""))
        evaluation = registry.get(evidence.get("evaluation_artifact_id", ""))
        if not checkpoint or checkpoint.get("kind") != "checkpoint" or checkpoint.get("sha256") != sha256:
            raise ValueError("Checkpoint promotion denied: registered weights are required")
        if (
            not evaluation
            or evaluation.get("kind") != "evaluation"
            or evaluation.get("sha256") != evidence.get("evaluation_sha256")
        ):
            raise ValueError("Checkpoint promotion denied: registered evaluation is required")
        self.promoted = checkpoint_id
        return checkpoint_id

    def snapshot(self) -> str:
        return encode(
            {
                "policy": json.loads(self.policy) if self.policy else {},
                "reviews": json.loads(self.reviews),
                "promoted": self.promoted,
            }
        )


class RLWorldActor(Actor):
    state: str = persisted("")

    def _public(self, state: dict[str, object]) -> str:
        return encode({k: v for k, v in state.items() if k != "_answer"})

    def reset(self, episode_id: str, task: str, answer: str, seed: int) -> str:
        if self.state:
            s = json.loads(self.state)
            if s["episode_id"] == episode_id:
                if s["task"] != task or s["seed"] != seed or s.get("_answer") != answer:
                    raise ValueError("Episode reset conflict")
                return self._public(s)
            if not s["terminated"]:
                raise ValueError("Active episode cannot be overwritten")
        self.state = encode(
            {
                "episode_id": episode_id,
                "task": task,
                "seed": seed,
                "version": 0,
                "terminated": False,
                "reward": None,
                "actions": [],
                "reward_version": "exact-answer-v1",
                "_answer": answer,
            }
        )
        return self._public(json.loads(self.state))

    def observe(self) -> str:
        if not self.state:
            raise ValueError("Environment is not initialized")
        return self._public(json.loads(self.state))

    def step(self, action_json: str) -> str:
        a, s = json.loads(action_json), json.loads(self.state)
        if a["episode_id"] != s["episode_id"]:
            raise ValueError("Episode mismatch")
        existing = next((x for x in s["actions"] if x["action_id"] == a["action_id"]), None)
        if existing:
            if existing["answer"] != a["answer"]:
                raise ValueError("Action ID conflict")
            return self._public(s)
        if a["expected_version"] != s["version"]:
            raise ValueError("State version conflict")
        if s["terminated"]:
            raise ValueError("Episode terminated")
        try:
            correct = Decimal(a["answer"].replace(",", "").strip()) == Decimal(s["_answer"])
        except (InvalidOperation, ValueError):
            correct = False
        s["reward"], s["terminated"], s["version"] = int(correct), True, s["version"] + 1
        s["actions"] = [
            *s["actions"],
            {"action_id": a["action_id"], "answer": a["answer"], "reward": s["reward"]},
        ]
        s["checksum"] = hashlib.sha256(self._public(s).encode()).hexdigest()
        self.state = encode(s)
        return self._public(s)


class ArtifactRegistryActor(Actor):
    registry: str = persisted("{}")

    def register(self, artifact_json: str) -> str:
        a, r = json.loads(artifact_json), json.loads(self.registry)
        if a["id"] in r and r[a["id"]] != a:
            raise ValueError("Immutable artifact conflict")
        if len(a["sha256"]) != 64:
            raise ValueError("SHA-256 required")
        r[a["id"]] = a
        self.registry = encode(r)
        return artifact_json

    def snapshot(self) -> str:
        return self.registry


AGENT_ROLES = ("director", "data", "trainer", "evaluator", "governance")
MAX_ORDERS = 40
MAX_ASSIGNMENTS = 8


class TeamActor(Actor):
    """One campaign's chain of command: what Hephaestus has told its research agents to do.

    Committed here before the worker acts on it, so an order survives a crash of either side, and an
    assignment is handed out exactly once. It holds instructions, never authority over the result:
    budgets, governance and the sealed final live in their own actors and take no input from this one.
    """

    state: str = persisted("")

    def _load(self) -> dict[str, Any]:
        if self.state:
            loaded: dict[str, Any] = json.loads(self.state)
            return loaded
        return {
            "orders": [],
            "settings": {},
            "held": None,
            "assignments": [],
            "finish": None,
            "version": 0,
        }

    def _save(self, state: dict[str, Any]) -> str:
        state["version"] += 1
        self.state = encode(state)
        return self.state

    def snapshot(self) -> str:
        return encode(self._load())

    def order(self, order_json: str) -> str:
        """A standing order for some roles. The same id twice is one order, so a retry is safe."""
        order, state = json.loads(order_json), self._load()
        known = next((o for o in state["orders"] if o["id"] == order["id"]), None)
        if known:
            if known["text"] != order["text"] or known["roles"] != order["roles"]:
                raise ValueError("Order conflicts with an earlier one")
            return encode(known)
        if not order["roles"] or set(order["roles"]) - set(AGENT_ROLES):
            raise ValueError("Orders name research agents")
        if len(state["orders"]) >= MAX_ORDERS:
            raise ValueError(f"This campaign already has {MAX_ORDERS} orders")
        entry = {**order, "at": time.time(), "read": []}
        state["orders"] = [*state["orders"], entry]
        self._save(state)
        return encode(entry)

    def orders_for(self, role: str) -> str:
        return encode([o for o in self._load()["orders"] if role in o["roles"]])

    def mark_read(self, role: str, order_ids_json: str, where: str) -> str:
        """Proof that a role had these orders in front of it when it made this decision."""
        wanted, state = set(json.loads(order_ids_json)), self._load()
        for order in state["orders"]:
            read = {"role": role, "where": where}
            if order["id"] in wanted and read not in order["read"]:
                order["read"] = [*order["read"], read]
        return self._save(state)

    def configure(self, role: str, settings_json: str) -> str:
        """Change how one role works. A null value goes back to the lab's default."""
        if role not in AGENT_ROLES:
            raise ValueError("Settings name a research agent")
        state = self._load()
        merged = {**state["settings"].get(role, {}), **json.loads(settings_json)}
        state["settings"][role] = {k: v for k, v in merged.items() if v is not None}
        self._save(state)
        return encode(state["settings"][role])

    def hold(self, reason: str) -> str:
        state = self._load()
        if not state["held"]:
            state["held"] = {"reason": reason[:500], "at": time.time()}
            self._save(state)
        return encode(state)

    def release(self) -> str:
        state = self._load()
        if state["held"]:
            state["held"] = None
            self._save(state)
        return encode(state)

    def assign(self, assignment_json: str) -> str:
        """Queue the next experiment. It still goes through preflight, budgets and every review."""
        assignment, state = json.loads(assignment_json), self._load()
        known = next((a for a in state["assignments"] if a["id"] == assignment["id"]), None)
        if known:
            return encode(known)
        if len(state["assignments"]) >= MAX_ASSIGNMENTS:
            raise ValueError(f"This campaign already has {MAX_ASSIGNMENTS} assigned experiments")
        entry = {**assignment, "at": time.time(), "taken_by": None}
        state["assignments"] = [*state["assignments"], entry]
        self._save(state)
        return encode(entry)

    def take_assignment(self, experiment_id: str) -> str:
        """Hand the worker the next assigned experiment, once. Asked again for the same experiment,
        it returns the same one, so a retried call can't skip or repeat an assignment."""
        state = self._load()
        for assignment in state["assignments"]:
            if assignment["taken_by"] == experiment_id:
                return encode(assignment)
        for assignment in state["assignments"]:
            if assignment["taken_by"] is None:
                assignment["taken_by"] = experiment_id
                self._save(state)
                return encode(assignment)
        return ""

    def finish(self, reason: str) -> str:
        """Stop proposing experiments and go to the final with what the campaign has."""
        state = self._load()
        if not state["finish"]:
            state["finish"] = {"reason": reason[:500], "at": time.time()}
            self._save(state)
        return encode(state)


MIND_TABLES = (
    "CREATE TABLE IF NOT EXISTS cursors (conversation TEXT PRIMARY KEY, position INTEGER NOT NULL)",
    """CREATE TABLE IF NOT EXISTS turns (
        id INTEGER PRIMARY KEY AUTOINCREMENT, conversation TEXT NOT NULL, source TEXT NOT NULL,
        kind TEXT NOT NULL, role TEXT NOT NULL, text TEXT NOT NULL, status TEXT NOT NULL, reply TEXT,
        attempts INTEGER NOT NULL DEFAULT 0, began_at REAL NOT NULL, ended_at REAL,
        UNIQUE(conversation, source))""",
    """CREATE TABLE IF NOT EXISTS actions (
        key TEXT PRIMARY KEY, turn_id INTEGER NOT NULL, tool TEXT NOT NULL, arguments TEXT NOT NULL,
        status TEXT NOT NULL, result TEXT, began_at REAL NOT NULL, ended_at REAL)""",
    "CREATE INDEX IF NOT EXISTS actions_turn ON actions(turn_id)",
    """CREATE TABLE IF NOT EXISTS counters (
        name TEXT NOT NULL, bucket TEXT NOT NULL, used REAL NOT NULL, PRIMARY KEY(name, bucket))""",
    """CREATE TABLE IF NOT EXISTS approvals (
        id INTEGER PRIMARY KEY AUTOINCREMENT, conversation TEXT NOT NULL, code_hash TEXT NOT NULL,
        tool TEXT NOT NULL, arguments TEXT NOT NULL, reason TEXT NOT NULL, status TEXT NOT NULL,
        held_at REAL NOT NULL, expires_at REAL NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS audit (
        seq INTEGER PRIMARY KEY AUTOINCREMENT, entry TEXT NOT NULL, previous TEXT NOT NULL,
        hash TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS watches (
        campaign TEXT PRIMARY KEY, conversation TEXT NOT NULL, mission INTEGER, state TEXT NOT NULL,
        reported INTEGER NOT NULL DEFAULT 0)""",
    """CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT, conversation TEXT NOT NULL, sender TEXT NOT NULL,
        text TEXT NOT NULL, at REAL NOT NULL, delivered INTEGER NOT NULL,
        attempts INTEGER NOT NULL DEFAULT 0)""",
    """CREATE TABLE IF NOT EXISTS notes (
        id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, at REAL NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS missions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, conversation TEXT NOT NULL, goal TEXT NOT NULL,
        status TEXT NOT NULL, max_campaigns INTEGER NOT NULL, max_wakes INTEGER NOT NULL,
        campaigns TEXT NOT NULL DEFAULT '[]', wakes INTEGER NOT NULL DEFAULT 0, seen TEXT NOT NULL DEFAULT '[]',
        last_wake REAL NOT NULL DEFAULT 0, outcome TEXT, began_at REAL NOT NULL, ended_at REAL)""",
    """CREATE TABLE IF NOT EXISTS checks (
        id INTEGER PRIMARY KEY AUTOINCREMENT, conversation TEXT NOT NULL, note TEXT NOT NULL,
        due_at REAL NOT NULL, fired INTEGER NOT NULL DEFAULT 0)""",
    "CREATE TABLE IF NOT EXISTS facts (name TEXT PRIMARY KEY, value TEXT NOT NULL)",
)
MIND_SCHEMA = 1
KEEP_TURNS = 200
KEEP_MESSAGES = 400
MAX_WAITING = 8
MAX_NOTES = 50
MAX_CHECKS = 20
MAX_DELIVERY_ATTEMPTS = 5
TERMINAL_TURN = {"done", "interrupted"}


def _day(moment: float | None = None) -> str:
    return datetime.fromtimestamp(time.time() if moment is None else moment, UTC).strftime("%Y-%m-%d")


def _hour() -> str:
    return datetime.fromtimestamp(time.time(), UTC).strftime("%Y-%m-%dT%H")


def _chain(previous: str, entry: str) -> str:
    return hashlib.sha256((previous + entry).encode()).hexdigest()


class HephaestusActor(Actor):
    """Hephaestus's durable mind, one for the lab, shared by Agent chat and iMessage.

    Everything it must not lose or do twice commits here in one transaction per call: where it has read
    to in each chat, each turn and the actions it took, approvals, limits, the audit chain, what it
    watches and its missions. Rows rather than persisted fields, so a call costs the same however long
    the history gets.
    """

    schema: int = persisted(0)

    def _rows(self, sql: str, *bindings: SqliteValue) -> list[dict[str, SqliteValue]]:
        if self.schema != MIND_SCHEMA:
            for statement in MIND_TABLES:
                self.db.exec(statement)
            self.schema = MIND_SCHEMA
        return self.db.exec(sql, *bindings)

    def _one(self, sql: str, *bindings: SqliteValue) -> dict[str, SqliteValue] | None:
        rows = self._rows(sql, *bindings)
        return rows[0] if rows else None

    # Conversations: where it has read to, and each turn it took.

    def cursor(self, conversation: str) -> int:
        """Where it has read to in this chat, or -1 before it joins."""
        row = self._one("SELECT position FROM cursors WHERE conversation=?", conversation)
        return int(cast(int, row["position"])) if row else -1

    def advance(self, conversation: str, position: int) -> int:
        self._rows(
            "INSERT INTO cursors VALUES(?,?) ON CONFLICT(conversation) "
            "DO UPDATE SET position=max(position, excluded.position)",
            conversation,
            position,
        )
        return self.cursor(conversation)

    def begin_turn(
        self, conversation: str, source: str, kind: str, role: str, text: str, position: int, show: bool
    ) -> str:
        """Record a turn and move the read position in one commit. The same source twice is one turn,
        so two processes, or a retry after a crash, can't act on one message twice."""
        known = self._one(
            "SELECT id, status FROM turns WHERE conversation=? AND source=?", conversation, source
        )
        if position >= 0:
            self.advance(conversation, position)
        if known:
            return encode({"id": known["id"], "status": known["status"], "new": False})
        self._rows(
            "INSERT INTO turns (conversation, source, kind, role, text, status, began_at) "
            "VALUES(?,?,?,?,?,'open',?)",
            conversation,
            source,
            kind,
            role,
            text,
            time.time(),
        )
        turn = self._rows("SELECT last_insert_rowid() AS id")[0]["id"]
        if show:
            self._post(conversation, "you", text, delivered=True)
        return encode({"id": turn, "status": "open", "new": True})

    def finish_turn(self, turn_id: int, status: str, reply: str, announce: bool) -> str:
        """End a turn and queue its reply in one commit, so a reply is never lost or written twice."""
        if status not in TERMINAL_TURN:
            raise ValueError("Unknown turn status")
        turn = self._one("SELECT conversation, status FROM turns WHERE id=?", turn_id)
        if not turn:
            raise ValueError("Unknown turn")
        if turn["status"] != "open":
            return encode({"finished": False})
        self._rows(
            "UPDATE turns SET status=?, reply=?, ended_at=? WHERE id=?", status, reply, time.time(), turn_id
        )
        conversation = str(turn["conversation"])
        posted = (
            self._post(conversation, "hephaestus", reply, delivered=False) if announce and reply else None
        )
        self._rows(
            "DELETE FROM actions WHERE turn_id IN (SELECT id FROM turns WHERE conversation=? AND "
            "id NOT IN (SELECT id FROM turns WHERE conversation=? ORDER BY id DESC LIMIT ?))",
            conversation,
            conversation,
            KEEP_TURNS,
        )
        self._rows(
            "DELETE FROM turns WHERE conversation=? AND "
            "id NOT IN (SELECT id FROM turns WHERE conversation=? ORDER BY id DESC LIMIT ?)",
            conversation,
            conversation,
            KEEP_TURNS,
        )
        return encode({"finished": True, "message": posted})

    def open_turns(self, conversation: str, before: float) -> str:
        """Turns that began before this process did and never ended: a crash interrupted them."""
        rows = self._rows(
            "SELECT id, source, kind, role, text, attempts, began_at FROM turns "
            "WHERE conversation=? AND status='open' AND began_at<? ORDER BY id",
            conversation,
            before,
        )
        return encode([{**row, "actions": self._actions(cast(int, row["id"]))} for row in rows])

    def resume_turn(self, turn_id: int) -> int:
        """Count an attempt to pick an interrupted turn back up, so a turn that crashes it can't loop."""
        self._rows("UPDATE turns SET attempts=attempts+1 WHERE id=?", turn_id)
        row = self._one("SELECT attempts FROM turns WHERE id=?", turn_id)
        return int(cast(int, row["attempts"])) if row else 0

    def memory(self, conversation: str, limit: int) -> str:
        """Its recent finished turns here, oldest first, with who was asking."""
        rows = self._rows(
            "SELECT role, kind, text, reply FROM turns WHERE conversation=? AND status='done' "
            "ORDER BY id DESC LIMIT ?",
            conversation,
            limit,
        )
        return encode(rows[::-1])

    def activity(self, conversation: str) -> str:
        """What it's working on right now, for the page: the open turn and its latest step."""
        turn = self._one(
            "SELECT id FROM turns WHERE conversation=? AND status='open' ORDER BY id DESC LIMIT 1",
            conversation,
        )
        if not turn:
            return encode({"working": False, "tool": None})
        step = self._one(
            "SELECT tool FROM actions WHERE turn_id=? ORDER BY began_at DESC LIMIT 1", cast(int, turn["id"])
        )
        return encode({"working": True, "tool": step["tool"] if step else None})

    # The action journal: what a turn did, so picking it back up never does it twice.

    def _actions(self, turn_id: int) -> list[dict[str, SqliteValue]]:
        return self._rows(
            "SELECT key, tool, arguments, status, result FROM actions WHERE turn_id=? ORDER BY began_at",
            turn_id,
        )

    def begin_action(self, key: str, turn_id: int, tool: str, arguments: str) -> str:
        """Returns 'new' to go ahead, 'done' with the recorded result, or 'started' when an earlier
        attempt began this action and never recorded an outcome. One that failed can be tried again."""
        known = self._one("SELECT status, result FROM actions WHERE key=?", key)
        if known and known["status"] == "failed":
            self._rows("UPDATE actions SET status='started', result=NULL WHERE key=?", key)
            return encode({"status": "new", "result": None})
        if known:
            return encode({"status": known["status"], "result": known["result"]})
        self._rows(
            "INSERT INTO actions (key, turn_id, tool, arguments, status, began_at) VALUES(?,?,?,?,'started',?)",
            key,
            turn_id,
            tool,
            arguments,
            time.time(),
        )
        return encode({"status": "new", "result": None})

    def finish_action(self, key: str, status: str, result: str) -> str:
        self._rows(
            "UPDATE actions SET status=?, result=?, ended_at=? WHERE key=?", status, result, time.time(), key
        )
        return encode({"recorded": True})

    # Limits: counted atomically, so Agent chat and iMessage share them and neither can overrun.

    def take(self, name: str, limit: float, amount: float, window: str) -> str:
        """Use some of a limit, or nothing at all if that would pass it."""
        bucket = _hour() if window == "hour" else _day()
        row = self._one("SELECT used FROM counters WHERE name=? AND bucket=?", name, bucket)
        used = float(cast(float, row["used"])) if row else 0.0
        if used + amount > limit:
            return encode({"ok": False, "used": used})
        self._add(name, bucket, amount)
        return encode({"ok": True, "used": used + amount})

    def give_back(self, name: str, amount: float) -> float:
        self._add(name, _day(), -amount)
        return self.used(name)

    def used(self, name: str) -> float:
        row = self._one("SELECT used FROM counters WHERE name=? AND bucket=?", name, _day())
        return float(cast(float, row["used"])) if row else 0.0

    def _add(self, name: str, bucket: str, amount: float) -> None:
        self._rows(
            "INSERT INTO counters VALUES(?,?,?) ON CONFLICT(name, bucket) DO UPDATE SET used=used+excluded.used",
            name,
            bucket,
            amount,
        )

    def begin_model_call(self, call_limit: float, token_limit: float) -> str:
        """May it make another model call today? Tokens are only known afterwards, so this refuses
        once they've passed the limit, and a call in flight can run a little over."""
        tokens = self.used("tokens")
        if tokens >= token_limit:
            return encode({"ok": False, "why": "tokens", "tokens": tokens})
        taken = json.loads(self.take("model_calls", call_limit, 1, "day"))
        if not taken["ok"]:
            return encode({"ok": False, "why": "calls", "tokens": tokens})
        return encode({"ok": True, "why": None, "tokens": tokens})

    def charge(self, model: str, usage_json: str) -> float:
        """What a model call really used, from the provider's own count."""
        usage, day = json.loads(usage_json), _day()
        details = usage.get("input_tokens_details") or {}
        reasoning = usage.get("output_tokens_details") or {}
        for name, amount in (
            ("tokens", int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)),
            ("tokens:input", int(usage.get("input_tokens") or 0)),
            ("tokens:output", int(usage.get("output_tokens") or 0)),
            ("tokens:cached", int(details.get("cached_tokens") or 0)),
            ("tokens:reasoning", int(reasoning.get("reasoning_tokens") or 0)),
            (
                f"tokens:model:{model}",
                int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0),
            ),
        ):
            if amount:
                self._add(name, day, amount)
        self._rows("DELETE FROM counters WHERE substr(bucket, 1, 10) < ?", _day(time.time() - 40 * 86400))
        return self.used("tokens")

    def usage(self, days: int) -> str:
        """Each day's counters, newest first."""
        cutoff = _day(time.time() - max(0, days - 1) * 86400)
        rows = self._rows(
            "SELECT name, bucket, used FROM counters WHERE length(bucket)=10 AND bucket>=? ORDER BY bucket DESC",
            cutoff,
        )
        found: dict[str, dict[str, float]] = {}
        for row in rows:
            found.setdefault(str(row["bucket"]), {})[str(row["name"])] = float(cast(float, row["used"]))
        return encode(found)

    # Approvals: held for a person's own reply. Only a hash of the code is kept here.

    def hold(
        self, conversation: str, code_hash: str, tool: str, arguments: str, reason: str, seconds: float
    ) -> str:
        now = time.time()
        self._rows(
            "UPDATE approvals SET status='expired' WHERE status='waiting' AND expires_at<=?",
            now,
        )
        waiting = self._rows(
            "SELECT id FROM approvals WHERE conversation=? AND status='waiting' ORDER BY id", conversation
        )
        for row in waiting[: max(0, len(waiting) - MAX_WAITING + 1)]:
            self._rows("UPDATE approvals SET status='replaced' WHERE id=?", row["id"])
        self._rows(
            "INSERT INTO approvals (conversation, code_hash, tool, arguments, reason, status, held_at, expires_at) "
            "VALUES(?,?,?,?,?,'waiting',?,?)",
            conversation,
            code_hash,
            tool,
            arguments,
            reason,
            now,
            now + seconds,
        )
        return encode({"held": True})

    def claim_approval(self, conversation: str, code_hash: str, decline: bool) -> str:
        """Use a code, once. Whatever happens next, the same code can never run anything again."""
        row = self._one(
            "SELECT id, tool, arguments, reason, expires_at FROM approvals "
            "WHERE conversation=? AND code_hash=? AND status='waiting' ORDER BY id DESC LIMIT 1",
            conversation,
            code_hash,
        )
        if not row:
            return encode({"found": False})
        expired = float(cast(float, row["expires_at"])) <= time.time()
        status = "expired" if expired else "declined" if decline else "approved"
        self._rows("UPDATE approvals SET status=? WHERE id=?", status, row["id"])
        return encode({"found": True, "status": status, "tool": row["tool"], "arguments": row["arguments"]})

    def waiting_approvals(self, conversation: str) -> str:
        return encode(
            self._rows(
                "SELECT tool, reason, expires_at FROM approvals WHERE conversation=? AND status='waiting' "
                "AND expires_at>? ORDER BY id",
                conversation,
                time.time(),
            )
        )

    # The audit chain: what was done and how it went, never the words of a request.

    def audit(self, entry_json: str) -> str:
        entry = encode({**json.loads(entry_json), "at": time.time()})
        last = self._one("SELECT hash FROM audit ORDER BY seq DESC LIMIT 1")
        previous = str(last["hash"]) if last else ""
        self._rows(
            "INSERT INTO audit (entry, previous, hash) VALUES(?,?,?)",
            entry,
            previous,
            _chain(previous, entry),
        )
        return encode({"recorded": True})

    def read_audit(self, limit: int) -> str:
        rows = self._rows("SELECT entry FROM audit ORDER BY seq DESC LIMIT ?", limit)
        return encode([json.loads(str(row["entry"])) for row in rows[::-1]])

    def verify_audit(self) -> str:
        """Walk the whole chain. A row changed or removed after the fact breaks it."""
        previous, count = "", 0
        for row in self._rows("SELECT seq, entry, previous, hash FROM audit ORDER BY seq"):
            if row["previous"] != previous or row["hash"] != _chain(previous, str(row["entry"])):
                return encode({"ok": False, "rows": count, "broken_at": row["seq"]})
            previous, count = str(row["hash"]), count + 1
        return encode({"ok": True, "rows": count, "broken_at": None})

    # Campaigns it watches: each ending reported exactly once.

    def watch(self, conversation: str, campaign: str, mission: int) -> str:
        self._rows(
            "INSERT INTO watches (campaign, conversation, mission, state) VALUES(?,?,?, 'watching') "
            "ON CONFLICT(campaign) DO NOTHING",
            campaign,
            conversation,
            mission if mission > 0 else None,
        )
        if mission > 0:
            row = self._one("SELECT campaigns FROM missions WHERE id=?", mission)
            if row:
                campaigns = [*json.loads(str(row["campaigns"])), campaign]
                self._rows("UPDATE missions SET campaigns=? WHERE id=?", encode(campaigns), mission)
        return encode({"watching": campaign})

    def watching(self, conversation: str) -> str:
        return encode(
            self._rows(
                "SELECT campaign, mission, state FROM watches WHERE conversation=? AND reported=0",
                conversation,
            )
        )

    def set_watch_state(self, campaign: str, state: str) -> str:
        self._rows("UPDATE watches SET state=? WHERE campaign=?", state, campaign)
        return encode({"state": state})

    def report_ended(self, campaign: str, text: str) -> str:
        """Queue the campaign's ending, the first time only, in one commit with marking it reported."""
        row = self._one("SELECT conversation, mission, reported FROM watches WHERE campaign=?", campaign)
        if not row or row["reported"]:
            return encode({"first": False, "mission": None})
        self._rows("UPDATE watches SET reported=1 WHERE campaign=?", campaign)
        if text:
            self._post(str(row["conversation"]), "hephaestus", text, delivered=False)
        return encode({"first": True, "mission": row["mission"]})

    def forget_watch(self, campaign: str) -> str:
        self._rows("UPDATE watches SET reported=1 WHERE campaign=?", campaign)
        return encode({"forgotten": campaign})

    def was_started(self, campaign: str) -> bool:
        return self._one("SELECT campaign FROM watches WHERE campaign=?", campaign) is not None

    # The chat itself: an outbox, so a reply survives a crash between writing it and sending it.

    def _post(self, conversation: str, sender: str, text: str, delivered: bool) -> int:
        self._rows(
            "INSERT INTO messages (conversation, sender, text, at, delivered) VALUES(?,?,?,?,?)",
            conversation,
            sender,
            text,
            time.time(),
            int(delivered),
        )
        posted = int(cast(int, self._rows("SELECT last_insert_rowid() AS id")[0]["id"]))
        self._rows(
            "DELETE FROM messages WHERE conversation=? AND "
            "id NOT IN (SELECT id FROM messages WHERE conversation=? ORDER BY id DESC LIMIT ?)",
            conversation,
            conversation,
            KEEP_MESSAGES,
        )
        return posted

    def post(self, conversation: str, sender: str, text: str) -> int:
        return self._post(conversation, sender, text, delivered=False)

    def undelivered(self, conversation: str) -> str:
        return encode(
            self._rows(
                "SELECT id, text FROM messages WHERE conversation=? AND delivered=0 AND attempts<? ORDER BY id",
                conversation,
                MAX_DELIVERY_ATTEMPTS,
            )
        )

    def mark_delivered(self, message_id: int, delivered: bool) -> str:
        if delivered:
            self._rows("UPDATE messages SET delivered=1 WHERE id=?", message_id)
        else:
            self._rows("UPDATE messages SET attempts=attempts+1 WHERE id=?", message_id)
        return encode({"recorded": True})

    def transcript(self, conversation: str, limit: int) -> str:
        rows = self._rows(
            "SELECT id, sender, text, at FROM messages WHERE conversation=? ORDER BY id DESC LIMIT ?",
            conversation,
            limit,
        )
        return encode(rows[::-1])

    # Long-term notes the model chooses to keep.

    def remember(self, text: str) -> int:
        count = self._rows("SELECT count(*) AS n FROM notes")[0]["n"]
        if cast(int, count) >= MAX_NOTES:
            raise ValueError(f"I already hold {MAX_NOTES} notes. Forget one first.")
        self._rows("INSERT INTO notes (text, at) VALUES(?,?)", text, time.time())
        return int(cast(int, self._rows("SELECT last_insert_rowid() AS id")[0]["id"]))

    def forget(self, note_id: int) -> str:
        self._rows("DELETE FROM notes WHERE id=?", note_id)
        return encode({"forgotten": note_id})

    def notes(self) -> str:
        return encode(self._rows("SELECT id, text FROM notes ORDER BY id"))

    # Missions: a goal it keeps working on between messages, within limits set when it began.

    def begin_mission(self, conversation: str, goal: str, max_campaigns: int, max_wakes: int) -> str:
        active = self._one(
            "SELECT id FROM missions WHERE conversation=? AND status='active' LIMIT 1", conversation
        )
        if active:
            raise ValueError("A mission is already active here. End it first.")
        self._rows(
            "INSERT INTO missions (conversation, goal, status, max_campaigns, max_wakes, began_at) "
            "VALUES(?,?,'active',?,?,?)",
            conversation,
            goal,
            max_campaigns,
            max_wakes,
            time.time(),
        )
        return self._mission(int(cast(int, self._rows("SELECT last_insert_rowid() AS id")[0]["id"])))

    def _mission(self, mission_id: int) -> str:
        row = self._one(
            "SELECT id, goal, status, max_campaigns, max_wakes, campaigns, wakes, outcome, began_at, ended_at "
            "FROM missions WHERE id=?",
            mission_id,
        )
        return encode({**row, "campaigns": json.loads(str(row["campaigns"]))} if row else None)

    def active_mission(self, conversation: str) -> str:
        row = self._one(
            "SELECT id FROM missions WHERE conversation=? AND status='active' LIMIT 1", conversation
        )
        return self._mission(int(cast(int, row["id"]))) if row else encode(None)

    def missions(self, conversation: str, limit: int) -> str:
        rows = self._rows(
            "SELECT id FROM missions WHERE conversation=? ORDER BY id DESC LIMIT ?", conversation, limit
        )
        return encode([json.loads(self._mission(int(cast(int, row["id"])))) for row in rows])

    def claim_wake(self, mission_id: int, event: str, gap_seconds: float) -> str:
        """May it wake for this event? Each event wakes it once, and a mission's wakes are limited."""
        row = self._one(
            "SELECT status, wakes, max_wakes, seen, last_wake FROM missions WHERE id=?", mission_id
        )
        if not row or row["status"] != "active":
            return encode({"wake": False, "why": "not active"})
        seen = json.loads(str(row["seen"]))
        if event in seen:
            return encode({"wake": False, "why": "seen"})
        if cast(int, row["wakes"]) >= cast(int, row["max_wakes"]):
            self._rows(
                "UPDATE missions SET status='paused', outcome=?, ended_at=? WHERE id=?",
                "Reached its limit of wake-ups.",
                time.time(),
                mission_id,
            )
            return encode({"wake": False, "why": "limit"})
        if time.time() - float(cast(float, row["last_wake"])) < gap_seconds:
            return encode({"wake": False, "why": "too soon"})
        self._rows(
            "UPDATE missions SET wakes=wakes+1, last_wake=?, seen=? WHERE id=?",
            time.time(),
            encode([*seen, event][-200:]),
            mission_id,
        )
        return encode({"wake": True, "why": None})

    def end_mission(self, mission_id: int, status: str, outcome: str) -> str:
        self._rows(
            "UPDATE missions SET status=?, outcome=?, ended_at=? WHERE id=? AND status IN ('active', 'paused')",
            status,
            outcome,
            time.time(),
            mission_id,
        )
        return self._mission(mission_id)

    # Check-ins it schedules for itself.

    def schedule(self, conversation: str, note: str, seconds: float) -> int:
        count = self._rows("SELECT count(*) AS n FROM checks WHERE fired=0")[0]["n"]
        if cast(int, count) >= MAX_CHECKS:
            raise ValueError(f"I already have {MAX_CHECKS} check-ins scheduled.")
        self._rows(
            "INSERT INTO checks (conversation, note, due_at) VALUES(?,?,?)",
            conversation,
            note,
            time.time() + seconds,
        )
        return int(cast(int, self._rows("SELECT last_insert_rowid() AS id")[0]["id"]))

    def claim_due(self, conversation: str) -> str:
        """Check-ins that have come due, each handed out once."""
        rows = self._rows(
            "SELECT id, note FROM checks WHERE conversation=? AND fired=0 AND due_at<=? ORDER BY due_at",
            conversation,
            time.time(),
        )
        for row in rows:
            self._rows("UPDATE checks SET fired=1 WHERE id=?", row["id"])
        return encode(rows)

    def scheduled(self, conversation: str) -> str:
        return encode(
            self._rows(
                "SELECT id, note, due_at FROM checks WHERE conversation=? AND fired=0 ORDER BY due_at",
                conversation,
            )
        )

    # Small facts that outlive any one chat, such as which VM is its own.

    def set_fact(self, name: str, value: str) -> str:
        if value:
            self._rows(
                "INSERT INTO facts VALUES(?,?) ON CONFLICT(name) DO UPDATE SET value=excluded.value",
                name,
                value,
            )
        else:
            self._rows("DELETE FROM facts WHERE name=?", name)
        return encode({"set": name})

    def get_fact(self, name: str) -> str:
        row = self._one("SELECT value FROM facts WHERE name=?", name)
        return str(row["value"]) if row else ""
