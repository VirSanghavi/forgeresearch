"""The running bot: reads new messages in its one chat, answers, reports on its campaigns and keeps its
missions moving. Where it has read to, what it's doing and what it owes the chat all live in Terse, so a
crash at any point loses nothing and repeats nothing."""

import json
import os
import time
from types import SimpleNamespace

from services.core.hephaestus import brain, imessage, ledger, tools

LABEL = "⚒️ Hephaestus\n"
TICK_SECONDS = 5
WATCH_SECONDS = 30
MAX_REPLIES_PER_HOUR = 30
MAX_FAILURES = 60
ECHO_SECONDS = 60
ENDED = {
    "completed": "finished",
    "failed": "failed",
    "cancelled": "was stopped",
    "interrupted": "was interrupted",
}


def _handle(value):
    value = value.strip().lower()
    return value.replace(" ", "").replace("-", "").replace("(", "").replace(")", "")


def operators():
    return {_handle(h) for h in os.environ.get("FORGE_HEPHAESTUS_OPERATORS", "").split(",") if h.strip()}


def owners():
    return {_handle(h) for h in os.environ.get("FORGE_HEPHAESTUS_OWNER", "").split(",") if h.strip()}


def role_of(message):
    """Apple's own sender identity decides this. Anything said in a message has no bearing on it."""
    sender = _handle(message["sender"]) if message["sender"] else None
    if message["mine"] or sender in owners():
        return "admin"
    return "operator" if sender in operators() else "viewer"


class Service:
    def __init__(
        self,
        chat_id,
        lab=None,
        messages=imessage,
        post=None,
        clock=time.monotonic,
        surface="imessage",
        max_replies_per_hour=MAX_REPLIES_PER_HOUR,
        max_reply=brain.MAX_REPLY,
    ):
        self.chat_id, self.messages, self.clock = chat_id, messages, clock
        self.post = post
        # Each chat has its own conversation, so one group's words never reach another's model calls.
        self.ledger = ledger.Ledger("agent-chat" if surface == "agent-chat" else f"imessage:{chat_id}")
        self.context = SimpleNamespace(
            lab=lab or tools.Lab(),
            surface=surface,
            ledger=self.ledger,
            turn=None,
            held=[],
            max_reply=max_reply,
        )
        self.max_replies_per_hour = max_replies_per_hour
        self.recent_texts = {}
        self.last_watch = None
        # Open turns from before this moment were interrupted. Later ones are this process's own.
        self.recover_before = time.time()
        self.active = set()

    def _model(self):
        return {"post": self.post} if self.post else {}

    def _may_post(self):
        allowed = self.ledger.take(
            f"replies:{self.ledger.conversation}", self.max_replies_per_hour, window="hour"
        )
        if not allowed:
            print("Reply limit reached; not posting.", flush=True)
        return allowed

    def say(self, text):
        """Post to the chat, under the label, within a limit that stops any runaway loop."""
        if self._may_post():
            self.ledger.post(text[: self.context.max_reply])
            self.deliver()

    def deliver(self):
        """Send what's owed to the chat. It's queued in Terse first, so a crash here sends it on restart,
        and a send that fails is tried again a few times."""
        for message in self.ledger.undelivered():
            try:
                self.messages.send(self.chat_id, LABEL + message["text"])
            except (imessage.MessagesError, OSError) as error:
                print(f"Could not post: {error}", flush=True)
                self.ledger.mark_delivered(message["id"], delivered=False)
                continue
            self.ledger.mark_delivered(message["id"])

    def handle(self, message):
        said = brain.address(message["text"])
        if said is None or self._is_echo(said):
            self.ledger.advance(message["rowid"])
            return
        brain.answer(
            self.context,
            role_of(message),
            said,
            source=message["rowid"],
            position=message["rowid"],
            announce=self._may_post(),
            **self._model(),
        )
        self.deliver()

    def _is_echo(self, said):
        """Messaging yourself stores each message twice, as sent and as received."""
        if ";-;" not in self.chat_id:
            return False
        now = self.clock()
        self.recent_texts = {t: at for t, at in self.recent_texts.items() if now - at < ECHO_SECONDS}
        seen = said in self.recent_texts
        self.recent_texts[said] = now
        return seen

    def read_new(self):
        cursor = self.ledger.cursor()
        if cursor < 0:
            # Start from now, in a new chat too: nothing said before Hephaestus joined is acted on.
            self.ledger.advance(self.messages.latest())
            return
        for message in self.messages.read(self.chat_id, cursor):
            try:
                self.handle(message)
            except ledger.Unavailable:
                # Nothing past this message was read, so it's picked up when Terse is back.
                self.recover_before = time.time()
                raise
            except Exception as error:
                print(f"Could not handle a message: {type(error).__name__}", flush=True)
                self.ledger.advance(message["rowid"])

    def recover(self):
        """Pick up turns a crash interrupted, and send anything it wrote but never sent."""
        if self.recover_before is None:
            return
        for turn in self.ledger.open_turns(self.recover_before):
            if turn["id"] not in self.active:
                brain.recover(self.context, turn, **self._model())
        self.recover_before = None
        self.deliver()

    def watch(self):
        """Report when a campaign it started ends, guard this Mac's memory, and wake for its missions."""
        self.recover()
        watching = self.ledger.watching()
        if watching:
            try:
                campaigns = {c["id"]: c for c in self.context.lab.call("GET", "/campaigns")}
            except tools.ToolError:
                campaigns = None  # The lab is down. Campaign state is durable, so this catches up.
            for row in watching if campaigns is not None else []:
                self._check(row, campaigns.get(row["campaign"]))
        self.deliver()
        self._missions()
        self._check_ins()
        self.deliver()

    def _check(self, row, campaign):
        cid = row["campaign"]
        if campaign is None:
            self.ledger.forget_watch(cid)
        elif campaign["status"] in tools.TERMINAL:
            # In one commit with marking it reported: a crash can't lose this or send it twice.
            self.ledger.report_ended(cid, self._ending(campaign))
        elif tools.runs_here():
            self._guard_memory(row)

    def _guard_memory(self, row):
        cid = row["campaign"]
        free = tools.free_memory_percent()
        floor = int(os.environ.get("FORGE_HEPHAESTUS_STOP_FREE_MEMORY") or 10)
        if free is not None and free < floor and row["state"] != "stopping":
            self.context.lab.call("POST", f"/campaigns/{cid}/cancel")
            self.ledger.set_watch_state(cid, "stopping")
            self.say(f"This Mac has only {free}% of its memory free, so I'm stopping {cid}.")

    def _missions(self):
        mission = self.ledger.active_mission()
        if not mission:
            return
        for event, why in self._mission_events(mission):
            gap = float(os.environ.get("FORGE_HEPHAESTUS_WAKE_SECONDS") or 60)
            claimed = self.ledger.claim_wake(mission["id"], event, 0 if event == "started" else gap)
            if claimed["why"] == "limit":
                self.say(
                    f"I've paused the mission after {mission['max_wakes']} wake-ups. Tell me to carry on."
                )
            if not claimed["wake"]:
                continue
            goal = json.dumps(mission["goal"])
            brain.wake(
                self.context,
                "admin",
                f"Mission {mission['id']}, goal in the owner's words: {goal}. {why}",
                f"mission:{mission['id']}:{event}",
                **self._model(),
            )
            self.deliver()
            return  # One a tick. Anything else still counts next time, with what this one did in view.

    def _mission_events(self, mission):
        """What a mission should wake for: its start, each experiment that ends and each campaign that ends."""
        yield "started", "The owner just approved this mission, so begin."
        for cid in mission["campaigns"]:
            try:
                campaign = self.context.lab.call("GET", f"/campaigns/{cid}")
            except tools.ToolError:
                continue
            for experiment in campaign.get("experiments") or []:
                if experiment.get("status") in ("completed", "rejected", "failed"):
                    yield (
                        f"{cid}:{experiment.get('id')}",
                        f"Experiment {experiment.get('id')} of {cid} ended ({experiment.get('status')}).",
                    )
            if campaign["status"] in tools.TERMINAL:
                yield f"{cid}:ended", f"Campaign {cid} ended ({campaign['status']})."

    def _check_ins(self):
        for check in self.ledger.claim_due():
            brain.wake(
                self.context,
                "admin",
                f"A check-in you scheduled came due: {json.dumps(check['note'])}.",
                f"check:{check['id']}",
                **self._model(),
            )

    @staticmethod
    def _ending(campaign):
        brief = tools._brief(campaign)
        line = f"{brief['id']} {ENDED[campaign['status']]}."
        if brief["error"] and campaign["status"] == "failed":
            return f"{line} {brief['error']}"
        claim = brief["improvement_claim"]
        if claim == "validated":
            return f"{line} Validated improvement of {brief['improvement_points']} points."
        if claim == "uncertain":
            return f"{line} The score rose {brief['improvement_points']} points, but that's within noise."
        if claim == "none":
            return f"{line} No improvement, so nothing is claimed."
        return line

    def tick(self):
        self.recover()
        self.read_new()
        now = self.clock()
        if self.last_watch is None or now - self.last_watch >= WATCH_SECONDS:
            self.last_watch = now
            self.watch()

    def run(self, sleep=time.sleep):
        """Keep going through a passing failure, since Messages can briefly hold its database. Terse being
        away is only a wait, since nothing is lost. Any other failure that lasts stops it, such as access
        being turned off."""
        failures, waiting = 0, False
        while True:
            try:
                self.tick()
                failures, waiting = 0, False
            except ledger.Unavailable:
                if not waiting:
                    print(
                        "Terse isn't answering. Waiting for it, and doing nothing until it's back.",
                        flush=True,
                    )
                waiting = True
            except Exception as error:
                failures += 1
                if failures == 1:
                    print(f"Can't read new messages, retrying: {error}", flush=True)
                if failures >= MAX_FAILURES:
                    print("Stopping: this has failed for 5 minutes.", flush=True)
                    return 2
            sleep(TICK_SECONDS)
