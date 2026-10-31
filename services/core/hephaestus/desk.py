"""The private conversation with Hephaestus in Agent chat: just the lab's owner and the bot.

The lab only answers on this machine, so whoever is typing here is its owner, and is an admin. The
conversation itself lives in Terse, so a restart of the lab loses neither the history nor a reply it
was in the middle of.
"""

import threading
import time
import uuid

from services.core import store
from services.core.hephaestus import brain, ledger
from services.core.hephaestus.service import LABEL, Service

KEEP = 200
MAX_REPLY = 6000
_lock = threading.Lock()
_service = {}


class Transport:
    """The page reads the conversation from Terse, so a queued reply has already arrived."""

    def send(self, _chat_id, text):
        return text.removeprefix(LABEL)

    def latest(self):
        return 0

    def read(self, _chat_id, _after):
        return []


def service():
    key = str(store.DATA)
    if key not in _service:
        _service[key] = Service(
            "agent-chat",
            messages=Transport(),
            surface="agent-chat",
            max_replies_per_hour=240,
            max_reply=MAX_REPLY,
        )
    return _service[key]


def history():
    book = ledger.Ledger("agent-chat")
    activity = book.activity()
    connected = brain.connected()
    return {
        "messages": [
            {
                "id": str(m["id"]),
                "sender": m["sender"],
                "text": m["text"],
                "at": m["at"],
            }
            for m in book.transcript(KEEP)
        ],
        "thinking": activity["working"],
        "activity": activity["tool"],
        "model": {
            "connected": connected,
            "provider": brain.provider() if connected else None,
            "name": brain.model_name() if connected else None,
            "tokens_today": int(book.used("tokens")),
        },
        "mission": book.active_mission(),
        "waiting": [a["tool"] for a in book.waiting_approvals()],
    }


def post(text):
    """Record what the owner said, as a turn. `respond` then does the slow part, off the request."""
    bot = service()
    turn = bot.ledger.begin_turn(uuid.uuid4().hex, "message", "admin", text, show=True)
    bot.active.add(turn["id"])
    return {"id": str(turn["id"]), "turn": turn["id"], "sender": "you", "text": text}


def respond(turn_id, text):
    bot = service()
    try:
        with _lock:
            said = brain.address(text)
            said = said if said is not None else text
            brain._run(bot.context, turn_id, "admin", said, bot.post or brain.call_model, bot._may_post())
            bot.deliver()
    except ledger.Unavailable:
        bot.recover_before = time.time()
    finally:
        bot.active.discard(turn_id)


def watch():
    """Report here when a campaign started here ends, and keep its missions moving."""
    with _lock:
        service().watch()
