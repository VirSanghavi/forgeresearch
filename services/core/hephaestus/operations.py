"""Which of the lab's operations Hephaestus may perform, in one table.

It can do what its owner can do in the command center, on the owner's request. Large or costly ones wait
for the owner's confirmation. A few it can't do at all, each because of what it would undermine.
"""

import re

ID = r"[A-Za-z0-9_.-]+"
# Words that can quote training data, questions, answers or production traffic.
SENSITIVE = {
    "answer",
    "answers",
    "context",
    "correction",
    "evidence",
    "examples",
    "expected",
    "failures",
    "feedback",
    "message",
    "messages",
    "output",
    "payload",
    "prediction",
    "predictions",
    "prompt",
    "question",
    "questions",
    "rationale",
    "reference",
    "rows",
    "samples",
    "text",
    "trace",
    "traces",
}
MAX_REPLY = 6000

REFUSED_GET = [
    (rf"^/campaigns/{ID}/artifacts/", "Artifacts can hold raw data, so open them in the command center."),
    (rf"^/campaigns/{ID}/trace$", "A campaign's trace holds raw data, so open it in the command center."),
    (r"^/hephaestus/", "That's this conversation."),
    (
        rf"^/packs/{ID}/teacher-data/{ID}$",
        "Teacher datasets hold raw data, so open them in the command center.",
    ),
]
REFUSED_POST = [
    (r"^/overseer/labels$", "Only a person may label an answer: labels become training data."),
    (r"^/overseer/traces$", "I can't submit production traffic: it would be fake evidence."),
    (r"^/overseer/connections$", "A new source needs a credential, so connect it in the form, not a chat."),
    (r"^/uploads", "Uploading data is done in the command center's setup form."),
    (r"^/packs(/preview)?$", "Creating a task from data is done in the command center's setup form."),
    (r"^/campaigns$", "Use start_training for that."),
    (rf"^/campaigns/{ID}/team/", "Use the tools for directing the team."),
    (rf"^/campaigns/{ID}/directives$", "Use direct_team for that."),
    (rf"^/campaigns/{ID}/agents/", "Use message_agent or ask_team for that."),
    (r"^/hephaestus/", "That's this conversation."),
]
ALLOWED_POST = [
    rf"^/campaigns/{ID}/probe$",
    r"^/overseer/cycle$",
    rf"^/overseer/connections/{ID}/restart$",
    rf"^/overseer/connections/{ID}/mapping$",
]
CONFIRMED_POST = {
    rf"^/campaigns/{ID}/cancel$": "This stops a running campaign.",
    rf"^/campaigns/{ID}/recording$": "This records this Mac's screen.",
    r"^/overseer/export$": "This sends reviewer labels to the lab as a production scan.",
    r"^/production/scans$": "This creates a production scan that campaigns can train on.",
    rf"^/packs/{ID}/teacher-data$": "This runs a teacher model, which can cost money.",
    rf"^/overseer/connections/{ID}/disconnect$": "This deletes a traffic connection and its credentials.",
}


def check(method, path):
    """Returns the reason this needs confirmation, or None. Raises ValueError when it's not allowed."""
    if not re.fullmatch(r"/[A-Za-z0-9_./?=&%-]*", path) or ".." in path:
        raise ValueError("That isn't a lab path.")
    route = path.split("?", 1)[0]
    if method == "GET":
        for pattern, why in REFUSED_GET:
            if re.search(pattern, route):
                raise ValueError(why)
        return None
    if method != "POST":
        raise ValueError("The lab's operations are GET and POST.")
    for pattern, why in REFUSED_POST:
        if re.search(pattern, route):
            raise ValueError(why)
    if any(re.search(pattern, route) for pattern in ALLOWED_POST):
        return None
    for pattern, why in CONFIRMED_POST.items():
        if re.search(pattern, route):
            return why
    raise ValueError("I don't have that operation.")


def redact(value):
    """Numbers, names and states, without anything that can quote data."""
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items() if k.lower() not in SENSITIVE}
    if isinstance(value, list):
        return [redact(item) for item in value[:50]]
    return value
