"""The Mac's Messages app: read one group chat from its database, and post to it.

Reading needs Full Disk Access for the process, and posting needs permission to control Messages.
macOS asks for or controls both, and only the Mac's owner can grant them. Only the configured chat is
ever queried, and this file stores nothing.
"""

import sqlite3
import subprocess
from pathlib import Path

DATABASE = Path.home() / "Library/Messages/chat.db"
# The text goes as an argument, never into the script, so a message can't become AppleScript.
SEND = (
    'on run argv\ntell application "Messages" to send (item 2 of argv) to chat id (item 1 of argv)\nend run'
)
CHATS = """
tell application "Messages"
  set found to ""
  repeat with c in chats
    set label to ""
    try
      set label to name of c as text
    end try
    set people to ""
    try
      set people to full name of participants of c as list
      set AppleScript's text item delimiters to ", "
      set people to people as text
    end try
    set found to found & (id of c) & tab & label & tab & people & linefeed
  end repeat
  return found
end tell
"""
READ = """
select m.ROWID as rowid, m.text as text, m.attributedBody as body, m.is_from_me as mine, h.id as sender
from message m
join chat_message_join j on j.message_id = m.ROWID
join chat c on c.ROWID = j.chat_id
left join handle h on h.ROWID = m.handle_id
where (c.chat_identifier = :chat or c.guid like '%;' || :chat) and m.ROWID > :after
  and coalesce(m.associated_message_type, 0) = 0
order by m.ROWID limit :limit
"""


class MessagesError(RuntimeError):
    """A failure whose message tells the Mac's owner what to do."""


def identifier(chat_id):
    """The last part of the id Messages uses: any;+;chat123 is chat123, and groups may use a UUID there."""
    return chat_id.rsplit(";", 1)[-1]


def decode_body(body):
    """The text of a message stored only as an attributed string, as recent macOS versions do."""
    if not body or b"NSString" not in body:
        return None
    content = body.split(b"NSString", 1)[1][5:]
    if not content:
        return None
    if content[0] == 0x81:
        length, start = int.from_bytes(content[1:3], "little"), 3
    elif content[0] == 0x82:
        length, start = int.from_bytes(content[1:5], "little"), 5
    else:
        length, start = content[0], 1
    return content[start : start + length].decode("utf-8", "replace")


def _connect(path):
    try:
        return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
    except sqlite3.Error as error:
        raise MessagesError(_denied()) from error


def _denied():
    return (
        "Can't read the Messages database. In System Settings → Privacy & Security → Full Disk Access, "
        "allow the app that runs Hephaestus, then start it again."
    )


def latest(path=DATABASE):
    """The newest message's position: where to start, so old messages are never acted on."""
    db = _connect(path)
    try:
        return db.execute("select coalesce(max(ROWID), 0) from message").fetchone()[0]
    except sqlite3.Error as error:
        raise MessagesError(_denied()) from error
    finally:
        db.close()


def read(chat_id, after, path=DATABASE, limit=50):
    """New messages in this one chat, without reactions, oldest first."""
    db = _connect(path)
    db.row_factory = sqlite3.Row
    try:
        rows = db.execute(READ, {"chat": identifier(chat_id), "after": after, "limit": limit}).fetchall()
    except sqlite3.Error as error:
        raise MessagesError(_denied()) from error
    finally:
        db.close()
    return [
        {
            "rowid": row["rowid"],
            "text": row["text"] or decode_body(row["body"]) or "",
            "mine": bool(row["mine"]),
            "sender": row["sender"],
        }
        for row in rows
    ]


def count(chat_id, path=DATABASE):
    """How many messages a chat has. Touches no message text: it proves access and nothing more."""
    db = _connect(path)
    try:
        return db.execute(
            "select count(*) from chat_message_join j join chat c on c.ROWID = j.chat_id"
            " where c.chat_identifier = :chat or c.guid like '%;' || :chat",
            {"chat": identifier(chat_id)},
        ).fetchone()[0]
    except sqlite3.Error as error:
        raise MessagesError(_denied()) from error
    finally:
        db.close()


def send(chat_id, text):
    done = subprocess.run(["osascript", "-e", SEND, chat_id, text], capture_output=True, timeout=30)
    if done.returncode:
        raise MessagesError("Messages would not send. Allow this app to control Messages when macOS asks.")


def chats():
    """Group chats, by id, name and people, from the Messages app. Reads no message text."""
    done = subprocess.run(["osascript", "-e", CHATS], capture_output=True, text=True, timeout=30)
    if done.returncode:
        raise MessagesError("Can't list chats. Allow this app to control Messages when macOS asks.")
    found = []
    for line in done.stdout.splitlines():
        chat_id, name, people = (line.split("\t") + ["", ""])[:3]
        if ";+;" in chat_id:
            found.append({"id": chat_id, "name": "" if name == "missing value" else name, "people": people})
    return found
