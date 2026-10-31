"""Transactional read model and durable event journal. Terse owns critical state."""

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
DATA.mkdir(exist_ok=True)
ARTIFACTS.mkdir(exist_ok=True)
LOCK = threading.RLock()


def now():
    return datetime.now(timezone.utc).isoformat()


def db():
    c = sqlite3.connect(DATA / "forge.db", timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("CREATE TABLE IF NOT EXISTS campaigns(id TEXT PRIMARY KEY, snapshot TEXT NOT NULL)")
    c.execute(
        "CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY,campaign_id TEXT NOT NULL, timestamp TEXT NOT NULL,payload TEXT NOT NULL)"
    )
    c.execute("CREATE INDEX IF NOT EXISTS events_campaign ON events(campaign_id,timestamp)")
    return c


def save(snapshot):
    with LOCK, db() as c:
        snapshot["updated_at"] = now()
        c.execute(
            "INSERT INTO campaigns VALUES(?,?) ON CONFLICT(id) DO UPDATE SET snapshot=excluded.snapshot",
            (snapshot["id"], json.dumps(snapshot)),
        )


def get(campaign_id):
    with db() as c:
        r = c.execute("SELECT snapshot FROM campaigns WHERE id=?", (campaign_id,)).fetchone()
        return json.loads(r[0]) if r else None


def patch(campaign_id, update):
    """Serialize read-modify-write across API, recorder and worker processes."""
    with db() as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute("SELECT snapshot FROM campaigns WHERE id=?", (campaign_id,)).fetchone()
        if not row:
            raise KeyError(campaign_id)
        snapshot = json.loads(row[0])
        update(snapshot)
        snapshot["updated_at"] = now()
        connection.execute("UPDATE campaigns SET snapshot=? WHERE id=?", (json.dumps(snapshot), campaign_id))
    return snapshot


def campaigns():
    with db() as c:
        rows = c.execute("SELECT snapshot FROM campaigns ORDER BY rowid DESC").fetchall()
        result = []
        for row in rows:
            try:
                snapshot = json.loads(row[0])
                if (
                    isinstance(snapshot, dict)
                    and isinstance(snapshot.get("id"), str)
                    and isinstance(snapshot.get("status"), str)
                ):
                    result.append(snapshot)
            except (ValueError, TypeError):
                continue
        return result


def event(campaign_id, kind, text, agent="director", **details):
    e = {
        "id": uuid.uuid4().hex,
        "campaign_id": campaign_id,
        "timestamp": now(),
        "kind": kind,
        "text": text,
        "agent": agent,
        **details,
    }
    with LOCK, db() as c:
        c.execute("INSERT INTO events VALUES(?,?,?,?)", (e["id"], campaign_id, e["timestamp"], json.dumps(e)))
        folder = ARTIFACTS / campaign_id
        folder.mkdir(exist_ok=True)
        with (folder / "events.jsonl").open("a") as f:
            f.write(json.dumps(e) + "\n")
            f.flush()
            os.fsync(f.fileno())
    return e


def events(campaign_id, after=None):
    with db() as c:
        rows = c.execute(
            "SELECT payload FROM events WHERE campaign_id=? ORDER BY timestamp,rowid", (campaign_id,)
        ).fetchall()
    result = []
    for row in rows:
        try:
            e = json.loads(row[0])
            if after is None or e["timestamp"] > after:
                result.append(e)
        except (ValueError, KeyError, TypeError):
            continue
    return result
