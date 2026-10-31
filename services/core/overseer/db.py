"""Local overseer storage: one SQLite file, separate from the campaign store."""

import json
import sqlite3
import threading

from services.core import store

LOCK = threading.RLock()
SCHEMA = """
create table if not exists traces (
  key text primary key,
  source text not null,
  trace_id text not null,
  pack_id text not null,
  received_at text not null,
  model_version text,
  latency_ms real,
  completion_tokens integer,
  finish_reason text,
  sealed_overlap integer not null default 0,
  payload text not null,
  checked integer not null default 0,
  sampled integer not null default 0,
  suspicious integer not null default 0,
  hard integer not null default 0
);
create index if not exists traces_pack on traces (pack_id, received_at);
create table if not exists signals (
  key text not null, kind text not null, evidence text not null, primary key (key, kind)
);
create table if not exists judgments (
  key text primary key, cache_key text not null, verdict text not null, result text not null,
  usd real not null, model text not null, created_at text not null
);
create table if not exists judge_cache (cache_key text primary key, result text not null);
create table if not exists spend (day text primary key, usd real not null, calls integer not null);
create table if not exists labels (
  key text primary key, answer text not null, reviewer text not null, created_at text not null
);
create table if not exists alerts (
  id text primary key, day text not null, pack_id text not null, kind text not null,
  detail text not null, created_at text not null
);
create table if not exists connections (
  id text primary key,
  name text not null,
  pack_id text not null,
  kind text not null,
  provider text not null,
  config text not null,
  status text not null,
  detail text not null default '',
  vm_id text,
  token_id text,
  secret_names text,
  cursor text,
  agent text,
  restarts integer not null default 0,
  daily_cap integer not null,
  data_terms text not null,
  last_seen_at text,
  created_at text not null,
  updated_at text not null
);
"""


def connect():
    path = store.DATA / "overseer.db"
    db = sqlite3.connect(path, timeout=30, isolation_level=None)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    return db


def rows(sql, args=()):
    with LOCK:
        db = connect()
        try:
            return [dict(r) for r in db.execute(sql, args).fetchall()]
        finally:
            db.close()


def execute(statements):
    """Run (sql, args) pairs in one immediate transaction."""
    with LOCK:
        db = connect()
        try:
            db.execute("begin immediate")
            for sql, args in statements:
                db.execute(sql, args)
            db.execute("commit")
        except Exception:
            db.execute("rollback")
            raise
        finally:
            db.close()


def payload(row):
    return json.loads(row["payload"])
