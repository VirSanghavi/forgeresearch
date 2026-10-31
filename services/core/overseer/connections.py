"""Connections: a customer links a traffic source once, and the overseer's agent takes it from there.

The lab issues the agent a scoped ingest token, starts its VM, and supervises it on every overseer
cycle: it restarts a stalled agent, replaces a lost VM from the last reported cursor, gives up after a
bounded number of attempts rather than spend without limit, and raises an alert when traffic stops
flowing. Credentials are never returned by the API or written to the database.
"""

import json
import os
import threading
import uuid
from datetime import UTC, datetime
from urllib.parse import urlencode, urlsplit

from services.core import packs, store
from services.core.overseer import agent, db, pull, vms

MAX_RESTARTS = 5
STALE_SECONDS = 180
FIELDS = {*agent.CANDIDATES, "latency_seconds"}
# What only a person can fix. Lost contact is raised separately, by the supervisor.
ATTENTION = {"needs_mapping", "source_auth_failed", "sink_auth_failed", "capped"}
# What may clear by itself, so it is only raised once it has lasted.
TRANSIENT = {"source_error", "sink_unreachable"}
SILENCE_SECONDS = 3600
LANGFUSE_MAPPING = {
    "id": "id",
    "prompt": "input",
    "output": "output",
    "timestamp": "startTime",
    "model_version": "model",
    "latency_seconds": "latency",
    "prompt_tokens": "usageDetails.input",
    "completion_tokens": "usageDetails.output",
    "deployment": "environment",
}
_supervising = threading.Lock()


def max_vms():
    return int(os.environ.get("FORGE_OVERSEER_MAX_VMS") or 3)


def _check_url(url, provider):
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.scheme not in {"http", "https"} or not host:
        raise ValueError("The source URL must start with https://")
    if parts.username or parts.password:
        raise ValueError("Put credentials in the secret field, not the URL")
    loopback = host in {"localhost", "127.0.0.1", "::1"}
    if parts.scheme != "https" and (provider.needs_https or not loopback):
        raise ValueError("The source URL must use https")
    if loopback and provider.needs_https:
        raise ValueError(
            "A VM can't reach this machine or a private network. Run the agent inside that network instead"
        )


def _check_mapping(mapping):
    for field, path in mapping.items():
        if field not in FIELDS:
            raise ValueError(f"Unknown mapping field {field}")
        if not isinstance(path, str) or not path.strip() or len(path) > 200:
            raise ValueError(f"The mapping for {field} must be a field path")


def source_config(request, provider):
    """The agent's view of a source, from what the customer filled in. Holds no credential."""
    auth = {"type": request.get("auth_type") or "none"}
    if request["kind"] == "langfuse":
        if not request.get("username"):
            raise ValueError("Langfuse needs its public key, as the username")
        base = (request.get("url") or "https://cloud.langfuse.com").rstrip("/")
        query = urlencode(
            {"type": "GENERATION", "fields": "core,basic,io,model,metrics,usage", "limit": 200}, safe=","
        )
        config = {
            "url": f"{base}/api/public/v2/observations?{query}",
            "cursor_param": "fromStartTime",
            "cursor_field": "startTime",
            # Only new traffic unless told otherwise: an old project could spend the daily cap at once.
            "start_cursor": request.get("start_cursor") or store.now(),
            "page_cursor_param": "cursor",
            "page_cursor_path": "meta.cursor",
            # A generation still in flight has no output yet, and is picked up on a later pass.
            "cursor_overlap_seconds": 300,
            "records_path": "data",
            "mapping": {**LANGFUSE_MAPPING, **(request.get("mapping") or {})},
        }
        auth = {"type": "basic", "username": request["username"]}
    else:
        if not request.get("url"):
            raise ValueError("A source URL is required")
        config = {
            "url": request["url"],
            "cursor_param": request.get("cursor_param"),
            "cursor_field": request.get("cursor_field"),
            "start_cursor": request.get("start_cursor"),
            "page_cursor_param": request.get("page_cursor_param"),
            "page_cursor_path": request.get("page_cursor_path"),
            "records_path": request.get("records_path"),
            "mapping": request.get("mapping") or {},
        }
        if "{cursor}" in request["url"] and not request.get("start_cursor"):
            raise ValueError("A URL with {cursor} needs a value to start from")
        if bool(config["page_cursor_param"]) != bool(config["page_cursor_path"]):
            raise ValueError("Paging needs both the parameter and where the reply holds the next cursor")
        if auth["type"] == "api-key":
            if not request.get("header"):
                raise ValueError("An API-key source needs the header name")
            auth.update(header=request["header"], prefix=request.get("prefix") or "")
        elif auth["type"] == "basic":
            auth["username"] = request.get("username") or ""
    if auth["type"] != "none" and not request.get("secret"):
        raise ValueError("This source needs its read-only credential")
    _check_url(config["url"], provider)
    _check_mapping(config["mapping"])
    return {**config, "auth": auth, "poll_seconds": request.get("poll_seconds") or 60}


def _update(connection_id, **fields):
    fields["updated_at"] = store.now()
    names = ", ".join(f"{name} = ?" for name in fields)
    db.execute([(f"update connections set {names} where id = ?", (*fields.values(), connection_id))])


def _get(connection_id):
    found = db.rows("select * from connections where id = ?", (connection_id,))
    if not found:
        raise KeyError(connection_id)
    return found[0]


def public(row):
    """What the API returns. The URL is reduced to its host: a query string can carry a key."""
    config = json.loads(row["config"])
    state = json.loads(row["agent"] or "{}")
    return {
        "id": row["id"],
        "name": row["name"],
        "task_pack": row["pack_id"],
        "kind": row["kind"],
        "provider": row["provider"],
        "host": vms.host_of(config["url"]),
        "status": row["status"],
        "detail": row["detail"],
        "has_vm": bool(row["vm_id"]),
        "restarts": row["restarts"],
        "mapping": state.get("mapping") or config.get("mapping") or {},
        "fields_seen": state.get("fields_seen") or [],
        "counts": {k: state.get(k, 0) for k in ("fetched", "accepted", "duplicates", "rejected", "skipped")},
        "last_success_at": state.get("last_success_at"),
        "last_seen_at": row["last_seen_at"],
        "created_at": row["created_at"],
    }


def listing(pack_id=None):
    where, args = ("where pack_id = ?", (pack_id,)) if pack_id else ("", ())
    rows = db.rows(f"select * from connections {where} order by created_at desc", args)
    return {"connections": [public(row) for row in rows], **vms.describe()}


def create(request):
    """Record a connection. `start` then does the slow part, so the caller isn't held up by a VM boot."""
    pack = packs.get(request["task_pack"])
    name = request.get("provider") or vms.default()
    provider = vms.PROVIDERS.get(name)
    if provider is None:
        raise ValueError(f"Unknown provider {name}")
    ready, reason = provider.ready()
    if not ready:
        raise ValueError(reason)
    if name != "local":
        running = db.rows(
            "select count(*) as n from connections where provider != 'local'"
            " and (vm_id is not null or status in ('provisioning', 'restarting'))"
        )[0]["n"]
        if running >= max_vms():
            raise ValueError(f"At most {max_vms()} VMs (FORGE_OVERSEER_MAX_VMS): disconnect one first")
    config = source_config(request, provider)
    connection_id = f"conn-{uuid.uuid4().hex[:10]}"
    if name == "local":
        api = os.environ.get("FORGE_API_URL", "http://127.0.0.1:8000").rstrip("/")
        sink_url = f"{api}/overseer/traces"
    else:
        base = os.environ["SUPABASE_URL"].rstrip("/")
        sink_url = os.environ.get("FORGE_OVERSEER_INGEST_URL") or f"{base}/functions/v1/ingest"
        if not sink_url.startswith("https://"):
            raise ValueError("A VM ships traces over https: set FORGE_OVERSEER_INGEST_URL")
    config.update(source=connection_id, task_pack=pack.id, sink_url=sink_url)
    now = store.now()
    db.execute(
        [
            (
                "insert into connections (id, name, pack_id, kind, provider, config, status, detail, cursor,"
                " daily_cap, data_terms, created_at, updated_at) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    connection_id,
                    request["name"],
                    pack.id,
                    request["kind"],
                    name,
                    json.dumps(config),
                    "provisioning",
                    "",
                    config.get("start_cursor"),
                    request.get("daily_cap") or 10_000,
                    request["data_terms"],
                    now,
                    now,
                ),
            )
        ]
    )
    return public(_get(connection_id))


def _safe_message(error):
    # Provider messages can echo what was sent to them, so only our own wording is shown or stored.
    return str(error) if isinstance(error, vms.VMError) else f"The provider failed ({type(error).__name__})"


def start(connection_id, source_secret):
    """Issue the token, store the credentials with the provider and boot. Undoes it all on failure."""
    connection = _get(connection_id)
    config = json.loads(connection["config"])
    provider = vms.PROVIDERS[connection["provider"]]
    secrets = {"FORGE_SOURCE_SECRET": source_secret} if source_secret else {}
    try:
        if connection["provider"] != "local":
            issued = pull.create_token(connection_id, connection["pack_id"], connection["daily_cap"])
            _update(connection_id, token_id=issued["id"])
            secrets["FORGE_INGEST_TOKEN"] = issued["token"]
        names = provider.store_secrets(connection, config, secrets)
        _update(connection_id, secret_names=json.dumps(names))
        connection = _get(connection_id)
        vm_id = provider.provision(connection, config)
        _update(connection_id, vm_id=vm_id, status="starting", detail="")
    except Exception as error:
        _update(connection_id, status="failed", detail=_safe_message(error))
        try:
            _release(_get(connection_id), provider)
        except Exception as cleanup:
            _update(connection_id, status="disconnect_failed", detail=_safe_message(cleanup))


def _release(connection, provider):
    """Stop compute and remove credentials, token included. Raises if anything could not be undone."""
    provider.remove(connection)
    if connection["token_id"]:
        pull.revoke_token(connection["token_id"])
    _update(connection["id"], vm_id=None, secret_names=None, token_id=None)


def set_mapping(connection_id, mapping):
    """Tell a running agent which fields hold what; it applies this on its next wake."""
    _check_mapping(mapping)
    connection = _get(connection_id)
    config = json.loads(connection["config"])
    config["mapping"] = {**config.get("mapping", {}), **mapping}
    vms.PROVIDERS[connection["provider"]].write_config(connection, config)
    _update(connection_id, config=json.dumps(config), status="starting", detail="")
    return public(_get(connection_id))


def restart(connection_id):
    """Give a connection that was given up on another round of attempts."""
    connection = _get(connection_id)
    if not connection["secret_names"]:
        raise ValueError("Its credentials were removed when it failed: disconnect it and connect again")
    _update(connection_id, restarts=0, status="restarting", detail="")
    try:
        _supervise_one(_get(connection_id))
    except Exception as error:
        _update(connection_id, detail=_safe_message(error))
    return public(_get(connection_id))


def disconnect(connection_id):
    """Stop the VM, delete its credentials and revoke its token. Traces already received stay."""
    connection = _get(connection_id)
    provider = vms.PROVIDERS[connection["provider"]]
    try:
        _release(connection, provider)
    except Exception as error:
        # Never forget a connection that may still be running and billing.
        _update(connection_id, status="disconnect_failed", detail=_safe_message(error))
        raise vms.VMError("Could not fully disconnect. Nothing was forgotten: try again") from error
    db.execute([("delete from connections where id = ?", (connection_id,))])
    return {"id": connection_id, "disconnected": True}


def _alert(connection, reason):
    day = datetime.now(UTC).strftime("%Y-%m-%d")
    db.execute(
        [
            (
                "insert or ignore into alerts (id, day, pack_id, kind, detail, created_at) values (?, ?, ?, ?, ?, ?)",
                (
                    f"{day}:{connection['pack_id']}:connector_down:{connection['id']}:{reason[:40]}",
                    day,
                    connection["pack_id"],
                    "connector_down",
                    json.dumps({"connection": connection["name"], "reason": reason}),
                    store.now(),
                ),
            )
        ]
    )


def _age(moment):
    """Seconds since an ISO time, or None when there is no usable time."""
    try:
        return (datetime.now(UTC) - datetime.fromisoformat(moment)).total_seconds()
    except (TypeError, ValueError):
        return None


def _fresh(state):
    age = _age(state.get("heartbeat_at"))
    return age is not None and age < STALE_SECONDS


def _give_up(connection, provider, reason):
    # Compute stops here: repeating a failing boot is how an unattended agent runs up a bill.
    provider.halt(connection)
    _update(connection["id"], status="failed", detail=reason, vm_id=None)
    _alert(connection, reason)


def _replace_vm(connection, provider):
    config = json.loads(connection["config"])
    if connection["cursor"] is not None:
        config["start_cursor"] = connection["cursor"]
    vm_id = provider.provision(connection, config)
    _update(connection["id"], vm_id=vm_id, status="starting", detail="")


def _supervise_one(connection):
    provider = vms.PROVIDERS[connection["provider"]]
    state = provider.read(connection) if connection["vm_id"] else None
    if state and _fresh(state):
        changes = {
            "agent": json.dumps(state),
            "status": state.get("status", "starting"),
            "detail": state.get("detail", ""),
            "last_seen_at": store.now(),
            "restarts": 0,
        }
        if state.get("cursor") is not None:
            changes["cursor"] = str(state["cursor"])
        _update(connection["id"], **changes)
        status = state.get("status")
        silent = _age(state.get("last_success_at") or connection["created_at"])
        if status in ATTENTION or (status in TRANSIENT and silent is not None and silent > SILENCE_SECONDS):
            _alert(connection, state.get("detail") or status)
        return
    if connection["restarts"] >= MAX_RESTARTS:
        _give_up(connection, provider, f"The agent did not recover after {MAX_RESTARTS} attempts")
        return
    _update(
        connection["id"],
        restarts=connection["restarts"] + 1,
        status="restarting",
        detail="The agent stopped reporting" if state is not None else "The VM is gone; replacing it",
    )
    # Starting twice is harmless: the agent holds a lock and a second copy exits.
    if state is None:
        _replace_vm(_get(connection["id"]), provider)
    else:
        provider.ensure_running(connection)


def supervise():
    """One pass over every connection. One connection failing never holds up the others."""
    if not _supervising.acquire(blocking=False):
        return {"supervised": 0, "skipped": "Already running"}
    try:
        rows = db.rows(
            "select * from connections where status not in ('provisioning', 'failed', 'disconnect_failed')"
        )
        for connection in rows:
            try:
                _supervise_one(connection)
            except Exception as error:
                # The control plane being unreachable says nothing about the agent, so it costs no
                # attempt: the VM carries on without us.
                _update(connection["id"], detail=f"Could not check on the agent: {_safe_message(error)}")
                silent = _age(connection["last_seen_at"] or connection["created_at"])
                if silent is not None and silent > SILENCE_SECONDS:
                    _alert(connection, "The lab has lost contact with the agent")
        return {"supervised": len(rows), "skipped": None}
    finally:
        _supervising.release()
