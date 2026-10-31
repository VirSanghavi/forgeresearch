"""Pull traces a deployment posted to the Supabase buffer (edge function `ingest`).

The lab never exposes a port. It reads unpulled rows with the service key it already uses for the
evidence archive, stores them locally and marks them pulled.
"""

import os

import httpx

from services.core.overseer import traces

BATCH = 500


def configured():
    return bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_ROLE_KEY"))


def _client():
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return httpx.Client(
        base_url=os.environ["SUPABASE_URL"].rstrip("/") + "/rest/v1",
        headers={"apikey": key, "Authorization": f"Bearer {key}"},
        timeout=30,
    )


def pull(client=None):
    owned = client is None
    client = client or _client()
    total, rejected = 0, 0
    try:
        while True:
            reply = client.get(
                "/forge_traces",
                params={
                    "select": "id,source,pack_id,payload,received_at",
                    "pulled_at": "is.null",
                    "order": "id",
                    "limit": BATCH,
                },
            )
            reply.raise_for_status()
            rows = reply.json()
            if not rows:
                break
            groups = {}
            for row in rows:
                groups.setdefault((row["source"], row["pack_id"]), []).append(row)
            for (source, pack_id), group in groups.items():
                try:
                    result = traces.ingest(
                        [r["payload"] for r in group], source, pack_id, group[0]["received_at"]
                    )
                    rejected += len(result["rejected"])
                except KeyError:
                    rejected += len(group)  # Unknown task pack: kept in the buffer's history, not here.
            ids = ",".join(str(r["id"]) for r in rows)
            done = client.patch("/forge_traces", params={"id": f"in.({ids})"}, json={"pulled_at": "now()"})
            done.raise_for_status()
            total += len(rows)
            if len(rows) < BATCH:
                break
    finally:
        if owned:
            client.close()
    return {"pulled": total, "rejected": rejected}


def create_token(source, pack_id, daily_cap=10_000, client=None):
    """A new ingest token. Only its hash is stored; the token is shown once."""
    import hashlib
    import secrets
    import uuid

    token = "fo_" + secrets.token_urlsafe(32)
    row = {
        "id": f"tok-{uuid.uuid4().hex[:10]}",
        "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
        "source": source,
        "pack_id": pack_id,
        "daily_cap": daily_cap,
    }
    owned = client is None
    client = client or _client()
    try:
        client.post("/forge_ingest_tokens", json=row).raise_for_status()
    finally:
        if owned:
            client.close()
    return {"id": row["id"], "token": token, "source": source, "pack_id": pack_id, "daily_cap": daily_cap}


def revoke_token(token_id, client=None):
    owned = client is None
    client = client or _client()
    try:
        client.patch(
            "/forge_ingest_tokens", params={"id": f"eq.{token_id}"}, json={"revoked": True}
        ).raise_for_status()
    finally:
        if owned:
            client.close()
    return {"id": token_id, "revoked": True}
