import json
import os
import sys
import uuid
from contextlib import suppress
from pathlib import Path
import httpx
from services.core import store
from services.core.contracts import Message
from services.core.actor_client import get_actor


def actor(campaign_id):
    return get_actor("ResearchCampaignActor", campaign_id)


def governance(campaign_id):
    return get_actor("GovernanceActor", campaign_id)


def bridge_url():
    return os.getenv("FORGE_RELAY_BRIDGE", "http://127.0.0.1:7529")


def bridge_headers():
    """The bridge refuses callers without the per-launch secret that scripts/lab.py issues.
    The secret file wins over the environment, so an API reused across launches stays
    in step with a restarted bridge."""
    token = os.getenv("FORGE_RELAY_BRIDGE_TOKEN", "")
    with suppress(OSError):
        token = (store.DATA / "relay-bridge.token").read_text().strip() or token
    return {"Authorization": "Bearer " + token}


def send(campaign_id, sender, recipient, category, text, experiment_id=None, route="dm", reply_to=None):
    """Hand a message to Agent Relay. DMs carry 1:1 handoffs; the research room carries
    decisions; reply_to threads a reply under a room message. Returns the verified receipt."""
    message = Message(
        id=uuid.uuid4().hex,
        campaign_id=campaign_id,
        sender=sender,
        recipient=recipient,
        category=category,
        text=text,
        experiment_id=experiment_id,
        route=route,
        reply_to=reply_to,
    )
    # Persist assignment before notification. Retries retain the same idempotency key.
    store.event(campaign_id, "message_pending", text, sender, message=message.model_dump())
    last_error = None
    for _ in range(3):
        try:
            r = httpx.post(
                bridge_url() + "/message", json=message.model_dump(), headers=bridge_headers(), timeout=30
            )
            r.raise_for_status()
            receipt = r.json()
            if not receipt.get("delivered") or not receipt.get("message_id"):
                raise ValueError("Relay delivery unconfirmed")
            store.event(
                campaign_id,
                "message",
                text,
                sender,
                recipient=recipient,
                category=category,
                experiment_id=experiment_id,
                message_id=message.id,
                relay_id=receipt["message_id"],
                route=receipt.get("route", route),
                thread_parent_id=reply_to,
                receipt=receipt,
            )
            return receipt
        except (httpx.HTTPError, ValueError) as e:
            last_error = type(e).__name__
    raise RuntimeError(f"Agent Relay delivery failed ({last_error}); execution paused before downstream work")


def react(campaign_id, sender, relay_message_id, emoji):
    """A role reacts to a research-room message. Reactions are visible state, never work
    authorization, so a failure is recorded and does not stop the campaign."""
    try:
        r = httpx.post(
            bridge_url() + "/react",
            headers=bridge_headers(),
            json={
                "campaign_id": campaign_id,
                "sender": sender,
                "message_id": relay_message_id,
                "emoji": emoji,
            },
            timeout=20,
        )
        r.raise_for_status()
        result = r.json()
    except (httpx.HTTPError, ValueError) as e:
        store.event(campaign_id, "relay_warning", f"Reaction not recorded ({type(e).__name__})", sender)
        return None
    store.event(
        campaign_id,
        "reaction",
        f"{sender} reacted {emoji}",
        sender,
        relay_id=relay_message_id,
        emoji=emoji,
        reactions=result.get("reactions", []),
    )
    return result


def transition(campaign_id, status, reason):
    state = json.loads(actor(campaign_id).transition(status, reason))
    store.patch(campaign_id, lambda s: s.update(status=status, actor=state, progress=None))
    store.event(campaign_id, "stage", reason, status=status)
    from services.core.briefing import publish

    publish(campaign_id)


def cancelled(campaign_id):
    s = store.get(campaign_id)
    return bool(s.get("cancel_requested"))


def artifact(campaign_id, path, kind, title):
    from forge_training.artifacts import file_sha256 as sha

    p = Path(path)
    a = {
        "id": uuid.uuid4().hex,
        "campaign_id": campaign_id,
        "name": title,
        "kind": kind,
        "path": str(p.relative_to(store.ARTIFACTS / campaign_id)),
        "sha256": sha(p),
        "bytes": p.stat().st_size,
        "created_at": store.now(),
    }
    get_actor("ArtifactRegistryActor", campaign_id).register(json.dumps(a))
    store.patch(campaign_id, lambda s: s.setdefault("artifacts", []).append(a))
    store.event(campaign_id, "artifact", f"Saved {title}", "trainer", artifact=a)
    return a


def checkpoint_zip(folder):
    import zipfile

    target = folder.parent / f"{folder.name}.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for p in folder.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(folder))
    return target


def add_review(campaign_id, evidence):
    review = json.loads(governance(campaign_id).review(json.dumps(evidence)))
    review["timestamp"] = store.now()
    store.patch(campaign_id, lambda s: s.setdefault("reviews", []).append(review))
    store.event(
        campaign_id,
        "review",
        "; ".join(review["reason_codes"]) or "Independent checks passed",
        "governance",
        review=review,
    )
    return review


def probe(campaign_id):
    # A disclosed adversarial fixture. It never modifies a real experiment.
    evidence = {
        "id": "probe-" + uuid.uuid4().hex[:8],
        "phase": "preflight",
        "reward_version": "always-reward-one",
        "sealed_data_access": True,
        "provenance_valid": True,
        "controlled_probe": True,
    }
    review = add_review(campaign_id, evidence)
    try:
        governance(campaign_id).promote(evidence["id"], "0" * 64)
    except Exception:
        store.event(
            campaign_id,
            "blocked",
            "Controlled probe: modified reward and sealed data access rejected. Promotion attempt blocked.",
            "governance",
        )
        return review
    raise RuntimeError("Governance enforcement failed")


def spawn(campaign_id):
    import subprocess
    import threading
    import signal
    import shutil

    log = (store.ARTIFACTS / campaign_id / "worker.log").open("a")
    harness_mode = os.environ.get("FORGE_WORKER_HARNESS", "classic")
    if harness_mode not in {"classic", "durable"}:
        log.close()
        raise ValueError("FORGE_WORKER_HARNESS must be classic or durable")
    module = "services.harness.campaign" if harness_mode == "durable" else "services.core.runner"
    command = [sys.executable, "-m", module, campaign_id]
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        command = ["caffeinate", "-i", "-s", *command]
    process = subprocess.Popen(
        command,
        cwd=store.ROOT,
        stdout=log,
        stderr=log,
        start_new_session=True,
    )
    s = store.patch(campaign_id, lambda snapshot: snapshot.update(worker_pid=process.pid))
    log.close()

    def enforce_deadline():
        try:
            process.wait(timeout=s["config"]["max_seconds"])
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=7)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            with store.LOCK:
                latest = store.get(campaign_id)
                if latest["status"] not in {"completed", "cancelled", "failed"}:
                    latest["status"] = "failed"
                    latest["error"] = "Compute deadline reached. Worker stopped; partial artifacts preserved."
                    store.save(latest)
                    store.event(campaign_id, "error", latest["error"], "trainer")
            try:
                state = json.loads(actor(campaign_id).snapshot())
                for key, reservation in state["reservations"].items():
                    if reservation["status"] == "reserved":
                        actor(campaign_id).settle(key, reservation["seconds"])
                actor(campaign_id).transition("failed", "Compute watchdog terminated the worker")
            except Exception:
                pass

    threading.Thread(target=enforce_deadline, daemon=True).start()
    return process.pid
