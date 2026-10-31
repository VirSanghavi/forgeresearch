"""Crash-test endpoints for the Terse runtime. Loopback only, and only with FORGE_CHAOS=1."""

import os
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from services.core import chaos, store
from services.core.lifecycle import TERMINAL

router = APIRouter(prefix="/chaos/actors")
incidents = chaos.Incidents()
LOOPBACK = {"127.0.0.1", "::1"}


def enabled():
    return os.environ.get("FORGE_CHAOS") == "1"


class KillRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    campaign_id: str | None = Field(default=None, pattern=r"^forge-[0-9a-f]{8}$")


@router.post("/kill", status_code=202)
def kill(request: Request, body: KillRequest | None = None):
    if not enabled():
        raise HTTPException(403, "Crash tests are disabled. Start the lab with FORGE_CHAOS=1.")
    if not request.client or request.client.host not in LOOPBACK:
        raise HTTPException(403, "Crash tests are accepted only from this machine")
    campaigns = store.campaigns()
    wanted = body.campaign_id if body else None
    target = next(
        (c for c in campaigns if c["id"] == wanted)
        if wanted
        else (c for c in campaigns if c["status"] not in TERMINAL),
        None,
    ) or (None if wanted else (campaigns[0] if campaigns else None))
    if not target:
        raise HTTPException(404, "No campaign to prove actor state against")
    try:
        incident = incidents.begin(target["id"], [e["id"] for e in target.get("experiments", [])])
    except RuntimeError as error:
        raise HTTPException(409, str(error)) from error
    store.event(
        target["id"],
        "runtime_killed",
        f"Crash test: SIGKILL sent to the Terse runtime (pid {incident['kill']['pid']}).",
        "data",
        incident_id=incident["id"],
    )
    return status()


@router.get("")
def status():
    runtime = chaos.supervised_runtime(incidents.folder)
    return {
        "enabled": enabled(),
        "supervised": bool(runtime),
        "runtime": {"pid": runtime["pid"], "restarts": runtime["restarts"]} if runtime else None,
        "incidents": incidents.view(store.get, store.events),
    }
