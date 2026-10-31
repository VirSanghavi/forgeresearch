"""Hephaestus's authority over one campaign's research agents.

Orders, per-role settings, holds, assigned experiments and the call to finish all commit in the campaign's
TeamActor, and the worker reads them there at each decision. This is control of how the agents work. It
gives no way to change what counts: budgets, governance and the sealed final don't read any of it.
"""

import json
import os
import time

from services.core import agents
from services.core.actor_client import get_actor

HOLD_POLL_SECONDS = 2


def control(campaign_id):
    return get_actor("TeamActor", campaign_id)


def snapshot(campaign_id):
    return json.loads(control(campaign_id).snapshot())


def ceilings():
    """The most anyone can give one agent's decision, however it's asked."""
    return {
        "max_subagents": int(os.getenv("FORGE_AGENT_CEILING_SUBAGENTS", "8")),
        "depth": int(os.getenv("FORGE_AGENT_CEILING_DEPTH", "3")),
        "max_calls": int(os.getenv("FORGE_AGENT_CEILING_CALLS", "40")),
    }


def clean_settings(requested):
    """Settings a role may run with, or a ValueError saying what's wrong with them."""
    found, limit = {}, ceilings()
    for name, value in requested.items():
        if value is None:
            found[name] = None
        elif name == "enabled":
            found[name] = bool(value)
        elif name == "model":
            found[name] = _deployed(str(value))
        elif name in limit:
            number = int(value)
            if not 0 <= number <= limit[name] or (name == "max_calls" and number < 1):
                raise ValueError(
                    f"{name} must be within {'1' if name == 'max_calls' else '0'}-{limit[name]}."
                )
            found[name] = number
        else:
            raise ValueError(f"Agents have no setting called {name}.")
    return found


def _deployed(model):
    known = agents.deployments()
    if known is not None and model not in known:
        raise ValueError(f"{model} isn't deployed. Deployed: {', '.join(sorted(known))}.")
    return model


def settings_for(campaign_id, role):
    return snapshot(campaign_id)["settings"].get(role, {})


def read_orders(campaign_id, role, where):
    """The orders this role must weigh in the decision it's making now, recorded as read by it."""
    orders = json.loads(control(campaign_id).orders_for(role))
    if orders:
        control(campaign_id).mark_read(role, json.dumps([o["id"] for o in orders]), where)
    return [o["text"] for o in orders]


def wait_while_held(campaign_id, stop, announce=lambda _state: None, sleep=time.sleep):
    """Pause here while Hephaestus holds the campaign. The clock keeps running, and a stop still works."""
    held = snapshot(campaign_id)["held"]
    if not held:
        return False
    announce(held)
    while held and not stop():
        sleep(HOLD_POLL_SECONDS)
        held = snapshot(campaign_id)["held"]
    return True


def roster(campaign, events, state):
    """Every role at a glance: how it's set up, what it has done and what it was told. Numbers and names
    only, apart from the orders, which are Hephaestus's own words."""
    calls = campaign.get("agent_calls") or []
    delivered = [e for e in events if e.get("kind") == "message" and e.get("relay_id")]
    roles = {}
    for role, title in agents.ROLES.items():
        mine = [c for c in calls if c.get("role") == role]
        setting = state["settings"].get(role, {})
        spoken = [e for e in delivered if e.get("agent") == role]
        roles[role] = {
            "title": title,
            "enabled": setting.get("enabled", True),
            "model": setting.get("model") or (agents.model(role) if agents.configured() else None),
            "limits": {k: setting.get(k, v) for k, v in agents.limits().items()},
            "decisions": len(mine),
            "model_calls": sum(int(c.get("calls") or 0) for c in mine),
            "tokens": sum(
                int((c.get("usage") or {}).get(k) or 0)
                for c in mine
                for k in ("input_tokens", "output_tokens")
            ),
            "subagents": sum(len(c.get("subagents") or []) for c in mine),
            "last_said": {
                "category": spoken[-1].get("category"),
                "to": spoken[-1].get("recipient"),
                "experiment": spoken[-1].get("experiment_id"),
            }
            if spoken
            else None,
        }
    return {
        "campaign": campaign["id"],
        "status": campaign["status"],
        "progress": campaign.get("progress"),
        "held": state["held"],
        "finishing": state["finish"],
        "agents": roles,
        "orders": [
            {"id": o["id"], "to": o["roles"], "text": o["text"], "read": o["read"]} for o in state["orders"]
        ],
        "assigned": [
            {
                "id": a["id"],
                "strategy": a["proposal"]["strategy"],
                "learning_rate": a["proposal"]["learning_rate"],
                "lora_rank": a["proposal"]["lora_rank"],
                "ran_as": a["taken_by"],
            }
            for a in state["assignments"]
        ],
    }


def lab_usage(campaigns, day):
    """What the research agents have used today, from the provider's own counts on each decision."""
    usage = {"model_calls": 0, "input_tokens": 0, "output_tokens": 0, "by_model": {}}
    for campaign in campaigns:
        if not str(campaign.get("created_at") or "").startswith(day):
            continue
        for call in campaign.get("agent_calls") or []:
            spent = call.get("usage") or {}
            tokens = int(spent.get("input_tokens") or 0) + int(spent.get("output_tokens") or 0)
            usage["model_calls"] += int(call.get("calls") or 0)
            usage["input_tokens"] += int(spent.get("input_tokens") or 0)
            usage["output_tokens"] += int(spent.get("output_tokens") or 0)
            name = call.get("model") or "unknown"
            usage["by_model"][name] = usage["by_model"].get(name, 0) + tokens
    return usage
