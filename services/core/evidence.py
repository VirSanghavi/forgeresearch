"""Cross-campaign evidence: results, trust checks and observability, read only from what was recorded.

Every value comes from a stored campaign snapshot, its events or a re-hash of its artifact files.
Nothing is estimated. A fact the campaign did not record is reported as not recorded, never as passed.
"""

from pathlib import Path

from services.core import store

# Re-hashing every artifact on each request is wasteful; a file is re-hashed only when it changes.
_HASHES: dict[tuple[str, int, int], str] = {}


def _sha256(path: Path):
    from forge_training.artifacts import file_sha256

    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _HASHES:
        _HASHES[key] = file_sha256(path)
    return _HASHES[key]


def _score(evaluation):
    if not isinstance(evaluation, dict) or not isinstance(evaluation.get("n"), int):
        return None
    return {"correct": evaluation.get("correct"), "n": evaluation["n"]}


def _check(check_id, label, state, detail):
    return {"id": check_id, "label": label, "state": state, "detail": detail}


def artifacts(campaign):
    """Re-hash each registered artifact exactly as the download route resolves it."""
    root = (store.ARTIFACTS / campaign["id"]).resolve()
    verified = mismatched = missing = size = 0
    for a in campaign.get("artifacts") or []:
        size += a.get("bytes") or 0
        p = (root / str(a.get("path", ""))).resolve()
        if not p.is_relative_to(root) or not p.is_file():
            missing += 1
        elif _sha256(p) == a.get("sha256"):
            verified += 1
        else:
            mismatched += 1
    total = len(campaign.get("artifacts") or [])
    return {"count": total, "verified": verified, "mismatched": mismatched, "missing": missing, "bytes": size}


def agents(campaign):
    calls = campaign.get("agent_calls")
    if not isinstance(calls, list):
        return None
    roles: dict[str, dict] = {}
    names: set[str] = set()
    for call in calls:
        if not isinstance(call, dict):
            continue
        usage = call.get("usage") or {}
        role = roles.setdefault(
            str(call.get("role")),
            {
                "role": call.get("role"),
                "model": call.get("model"),
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "subagents": 0,
                "vm_runs": 0,
            },
        )
        role["calls"] += call.get("calls") or 0
        role["vm_runs"] += call.get("vm_runs") or 0
        role["input_tokens"] += usage.get("input_tokens") or 0
        role["output_tokens"] += usage.get("output_tokens") or 0
        role["subagents"] += len(call.get("subagents") or [])
        names.update(call.get("subagents") or [])
    by_role = list(roles.values())
    return {
        "calls": sum(r["calls"] for r in by_role),
        "input_tokens": sum(r["input_tokens"] for r in by_role),
        "output_tokens": sum(r["output_tokens"] for r in by_role),
        "subagent_runs": sum(r["subagents"] for r in by_role),
        "vm_runs": sum(r["vm_runs"] for r in by_role),
        "subagent_kinds": sorted(names),
        "by_role": by_role,
    }


def vms(campaign):
    """The Freestyle VMs research roles used, as the campaign saved them when it closed them."""
    benches = [w for w in campaign.get("workbenches") or [] if isinstance(w, dict)]
    if not benches:
        return None
    return {
        "count": len(benches),
        "runs": sum(w.get("runs") or 0 for w in benches),
        "seconds": sum(w.get("seconds") or 0 for w in benches),
        "deleted": sum(1 for w in benches if w.get("deleted")),
    }


def relay(events):
    """Sends are logged before delivery; a delivery counts only with Relay's confirmed receipt."""
    sent = sum(1 for e in events if e.get("kind") == "message_pending")
    receipts = [e.get("receipt") or {} for e in events if e.get("kind") == "message" and e.get("relay_id")]
    delivered = sum(1 for r in receipts if r.get("delivered"))
    readback = sum(1 for r in receipts if r.get("recipient_readback_verified"))
    return {"sent": sent, "delivered": delivered, "readback_verified": readback}


def checks(campaign, events, files):
    actor, final = campaign.get("actor") or {}, campaign.get("final") or {}
    reviews = [r for r in campaign.get("reviews") or [] if isinstance(r, dict)]
    evidence = [r.get("evidence") or {} for r in reviews]
    probes = [e for e in evidence if e.get("controlled_probe")]
    real = [e for e in evidence if not e.get("controlled_probe")]
    has_final = bool(final.get("baseline"))
    result = []

    sealed = len(campaign.get("sealed_ids") or [])
    if not has_final:
        result.append(
            _check(
                "sealed_once",
                "Sealed test queried once",
                "not_applicable",
                f"No final evaluation. The campaign ended {campaign.get('status')}.",
            )
        )
    elif actor.get("final_claimed"):
        result.append(
            _check(
                "sealed_once",
                "Sealed test queried once",
                "pass",
                f"The one-time final claim is recorded in Terse; a second claim is refused. "
                f"{sealed} sealed questions.",
            )
        )
    else:
        result.append(
            _check(
                "sealed_once",
                "Sealed test queried once",
                "not_recorded",
                "No one-time final claim in the saved actor state.",
            )
        )

    if not has_final:
        result.append(
            _check(
                "selection_frozen",
                "Selection frozen before the final",
                "not_applicable",
                "No final evaluation.",
            )
        )
    elif actor.get("selection_frozen") and final.get("selection_basis") == "development only":
        chosen = (
            "the original model" if final.get("checkpoint_id") == "baseline" else final.get("checkpoint_id")
        )
        result.append(
            _check(
                "selection_frozen",
                "Selection frozen before the final",
                "pass",
                f"Chose {chosen} on development data only, then froze it.",
            )
        )
    else:
        result.append(
            _check(
                "selection_frozen",
                "Selection frozen before the final",
                "not_recorded",
                "The saved state does not show a frozen, development-only selection.",
            )
        )

    reloads = sum(1 for e in events if e.get("kind") == "stage" and "fresh model" in (e.get("text") or ""))
    trained = any(a.get("kind") == "checkpoint" for a in campaign.get("artifacts") or [])
    if not trained:
        result.append(
            _check(
                "reload", "Checkpoint reloaded independently", "not_applicable", "No checkpoint was saved."
            )
        )
    elif reloads:
        result.append(
            _check(
                "reload",
                "Checkpoint reloaded independently",
                "pass",
                f"{reloads} saved adapter{'s' if reloads != 1 else ''} loaded into a fresh model "
                f"before scoring.",
            )
        )
    else:
        result.append(
            _check(
                "reload",
                "Checkpoint reloaded independently",
                "not_recorded",
                "No fresh-model reload was logged for this campaign.",
            )
        )

    if not files["count"]:
        result.append(
            _check("sha256", "Artifacts match their SHA-256", "not_applicable", "No artifacts saved.")
        )
    elif files["mismatched"]:
        result.append(
            _check(
                "sha256",
                "Artifacts match their SHA-256",
                "fail",
                f"{files['mismatched']} of {files['count']} files differ from their registered hash.",
            )
        )
    elif files["missing"]:
        result.append(
            _check(
                "sha256",
                "Artifacts match their SHA-256",
                "partial",
                f"{files['verified']} of {files['count']} re-hashed and matched. "
                f"{files['missing']} not present on this machine.",
            )
        )
    else:
        result.append(
            _check(
                "sha256",
                "Artifacts match their SHA-256",
                "pass",
                f"All {files['count']} files re-hashed just now and matched.",
            )
        )

    if reviews:
        rejected = [r for r in reviews if r.get("decision") != "approved"]
        codes = sorted({c for r in reviews for c in r.get("reason_codes") or []})
        detail = f"{len(reviews) - len(rejected)} approved, {len(rejected)} rejected."
        if codes:
            detail += " Reason codes: " + ", ".join(codes) + "."
        result.append(_check("governance", "Governance reviewed every gate", "pass", detail))
    else:
        result.append(
            _check(
                "governance",
                "Governance reviewed every gate",
                "not_recorded",
                "No governance decisions were saved.",
            )
        )

    flags = [e.get("sealed_data_access") for e in real if "sealed_data_access" in e]
    if any(flags):
        result.append(
            _check(
                "no_sealed_access",
                "No sealed data in training",
                "fail",
                "A governance record shows sealed data access.",
            )
        )
    elif flags:
        detail = f"{len(flags)} governance records show no sealed data access."
        if probes:
            detail += f" {len(probes)} disclosed probe{'s' if len(probes) != 1 else ''} that tried it were rejected."
        result.append(_check("no_sealed_access", "No sealed data in training", "pass", detail))
    else:
        result.append(
            _check(
                "no_sealed_access",
                "No sealed data in training",
                "not_recorded",
                "Governance records do not include a sealed-access field.",
            )
        )

    hashes = {e.get("task_pack_sha256") for e in real if e.get("task_pack_sha256")}
    if len(hashes) == 1:
        result.append(
            _check(
                "pack_pinned",
                "Task pack hash pinned",
                "pass",
                f"Every gate saw task pack {next(iter(hashes))[:12]}.",
            )
        )
    elif hashes:
        result.append(
            _check(
                "pack_pinned",
                "Task pack hash pinned",
                "fail",
                f"Gates saw {len(hashes)} different task pack hashes.",
            )
        )
    else:
        result.append(
            _check(
                "pack_pinned",
                "Task pack hash pinned",
                "not_recorded",
                "No task pack hash in the governance records.",
            )
        )
    return result


def summarize(campaign, events):
    final = campaign.get("final") or {}
    production = final.get("production") or {}
    hardware = campaign.get("hardware") or {}
    actor = campaign.get("actor") or {}
    files = artifacts(campaign)
    return {
        "id": campaign["id"],
        "name": campaign.get("name"),
        "status": campaign.get("status"),
        "created_at": campaign.get("created_at"),
        "model_id": (campaign.get("config") or {}).get("model_id"),
        "director": campaign.get("director"),
        "hardware": {"backend": hardware.get("backend"), "name": hardware.get("name")} if hardware else None,
        "elapsed_seconds": campaign.get("elapsed_seconds"),
        "compute_seconds": campaign.get("compute_seconds"),
        "experiments": len(campaign.get("experiments") or []),
        "sealed": {
            "baseline": _score(final.get("baseline")),
            "checkpoint": _score(final.get("checkpoint")),
            "improvement_pp": final.get("improvement_pp"),
            "interval_pp": final.get("paired_bootstrap_95_pp"),
            "improved": final.get("improved_examples"),
            "regressed": final.get("regressed_examples"),
            "claim": final.get("improvement_claim"),
            "checkpoint_id": final.get("checkpoint_id"),
            "kept_original": final.get("checkpoint_id") == "baseline",
        }
        if final.get("baseline")
        else None,
        "production": {
            "baseline": production.get("baseline"),
            "checkpoint": production.get("checkpoint"),
            "improvement_pp": production.get("improvement_pp"),
            "interval_pp": production.get("paired_bootstrap_95_pp"),
            "claim": production.get("improvement_claim"),
        }
        if production
        else None,
        "checks": checks(campaign, events, files),
        "agents": agents(campaign),
        "relay": relay(events),
        "vms": vms(campaign),
        "terse": {"state_version": actor.get("version"), "decisions": len(actor.get("decisions") or [])}
        if actor
        else None,
        "artifacts": files,
    }


def build():
    return [summarize(c, store.events(c["id"])) for c in store.campaigns()]
