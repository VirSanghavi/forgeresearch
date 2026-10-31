"""Skimmable status reports derived exclusively from persisted research evidence."""

from services.core import packs, store


def build(snapshot, events):
    pack = packs.get(snapshot.get("config", {}).get("task_pack"))
    experiments = snapshot.get("experiments", [])
    active = next(
        (e for e in experiments if e["status"] in {"training", "evaluating", "approved", "proposed"}),
        None,
    )
    planning = snapshot["status"] == "diagnosing" and active is None
    current = active or (experiments[-1] if experiments and not planning else None)
    key = f"{snapshot['id']}-{current['id']}" if current else None
    reviews = snapshot.get("reviews", [])
    checkpoint = next((r for r in reversed(reviews) if r["id"] == key), None)
    preflight = next((r for r in reversed(reviews) if r["id"] == f"{key}-preflight"), None)
    blocked = [r for r in reviews if r["decision"] == "rejected"]
    context = snapshot.get("director_context")
    evidence = checkpoint["evidence"] if checkpoint else {}
    checks = [
        {"label": "Training permission", "state": preflight["decision"] if preflight else "pending"},
        {
            "label": "Changed model weights",
            "state": "verified" if evidence.get("weights_changed") else "pending",
        },
        {
            "label": "Independent checkpoint reload",
            "state": "verified" if evidence.get("independent_reload") else "pending",
        },
        {
            "label": "Artifact checksum",
            "state": "verified" if evidence.get("artifact_verified") else "pending",
        },
        {
            "label": pack.regression_title,
            "state": checkpoint["decision"] if checkpoint else "pending",
        },
    ]
    return {
        "as_of": snapshot.get("updated_at") or snapshot.get("created_at") or store.now(),
        "phase": snapshot["status"],
        "experiment_number": len(experiments) + 1
        if planning
        else experiments.index(current) + 1
        if current
        else 0,
        "experiment_limit": snapshot["config"]["max_experiments"],
        "experiment_id": current["id"] if current else None,
        "strategy": current["strategy"] if current else None,
        "hypothesis": current["hypothesis"] if current else None,
        "benchmarks": {
            "development_before": {k: snapshot["baseline"][k] for k in ["correct", "n", "score"]}
            if snapshot.get("baseline")
            else None,
            "development_best": {k: snapshot["best"]["evaluation"][k] for k in ["correct", "n", "score"]}
            if snapshot.get("best")
            else None,
            "final": {
                phase: {k: snapshot["final"][phase][k] for k in ["correct", "n", "score"]}
                for phase in ["baseline", "checkpoint"]
            }
            if snapshot.get("final")
            else None,
        },
        "progress": snapshot.get("progress"),
        "director": {
            "input_recorded": context is not None,
            "failure_examples": len(context["failures"]) if context else None,
            "previous_results": len(context["previous_experiments"]) if context else None,
            "development_score_seen": context["development_score"] if context else None,
            "final_test_data_provided": False,
        },
        "oversight": {
            "decision": checkpoint["decision"]
            if checkpoint
            else preflight["decision"]
            if preflight
            else "pending",
            "decision_scope": "checkpoint"
            if checkpoint
            else "training preflight"
            if preflight
            else "awaiting evidence",
            "checks": checks,
            "blocked_proposals": len(blocked),
            "last_block": blocked[-1] if blocked else None,
            "regression_scope": pack.regression_label,
        },
        "execution": {
            "terse_worlds": len(snapshot.get("environments", [])),
            "rl_rollouts": sum("policy_sha256" in e for e in snapshot.get("environments", [])),
            "relay_deliveries": sum(e["kind"] == "message" and bool(e.get("relay_id")) for e in events),
            "settled_compute_seconds": snapshot.get("actor", {}).get("consumed_seconds", 0),
            "time_limit_seconds": snapshot["config"]["max_seconds"],
            "hardware": snapshot.get("hardware"),
        },
    }


def publish(campaign_id):
    snapshot = store.get(campaign_id)
    report = build(snapshot, store.events(campaign_id))
    current = (
        f"Experiment {report['experiment_number']}/{report['experiment_limit']}"
        if report["experiment_number"]
        else "Baseline before training"
    )
    store.event(
        campaign_id,
        "briefing",
        f"{current} · {report['phase']}. Oversight: {report['oversight']['decision']} ({report['oversight']['decision_scope']}).",
        "director",
        briefing=report,
    )
    return report
