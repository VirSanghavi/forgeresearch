"""One overseer cycle: pull new traces, run tier one on them, judge the eligible ones, raise alerts."""

import json
import os
from datetime import UTC, datetime, timedelta

from forge_training.scoring import wilson_interval

from services.core import packs, store
from services.core.overseer import db, judge, traces

CHECK_BATCH = 2000
JUDGE_BATCH = 25


def percentile(values, q):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def check_new():
    """Tier one on every unchecked trace. Returns how many were checked."""
    pending = db.rows("select * from traces where checked = 0 order by received_at limit ?", (CHECK_BATCH,))
    slo = (
        float(os.environ["FORGE_OVERSEER_LATENCY_SLO_MS"])
        if os.environ.get("FORGE_OVERSEER_LATENCY_SLO_MS")
        else None
    )
    rate = judge.settings()["sample_rate"]
    statements, p99 = [], {}
    for row in pending:
        pack_id = row["pack_id"]
        if pack_id not in p99:
            recent = db.rows(
                "select latency_ms from traces where pack_id = ? and latency_ms is not null"
                " order by received_at desc limit 1000",
                (pack_id,),
            )
            p99[pack_id] = percentile([r["latency_ms"] for r in recent], 0.99) if len(recent) >= 50 else None
        trace = db.payload(row)
        found = traces.detect(trace, packs.get(pack_id), p99[pack_id], slo)
        kinds = {s["kind"] for s in found}
        for signal in found:
            statements.append(
                (
                    "insert or ignore into signals (key, kind, evidence) values (?, ?, ?)",
                    (row["key"], signal["kind"], json.dumps(signal["evidence"])),
                )
            )
        statements.append(
            (
                "update traces set checked = 1, hard = ?, suspicious = ?, sampled = ? where key = ?",
                (
                    int(bool(kinds & traces.HARD)),
                    int(bool(kinds & traces.SOFT)),
                    int(judge.sampled(row["key"], row["received_at"][:10], rate)),
                    row["key"],
                ),
            )
        )
    if statements:
        db.execute(statements)
    return len(pending)


def judge_eligible(limit=JUDGE_BATCH):
    """Judge suspicious and sampled traces until the batch or the daily budget runs out."""
    ready, reason = judge.available()
    if not ready:
        return {"judged": 0, "skipped": reason}
    from forge_training.distill import is_loopback

    if is_loopback(judge.settings()["url"]) and any(
        s["status"] not in {"completed", "cancelled", "failed", "interrupted"} for s in store.campaigns()
    ):
        return {"judged": 0, "skipped": "A campaign holds this machine; the local judge waits"}
    eligible = db.rows(
        "select t.* from traces t left join judgments j on j.key = t.key"
        " where j.key is null and t.checked = 1 and t.sealed_overlap = 0 and (t.suspicious = 1 or t.sampled = 1)"
        " order by t.sampled desc, t.received_at limit ?",
        (limit,),
    )
    judged = 0
    for row in eligible:
        try:
            result = judge.judge(row["key"], db.payload(row), packs.get(row["pack_id"]))
        except RuntimeError as error:
            return {"judged": judged, "skipped": f"Judge unavailable: {error}"}
        if result is None:
            return {"judged": judged, "skipped": "Daily judge budget reached"}
        judged += 1
    return {"judged": judged, "skipped": None}


def rates(pack_id, start, end):
    rows = db.rows(
        "select key, latency_ms, hard from traces where pack_id = ? and sealed_overlap = 0"
        " and received_at >= ? and received_at < ?",
        (pack_id, start, end),
    )
    kinds = db.rows(
        "select s.kind, count(*) as n from signals s join traces t on t.key = s.key where t.pack_id = ?"
        " and t.received_at >= ? and t.received_at < ? group by s.kind",
        (pack_id, start, end),
    )
    return len(rows), {k["kind"]: k["n"] for k in kinds}, percentile([r["latency_ms"] for r in rows], 0.95)


def alerts(now=None):
    """Drift: today against the trailing seven days, per pack. One alert per kind per day."""
    now = now or datetime.now(UTC)
    day = now.strftime("%Y-%m-%d")
    start_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    raised = []
    for pack_id in [r["pack_id"] for r in db.rows("select distinct pack_id from traces")]:
        n, counts, p95 = rates(
            pack_id, start_today.isoformat(), (start_today + timedelta(days=1)).isoformat()
        )
        base_n, base_counts, base_p95 = rates(
            pack_id, (start_today - timedelta(days=7)).isoformat(), start_today.isoformat()
        )
        if n < 50 or base_n < 50:
            continue
        for kind in set(counts) | set(base_counts):
            rate, base = counts.get(kind, 0) / n, base_counts.get(kind, 0) / base_n
            se = (base * (1 - base) / n) ** 0.5
            if rate - base >= max(0.02, 3 * se):
                raised.append((pack_id, kind, {"today": round(rate, 4), "baseline": round(base, 4), "n": n}))
        if p95 and base_p95 and p95 >= 1.5 * base_p95:
            raised.append((pack_id, "latency_p95", {"today_ms": p95, "baseline_ms": base_p95}))
        seen = {
            r["model_version"]
            for r in db.rows(
                "select distinct model_version from traces where pack_id = ? and received_at < ?",
                (pack_id, start_today.isoformat()),
            )
        }
        for version in db.rows(
            "select distinct model_version from traces where pack_id = ? and received_at >= ?",
            (pack_id, start_today.isoformat()),
        ):
            if version["model_version"] and version["model_version"] not in seen:
                raised.append((pack_id, "new_model_version", {"model_version": version["model_version"]}))
    db.execute(
        [
            (
                "insert or ignore into alerts (id, day, pack_id, kind, detail, created_at) values (?, ?, ?, ?, ?, ?)",
                (
                    f"{day}:{pack_id}:{kind}:{detail.get('model_version', '')}",
                    day,
                    pack_id,
                    kind,
                    json.dumps(detail),
                    store.now(),
                ),
            )
            for pack_id, kind, detail in raised
        ]
    )
    return len(raised)


def cycle():
    from services.core.overseer import connections, pull

    supervised = connections.supervise()
    pulled = pull.pull() if pull.configured() else {"pulled": 0, "skipped": "Supabase buffer not configured"}
    checked = check_new()
    judged = judge_eligible()
    return {
        "connections": supervised,
        "pulled": pulled,
        "checked": checked,
        "judged": judged,
        "alerts": alerts(),
    }


def summary(pack_id=None):
    where, args = ("where pack_id = ?", (pack_id,)) if pack_id else ("", ())
    rows = db.rows(
        f"select key, pack_id, model_version, latency_ms, hard, suspicious, sampled, sealed_overlap, payload from traces {where}",
        args,
    )
    live = [r for r in rows if not r["sealed_overlap"]]
    kinds = db.rows(
        "select s.kind, count(*) as n from signals s join traces t on t.key = s.key"
        + (" where t.pack_id = ?" if pack_id else "")
        + " group by s.kind",
        args,
    )
    judged = db.rows(
        "select j.verdict, j.result, t.sampled from judgments j join traces t on t.key = j.key"
        + (" where t.pack_id = ?" if pack_id else ""),
        args,
    )
    sample = [j for j in judged if j["sampled"]]
    failed = sum(j["verdict"] == "fail" for j in sample)
    negative = sum(
        1
        for r in live
        if (lambda f: f.get("rating", 0) < 0 or f.get("correction"))(
            json.loads(r["payload"]).get("feedback") or {}
        )
    )
    versions = {}
    for r in live:
        v = versions.setdefault(r["model_version"] or "unknown", {"traces": 0, "flagged": 0})
        v["traces"] += 1
        v["flagged"] += r["hard"] or r["suspicious"]
    return {
        "traces": len(rows),
        "sealed_overlap": len(rows) - len(live),
        "flagged": sum(1 for r in live if r["hard"] or r["suspicious"]),
        "signals": [
            {
                "kind": k["kind"],
                "label": traces.LABELS.get(k["kind"], k["kind"]),
                "count": k["n"],
                "rate": k["n"] / len(live) if live else 0,
            }
            for k in sorted(kinds, key=lambda k: -k["n"])
        ],
        "latency_ms": {
            q: percentile([r["latency_ms"] for r in live], v)
            for q, v in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))
        },
        "negative_feedback_rate": negative / len(live) if live else 0,
        "judged": {
            "total": len(judged),
            "random_sample": len(sample),
            "sample_failures": failed,
            "sample_failure_rate": failed / len(sample) if sample else None,
            "sample_ci95": wilson_interval(failed, len(sample)) if sample else None,
            "hallucinations": sum(json.loads(j["result"]).get("hallucination") is True for j in judged),
        },
        "models": versions,
        "labels": db.rows(
            "select count(*) as n from labels l join traces t on t.key = l.key"
            + (" where t.pack_id = ?" if pack_id else ""),
            args,
        )[0]["n"],
        "spend_today": judge.spend(),
        "budget": {k: judge.settings()[k] for k in ("daily_usd", "daily_calls", "sample_rate")},
        "judge": dict(zip(("ready", "reason"), judge.available(), strict=True)),
        "alerts": db.rows(
            "select * from alerts"
            + (" where pack_id = ?" if pack_id else "")
            + " order by created_at desc limit 20",
            args,
        ),
    }
