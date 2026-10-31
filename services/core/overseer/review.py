"""The review queue, reviewer labels and the hand-off back to the lab.

Only a reviewer writes a label. User corrections and judge rationales are shown beside a trace
for context and never pre-fill or become an answer: anyone can send a correction, and a judge
can be wrong, so either could poison training.
"""

import json

from services.core import packs, production, store
from services.core.overseer import db


def queue(pack_id, limit=50):
    """Flagged or judged-failing traces first, then the random sample, all still unlabeled."""
    rows = db.rows(
        "select t.*, j.result as judgment from traces t left join judgments j on j.key = t.key"
        " left join labels l on l.key = t.key where t.pack_id = ? and l.key is null and t.sealed_overlap = 0"
        " and (t.hard = 1 or t.suspicious = 1 or t.sampled = 1 or j.verdict = 'fail')"
        " order by (t.hard = 1 or j.verdict = 'fail') desc, t.sampled desc, t.received_at desc limit ?",
        (pack_id, limit),
    )
    signals = {}
    for s in db.rows(
        "select s.* from signals s join traces t on t.key = s.key where t.pack_id = ?", (pack_id,)
    ):
        signals.setdefault(s["key"], []).append({"kind": s["kind"], "evidence": json.loads(s["evidence"])})
    return [
        {
            "key": r["key"],
            "received_at": r["received_at"],
            "model_version": r["model_version"],
            "latency_ms": r["latency_ms"],
            "sampled": bool(r["sampled"]),
            "trace": db.payload(r),
            "signals": signals.get(r["key"], []),
            "judgment": json.loads(r["judgment"]) if r["judgment"] else None,
        }
        for r in rows
    ]


def label(key, answer, reviewer):
    rows = db.rows("select pack_id, sealed_overlap from traces where key = ?", (key,))
    if not rows:
        raise KeyError("Trace not found")
    if rows[0]["sealed_overlap"]:
        raise ValueError("Sealed questions are never labeled for training")
    pack = packs.get(rows[0]["pack_id"])
    production.parse_labels(
        json.dumps({"id": key, "answer": answer}), None if pack.id == packs.DEFAULT else pack
    )
    db.execute(
        [
            (
                "insert or replace into labels (key, answer, reviewer, created_at) values (?, ?, ?, ?)",
                (key, answer, reviewer[:80], store.now()),
            )
        ]
    )
    return {"key": key, "labeled": True}


def export(pack_id, name, limit=production.MAX_TRACES):
    """Recent traces and their reviewer labels as an ordinary production scan for the lab."""
    pack = packs.get(pack_id)
    rows = db.rows(
        "select t.key, t.payload, l.answer from traces t left join labels l on l.key = t.key"
        " where t.pack_id = ? and t.sealed_overlap = 0 order by l.key is null, t.received_at desc limit ?",
        (pack.id, limit),
    )
    if not any(r["answer"] for r in rows):
        raise ValueError("Label at least one failing trace before sending it to the lab")
    keep = ("prompt", "output", "finish_reason", "feedback", "timestamp")
    traces_text = "".join(
        json.dumps({"id": r["key"], **{k: v for k, v in json.loads(r["payload"]).items() if k in keep}})
        + "\n"
        for r in rows
    )
    labels_text = "".join(
        json.dumps({"id": r["key"], "answer": r["answer"]}) + "\n" for r in rows if r["answer"]
    )
    report = production.save_scan(
        name, traces_text, labels_text, source="overseer", pack=None if pack.id == packs.DEFAULT else pack
    )
    return production.summary(report)
