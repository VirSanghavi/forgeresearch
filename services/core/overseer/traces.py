"""Trace intake (schema v2) and tier-one detectors: free, deterministic, run on every trace."""

import hashlib
import json
import re
import unicodedata
from collections import Counter

from forge_training.data import question_key

from services.core import packs, production, store
from services.core.overseer import db

MAX_BATCH = 500
HARD = {"empty", "truncated", "format_failure", "missing_final_answer", "arithmetic_slip", "user_reported"}
SOFT = {"refusal", "repetition", "language_mismatch", "latency_outlier"}
LABELS = {
    "empty": "The model returned nothing",
    "refusal": "The model declined to answer",
    "repetition": "The answer loops on the same phrase",
    "language_mismatch": "The answer is in a different script than the question",
    "latency_outlier": "Much slower than this deployment's usual latency",
    **production.DETECTORS,
    **production.GENERIC_DETECTORS,
}
REFUSAL = re.compile(
    r"\b(?:I can(?:'|no)?t (?:help|assist|provide|do that)|I'?m (?:sorry|unable)|I am unable|"
    r"I cannot (?:help|assist|provide|comply)|as an AI(?: language model)?)\b",
    re.IGNORECASE,
)
_sealed_cache = {}


def normalize(row):
    """A v1 production trace plus the v2 fields the overseer keeps. Raises ValueError."""
    trace = production._trace(row)
    for name, limit in (("model_version", 100), ("deployment", 100)):
        if isinstance(row.get(name), str):
            trace[name] = row[name][:limit]
    if isinstance(row.get("latency_ms"), int | float) and row["latency_ms"] >= 0:
        trace["latency_ms"] = float(row["latency_ms"])
    tokens = row.get("tokens")
    if isinstance(tokens, dict):
        trace["tokens"] = {
            k: int(tokens[k]) for k in ("prompt", "completion") if isinstance(tokens.get(k), int)
        }
    if isinstance(row.get("context"), str):
        if len(row["context"]) > 32000:
            raise ValueError("context too long")
        trace["context"] = row["context"]
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        if len(metadata) > 16:
            raise ValueError("at most 16 metadata keys")
        trace["metadata"] = {
            str(k)[:64]: v
            for k, v in metadata.items()
            if isinstance(v, str | int | float | bool) or v is None
        }
    return trace


def sealed_keys(pack):
    if pack.id not in _sealed_cache:
        from services.core import distill

        path = distill.sealed_keys(pack)
        _sealed_cache[pack.id] = {line.strip() for line in path.read_text().splitlines() if line.strip()}
    return _sealed_cache[pack.id]


def ingest(rows, source, pack_id, received_at=None):
    """Store new traces. Duplicates (same source and id) are ignored, not overwritten."""
    pack = packs.get(pack_id)
    if len(rows) > MAX_BATCH:
        raise ValueError(f"At most {MAX_BATCH} traces per batch")
    sealed = sealed_keys(pack)
    statements, rejected, keys = [], [], set()
    for index, row in enumerate(rows):
        try:
            trace = normalize(row)
        except (ValueError, TypeError) as error:
            rejected.append({"index": index, "reason": str(error)[:120]})
            continue
        key = f"{source}:{trace['id']}"
        if key in keys:
            continue
        keys.add(key)
        overlap = hashlib.sha256(question_key(trace["prompt"]).encode()).hexdigest() in sealed
        # A sealed question is kept only as a flag: its text never enters overseer storage.
        stored = {"id": trace["id"], "sealed_overlap": True} if overlap else trace
        statements.append(
            (
                "insert or ignore into traces (key, source, trace_id, pack_id, received_at, model_version,"
                " latency_ms, completion_tokens, finish_reason, sealed_overlap, payload, checked)"
                " values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    key,
                    source,
                    trace["id"],
                    pack.id,
                    received_at or store.now(),
                    trace.get("model_version"),
                    trace.get("latency_ms"),
                    trace.get("tokens", {}).get("completion"),
                    trace.get("finish_reason"),
                    int(overlap),
                    json.dumps(stored),
                    int(overlap),
                ),
            )
        )
    before = db.rows("select count(*) as n from traces")[0]["n"]
    db.execute(statements)
    accepted = db.rows("select count(*) as n from traces")[0]["n"] - before
    return {"accepted": accepted, "duplicates": len(statements) - accepted, "rejected": rejected}


def _script(text):
    counts = Counter()
    for char in text:
        if char.isalpha():
            name = unicodedata.name(char, "")
            counts[name.split(" ")[0] if name else "OTHER"] += 1
    return counts.most_common(1)[0][0] if sum(counts.values()) >= 12 else None


def repetition(text):
    words = text.split()
    if len(words) < 40:
        return None
    grams = Counter(tuple(words[i : i + 4]) for i in range(len(words) - 3))
    gram, count = grams.most_common(1)[0]
    share = count * 4 / len(words)
    return (
        {"phrase": " ".join(gram)[:80], "repeats": count, "share": round(share, 2)}
        if count >= 5 and share >= 0.3
        else None
    )


def detect(trace, pack, latency_p99=None, slo_ms=None):
    """Signals for one trace: the pack's own detectors plus generic ones."""
    found = production.detect(trace, None if pack.id == packs.DEFAULT else pack)
    output = trace["output"].strip()
    if not output:
        found.append({"kind": "empty", "evidence": {}})
    elif REFUSAL.search(output[:600]):
        found.append({"kind": "refusal", "evidence": {"text": REFUSAL.search(output[:600]).group(0)}})
    if looped := repetition(output):
        found.append({"kind": "repetition", "evidence": looped})
    asked, answered = _script(trace["prompt"]), _script(output)
    if asked and answered and asked != answered:
        found.append({"kind": "language_mismatch", "evidence": {"prompt": asked, "output": answered}})
    latency = trace.get("latency_ms")
    limit = slo_ms or (latency_p99 * 1.5 if latency_p99 else None)
    if latency is not None and limit and latency > limit:
        found.append(
            {"kind": "latency_outlier", "evidence": {"latency_ms": latency, "limit_ms": round(limit, 1)}}
        )
    unique = {}
    for signal in found:
        unique.setdefault(signal["kind"], signal)
    return list(unique.values())
