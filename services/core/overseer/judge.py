"""Tier two: a small judge model under a hard daily budget, on a sample of traces only.

Which traces are judged: every suspicious trace (a soft tier-one signal) and a deterministic
random sample (hash of the trace and the day). The random sample is the only unbiased estimate
of the failure rate, so statistics report it separately with an interval.

Cost control: each call reserves its worst-case cost before it is sent and settles to the
measured usage after. When a reservation would cross the daily dollar or call cap, nothing is
sent (fail closed). Identical inputs are answered from a cache. A judge never proposes a
corrected answer: only reviewers write labels.
"""

import hashlib
import json
import os
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from services.core import store
from services.core.overseer import db

RUBRIC_VERSION = "overseer-judge-v1"
MAX_TOKENS = 300
PROMPT = """You review one answer a deployed model gave in production. Do not answer the question
yourself and do not write a corrected answer. Task: {task}. Judge whether the answer is usable.
If CONTEXT is given, say whether the answer is grounded in it; claims absent from the context
count as hallucination. Reply with JSON only:
{{"verdict": "ok"|"fail"|"unsure", "grounded": "yes"|"partial"|"no"|"n_a",
 "hallucination": true|false|null, "refusal": true|false, "format_ok": true|false,
 "failure_kinds": [short snake_case kinds], "confidence": 0..1, "rationale": "under 300 characters"}}

QUESTION:
{prompt}

CONTEXT:
{context}

ANSWER:
{output}"""


class Verdict(BaseModel):
    model_config = ConfigDict(extra="ignore")
    verdict: Literal["ok", "fail", "unsure"]
    grounded: Literal["yes", "partial", "no", "n_a"] = "n_a"
    hallucination: bool | None = None
    refusal: bool = False
    format_ok: bool = True
    failure_kinds: list[str] = Field(default_factory=list, max_length=8)
    confidence: float = Field(default=0.5, ge=0, le=1)
    rationale: str = Field(default="", max_length=300)


def settings():
    env = os.environ.get
    return {
        "url": env("FORGE_OVERSEER_JUDGE_URL"),
        "model": env("FORGE_OVERSEER_JUDGE_MODEL"),
        "key_env": env("FORGE_OVERSEER_JUDGE_KEY_ENV"),
        # A hosted judge sees production text; the operator must opt in explicitly.
        "send_traces": env("FORGE_OVERSEER_SEND_TRACES") == "1",
        "daily_usd": float(env("FORGE_OVERSEER_DAILY_USD", "1.00")),
        "daily_calls": int(env("FORGE_OVERSEER_DAILY_CALLS", "200")),
        "sample_rate": float(env("FORGE_OVERSEER_SAMPLE_RATE", "0.02")),
        "price_in_per_m": float(env("FORGE_OVERSEER_PRICE_IN_PER_M", "0")),
        "price_out_per_m": float(env("FORGE_OVERSEER_PRICE_OUT_PER_M", "0")),
    }


def today():
    return datetime.now(UTC).strftime("%Y-%m-%d")


def sampled(key, day, rate):
    return int(hashlib.sha256(f"{day}:{key}".encode()).hexdigest(), 16) % 10_000 < rate * 10_000


def available(config=None):
    """(ready, reason). Ready means a judge is configured and allowed to see traces."""
    from forge_training.distill import is_loopback

    config = config or settings()
    if not (config["url"] and config["model"]):
        return False, "No judge configured (FORGE_OVERSEER_JUDGE_URL and FORGE_OVERSEER_JUDGE_MODEL)"
    if not is_loopback(config["url"]) and not config["send_traces"]:
        return False, "A hosted judge needs FORGE_OVERSEER_SEND_TRACES=1 before it may see traces"
    return True, None


def spend(day=None):
    found = db.rows("select usd, calls from spend where day = ?", (day or today(),))
    return found[0] if found else {"usd": 0.0, "calls": 0}


def reserve(estimate, config):
    """Atomically claim budget for one call, or return False without sending anything."""
    day = today()
    with db.LOCK:
        connection = db.connect()
        try:
            connection.execute("begin immediate")
            row = connection.execute("select usd, calls from spend where day = ?", (day,)).fetchone()
            usd, calls = (row["usd"], row["calls"]) if row else (0.0, 0)
            if calls + 1 > config["daily_calls"] or usd + estimate > config["daily_usd"]:
                connection.execute("rollback")
                return False
            connection.execute(
                "insert into spend (day, usd, calls) values (?, ?, ?) on conflict(day) do update"
                " set usd = excluded.usd, calls = excluded.calls",
                (day, usd + estimate, calls + 1),
            )
            connection.execute("commit")
            return True
        finally:
            connection.close()


def settle(estimate, actual):
    db.execute([("update spend set usd = max(0, usd + ?) where day = ?", (actual - estimate, today()))])


def cost(prompt_tokens, completion_tokens, config):
    return (prompt_tokens * config["price_in_per_m"] + completion_tokens * config["price_out_per_m"]) / 1e6


def judge(key, trace, pack, config=None, client=None):
    """A verdict for one trace, from cache or one budgeted call. None when the budget is spent."""
    from forge_training.distill import Sampling, Teacher, TeacherClient

    config = config or settings()
    text = PROMPT.format(
        task=pack.display.name,
        prompt=trace["prompt"][:8000],
        context=(trace.get("context") or "(none)")[:8000],
        output=trace["output"][:8000],
    )
    cache_key = hashlib.sha256(json.dumps([config["model"], RUBRIC_VERSION, text]).encode()).hexdigest()
    cached = db.rows("select result from judge_cache where cache_key = ?", (cache_key,))
    usd = 0.0
    if cached:
        result = json.loads(cached[0]["result"])
    else:
        estimate = cost(len(text) // 3 + 50, MAX_TOKENS, config)
        if not reserve(estimate, config):
            return None
        client = client or TeacherClient(
            Teacher(
                kind="openai_compatible",
                base_url=config["url"],
                model=config["model"],
                api_key_env=config["key_env"],
                teacher_terms="Overseer judge. Its verdicts are never used as training answers.",
                extra_body={"response_format": {"type": "json_object"}},
                timeout_s=60,
                max_retries=1,
            )
        )
        try:
            reply = client.complete(
                [{"role": "user", "content": text}], Sampling(n=1, temperature=0, max_tokens=MAX_TOKENS), 0
            )
        except RuntimeError:
            settle(estimate, 0)
            raise
        usd = cost(reply["prompt_tokens"], reply["completion_tokens"], config)
        settle(estimate, usd)
        try:
            body = reply["text"]
            result = Verdict.model_validate_json(body[body.find("{") : body.rfind("}") + 1]).model_dump()
        except (ValidationError, ValueError):
            result = {"verdict": "unsure", "error": "judge_unparseable"}
        if trace.get("context") is None:
            result["hallucination"] = None  # Without context the judge cannot check grounding.
        db.execute(
            [
                (
                    "insert or ignore into judge_cache (cache_key, result) values (?, ?)",
                    (cache_key, json.dumps(result)),
                )
            ]
        )
    db.execute(
        [
            (
                "insert or replace into judgments (key, cache_key, verdict, result, usd, model, created_at)"
                " values (?, ?, ?, ?, ?, ?, ?)",
                (key, cache_key, result["verdict"], json.dumps(result), usd, config["model"], store.now()),
            )
        ]
    )
    return result
