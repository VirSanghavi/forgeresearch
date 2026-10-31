"""Teacher distillation: verified teacher-written solutions as a training dataset.

Any OpenAI-compatible chat endpoint can teach (Ollama, vLLM, LM Studio, llama.cpp, hosted APIs).
For every training question the teacher writes candidate solutions; only candidates the task's
checker accepts against the reference answer are kept. The output is an ordinary training-core
dataset plus a provenance record (teacher, sampling, checker, separation, terms of use, budget)
and every raw sample, so a reviewer can re-verify the dataset from bytes.

Separation is checked before the first request: no development, evaluation or sealed question is
ever sent to a teacher. Credentials are read from a named environment variable and never written.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .artifacts import file_sha256
from .config import DatasetManifest, DatasetSpec, Example
from .data import load_dataset, question_key, write_dataset
from .tasks import TaskSpec

TOOL_VERSION = "forge-distill-v1"
THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Teacher(Contract):
    kind: Literal["ollama", "vllm", "lmstudio", "llama_cpp", "openai_compatible"]
    # The OpenAI-compatible root, e.g. http://127.0.0.1:11434/v1
    base_url: str = Field(pattern=r"^https?://[^\s]+/v1/?$")
    model: str = Field(min_length=1, max_length=200)
    api_key_env: str | None = Field(default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    # Recorded verbatim: why this teacher's outputs may be used to train another model.
    teacher_terms: str = Field(min_length=20, max_length=2000)
    extra_body: dict = Field(default_factory=dict)
    strip_reasoning: bool = True
    system_prompt: str | None = Field(default=None, max_length=4000)
    timeout_s: float = Field(default=180, gt=0, le=1800)
    max_retries: int = Field(default=3, ge=0, le=10)
    unload_after: bool = True


class Sampling(Contract):
    n: int = Field(default=4, ge=1, le=16)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.95, gt=0, le=1)
    max_tokens: int = Field(default=768, ge=16, le=8192)
    seed: int = Field(default=0, ge=0)


class Budget(Contract):
    max_prompts: int | None = Field(default=None, ge=1)
    max_requests: int = Field(default=5000, ge=1)
    max_completion_tokens: int = Field(default=2_000_000, ge=1)
    max_seconds: int = Field(default=4 * 3600, ge=1)
    max_usd: float | None = Field(default=None, ge=0)
    price_in_per_m: float = Field(default=0, ge=0)
    price_out_per_m: float = Field(default=0, ge=0)


class DistillConfig(Contract):
    teacher: Teacher
    task: TaskSpec
    prompts: DatasetSpec
    exclude: list[DatasetSpec] = Field(default_factory=list)
    # Files of sha256(question_key) lines, for sets whose text must not be read (sealed data).
    exclude_keys: list[str] = Field(default_factory=list)
    sampling: Sampling = Field(default_factory=Sampling)
    selection: Literal["shortest_correct", "first_correct"] = "shortest_correct"
    verification: Literal["reference", "format_only"] = "reference"
    budget: Budget = Field(default_factory=Budget)
    output_dir: str


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(value) -> str:
    return hashlib.sha256(
        value if isinstance(value, bytes) else canonical(value).encode()
    ).hexdigest()


def key_hash(question: str) -> str:
    return hashlib.sha256(question_key(question).encode()).hexdigest()


def load_config(path: Path) -> DistillConfig:
    """Relative paths resolve against the configuration file, like forge-train."""
    path = Path(path).resolve()
    raw = json.loads(path.read_text())

    def resolve(value):
        return str((path.parent / value).resolve())

    raw["prompts"] = {k: resolve(v) for k, v in raw["prompts"].items()}
    raw["exclude"] = [{k: resolve(v) for k, v in spec.items()} for spec in raw.get("exclude", [])]
    raw["exclude_keys"] = [resolve(v) for v in raw.get("exclude_keys", [])]
    raw["output_dir"] = resolve(raw["output_dir"])
    return DistillConfig.model_validate(raw)


def _any_rows(spec: DatasetSpec):
    """Rows of a dataset of any purpose, checksum-verified against its manifest."""
    manifest = DatasetManifest.model_validate_json(Path(spec.manifest).read_text())
    if file_sha256(Path(spec.records)) != manifest.content_sha256:
        raise ValueError(f"Dataset checksum mismatch: {spec.records}")
    rows = [
        json.loads(line) for line in Path(spec.records).read_text().splitlines() if line.strip()
    ]
    return rows, manifest


def separation(config: DistillConfig, prompts) -> dict:
    """Refuse before any request when a prompt is also in an excluded set."""
    excluded, sources = set(), []
    for spec in config.exclude:
        rows, manifest = _any_rows(spec)
        excluded |= {key_hash(r["question"]) for r in rows}
        sources.append(
            {
                "kind": "dataset",
                "dataset_id": manifest.dataset_id,
                "sha256": manifest.content_sha256,
            }
        )
    for path in config.exclude_keys:
        keys = {line.strip() for line in Path(path).read_text().splitlines() if line.strip()}
        excluded |= keys
        sources.append(
            {"kind": "question_keys", "sha256": file_sha256(Path(path)), "count": len(keys)}
        )
    overlap = [p.id for p in prompts if key_hash(p.question) in excluded]
    if overlap:
        raise ValueError(
            f"{len(overlap)} prompts are in an excluded set (first: {overlap[0]}); refusing"
        )
    return {"verified": True, "excluded_sources": sources, "excluded_questions": len(excluded)}


# ---- teacher client -----------------------------------------------------------------------------


def public_url(url: str) -> str:
    """The endpoint without credentials or query strings, safe to record."""
    parts = urllib.parse.urlsplit(url)
    host = parts.hostname or ""
    netloc = f"{host}:{parts.port}" if parts.port else host
    return urllib.parse.urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def is_loopback(url: str) -> bool:
    host = urllib.parse.urlsplit(url).hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class TeacherHTTPError(RuntimeError):
    def __init__(self, code, detail):
        super().__init__(f"Teacher returned HTTP {code}")
        self.code = code
        self.detail = detail


# Reasoning models (OpenAI o-series, gpt-5 and later, also on Azure) refuse these sampling fields.
UNSUPPORTED = re.compile(r"Unsupported (parameter|value): '(max_tokens|temperature|top_p)'")


class TeacherClient:
    def __init__(self, teacher: Teacher):
        self.teacher = teacher
        # Set after the endpoint refuses sampling fields; the model's own defaults are used.
        self.model_defaults = False
        self.base = teacher.base_url.rstrip("/")
        self.root = self.base[: -len("/v1")]
        key = os.environ.get(teacher.api_key_env, "") if teacher.api_key_env else ""
        if teacher.api_key_env and not key:
            raise ValueError(f"Environment variable {teacher.api_key_env} is not set")
        self._headers = {"Content-Type": "application/json"} | (
            {"Authorization": f"Bearer {key}"} if key else {}
        )

    def _request(self, url, body=None, method=None):
        data = json.dumps(body).encode() if body is not None else None
        delay = 1.0
        for attempt in range(self.teacher.max_retries + 1):
            request = urllib.request.Request(url, data=data, headers=self._headers, method=method)
            try:
                with urllib.request.urlopen(request, timeout=self.teacher.timeout_s) as reply:
                    return json.loads(reply.read() or b"{}")
            except urllib.error.HTTPError as error:
                retry = error.code == 429 or error.code >= 500
                if not retry or attempt == self.teacher.max_retries:
                    detail = error.read()[:2000].decode("utf-8", "replace") if error.fp else ""
                    raise TeacherHTTPError(error.code, detail) from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                if attempt == self.teacher.max_retries:
                    raise RuntimeError(f"Teacher unreachable: {type(error).__name__}") from None
            time.sleep(delay)
            delay = min(delay * 2, 30)

    def models(self):
        return [m.get("id") for m in self._request(f"{self.base}/models").get("data", [])]

    def digest(self):
        if self.teacher.kind != "ollama":
            return None
        for model in self._request(f"{self.root}/api/tags").get("models", []):
            if model.get("name") == self.teacher.model or model.get("model") == self.teacher.model:
                return model.get("digest")
        return None

    def complete(self, messages, sampling: Sampling, seed: int):
        body = {
            "model": self.teacher.model,
            "messages": messages,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": sampling.max_tokens,
            "seed": seed,
            "stream": False,
        } | self.teacher.extra_body
        if self.model_defaults:
            body = self._model_defaults(body)
        try:
            reply = self._request(f"{self.base}/chat/completions", body)
        except TeacherHTTPError as error:
            if self.model_defaults or error.code != 400 or not UNSUPPORTED.search(error.detail):
                raise
            # The token cap still applies; temperature and top_p fall back to the model's defaults,
            # and every result says so.
            self.model_defaults = True
            body = self._model_defaults(body)
            reply = self._request(f"{self.base}/chat/completions", body)
        choice = reply["choices"][0]
        message = choice.get("message", {})
        content = message.get("content") or ""
        reasoning = bool(
            message.get("reasoning") or message.get("reasoning_content") or THINK.search(content)
        )
        if self.teacher.strip_reasoning:
            content = THINK.sub("", content).strip()
        usage = reply.get("usage") or {}
        return {
            "text": content,
            "finish_reason": choice.get("finish_reason"),
            "reasoning_returned": reasoning,
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "system_fingerprint": reply.get("system_fingerprint"),
            **(
                {"sampling": "model defaults (temperature, top_p refused)"}
                if self.model_defaults
                else {}
            ),
        }

    @staticmethod
    def _model_defaults(body):
        cap = body.get("max_tokens")
        body = {k: v for k, v in body.items() if k not in ("temperature", "top_p", "max_tokens")}
        return body | ({"max_completion_tokens": cap} if cap else {})

    def unload(self, wait_s=30):
        """Ask Ollama to drop the model from memory and wait until it has."""
        if self.teacher.kind != "ollama":
            return None
        self._request(f"{self.root}/api/generate", {"model": self.teacher.model, "keep_alive": 0})
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            loaded = self._request(f"{self.root}/api/ps").get("models", [])
            if not any(
                m.get("name") == self.teacher.model or m.get("model") == self.teacher.model
                for m in loaded
            ):
                return True
            time.sleep(0.5)
        return False


def messages_for(config: DistillConfig, question: str):
    user = {"role": "user", "content": f"{config.task.instruction}\n\n{question}"}
    system = (
        [{"role": "system", "content": config.teacher.system_prompt}]
        if config.teacher.system_prompt
        else []
    )
    return [*system, user]


def fingerprint(config: DistillConfig, client: TeacherClient) -> dict:
    return {
        "kind": config.teacher.kind,
        "endpoint": public_url(config.teacher.base_url),
        "model": config.teacher.model,
        "digest": client.digest(),
        "extra_body": config.teacher.extra_body,
        "system_prompt_sha256": sha(config.teacher.system_prompt or ""),
    }


def probe(config: DistillConfig) -> dict:
    return probe_teacher(config.teacher)


def probe_teacher(teacher: Teacher) -> dict:
    """List models, read the digest and send one tiny request. Nothing is written."""
    client = TeacherClient(teacher)
    models = client.models()
    sample = client.complete(
        [{"role": "user", "content": "Reply with the single word OK."}],
        Sampling(n=1, max_tokens=32),
        0,
    )
    return {
        "endpoint": public_url(teacher.base_url),
        "model_listed": teacher.model in models,
        "models": models[:50],
        "digest": client.digest(),
        "reply": sample["text"][:200],
        "reasoning_returned": sample["reasoning_returned"],
        "finish_reason": sample["finish_reason"],
    }


# ---- generation ---------------------------------------------------------------------------------


def _load_cache(path: Path) -> dict:
    cache = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                cache[entry["key"]] = entry
    return cache


def run(config: DistillConfig, on_progress=None) -> dict:
    """Generate, verify and write the dataset. Rerunning resumes from the request cache."""
    from .checkers import Checker

    started = time.monotonic()
    output = Path(config.output_dir)
    previous = output / "distillation.json"
    if previous.exists() and json.loads(previous.read_text())["status"] == "complete":
        raise ValueError(f"{output} already holds a finished distillation; choose a new output_dir")
    prompts, source = load_dataset(config.prompts, "train")
    if config.budget.max_prompts:
        prompts = prompts[: config.budget.max_prompts]
    separated = separation(config, prompts)  # Before any network call.
    checker = Checker(config.task.checker)
    lock, inherited = None, os.environ.get("FORGE_MODEL_LOCK_HELD")
    if is_loopback(config.teacher.base_url):
        from .locking import acquire_model_lock

        # A local teacher competes with training for the same memory.
        lock = acquire_model_lock("distill", wait=False)
    client = TeacherClient(config.teacher)
    teacher = fingerprint(config, client)
    output.mkdir(parents=True, exist_ok=True)
    cache_path = output / "cache" / "requests.jsonl"
    cache_path.parent.mkdir(exist_ok=True)
    cache = _load_cache(cache_path)
    spent = {"requests": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0}
    reasons: dict[str, int] = {}
    samples, kept = [], []
    status = "complete"
    prices = config.budget

    def exhausted():
        return (
            spent["requests"] >= prices.max_requests
            or spent["completion_tokens"] >= prices.max_completion_tokens
            or time.monotonic() - started >= prices.max_seconds
            or (prices.max_usd is not None and spent["usd"] >= prices.max_usd)
        )

    with cache_path.open("a", encoding="utf-8") as journal:
        for index, prompt in enumerate(prompts):
            candidates = []
            for k in range(config.sampling.n):
                key = sha([teacher, sha(prompt.question), config.sampling.model_dump(), k])
                if key in cache:
                    reply = cache[key]["reply"]
                    spent["cached"] += 1
                else:
                    if exhausted():
                        status = "budget_exhausted"
                        break
                    reply = client.complete(
                        messages_for(config, prompt.question),
                        config.sampling,
                        config.sampling.seed + k,
                    )
                    spent["requests"] += 1
                    journal.write(
                        canonical({"key": key, "prompt_id": prompt.id, "k": k, "reply": reply})
                        + "\n"
                    )
                    journal.flush()
                    os.fsync(journal.fileno())
                spent["prompt_tokens"] += reply["prompt_tokens"]
                spent["completion_tokens"] += reply["completion_tokens"]
                spent["usd"] = round(
                    (
                        spent["prompt_tokens"] * prices.price_in_per_m
                        + spent["completion_tokens"] * prices.price_out_per_m
                    )
                    / 1e6,
                    6,
                )
                if not reply["text"]:
                    reason = "empty"
                elif reply["finish_reason"] == "length":
                    reason = "truncated"
                elif config.verification == "reference":
                    reason = checker.score(reply["text"], prompt.answer)["reason"]
                else:
                    reason = (
                        "correct" if checker.format_ok(reply["text"]) else "invalid_answer_format"
                    )
                accepted = reason == "correct"
                reasons[reason] = reasons.get(reason, 0) + 1
                sample = {
                    "prompt_id": prompt.id,
                    "k": k,
                    "accepted": accepted,
                    "reason": reason,
                    **reply,
                }
                samples.append(sample)
                candidates.append(sample)
                if accepted and config.selection == "first_correct":
                    break
            correct = [c for c in candidates if c["accepted"]]
            if correct:
                best = (
                    min(correct, key=lambda c: (len(c["text"]), c["k"]))
                    if config.selection == "shortest_correct"
                    else correct[0]
                )
                kept.append(Example(id=prompt.id, question=prompt.question, answer=best["text"]))
            if on_progress:
                on_progress({"done": index + 1, "total": len(prompts), "kept": len(kept), **spent})
            if status != "complete":
                break
    samples_bytes = "".join(canonical(s) + "\n" for s in samples).encode()
    (output / "samples.jsonl").write_bytes(samples_bytes)
    train_bytes = "".join(row.model_dump_json() + "\n" for row in kept).encode()
    unloaded = client.unload() if config.teacher.unload_after else None
    if lock is not None:
        lock.close()
        # The lock marks this process as holding it; release that mark with the lock.
        if inherited is None:
            os.environ.pop("FORGE_MODEL_LOCK_HELD", None)
    record = {
        "tool_version": TOOL_VERSION,
        "status": status,
        "teacher": teacher,
        "teacher_terms": config.teacher.teacher_terms,
        "teacher_unloaded": unloaded,
        "reasoning_returned": any(s["reasoning_returned"] for s in samples),
        "system_fingerprints": sorted(
            {s["system_fingerprint"] for s in samples if s["system_fingerprint"]}
        ),
        "sampling": config.sampling.model_dump(),
        "selection": config.selection,
        "verification": config.verification,
        "task": {
            "task_id": config.task.task_id,
            "prompt_version": config.task.prompt_version,
            "prompt_template_sha256": sha(config.task.instruction),
            "checker_id": checker.checker_id,
        },
        "source": {
            "dataset_id": source.dataset_id,
            "content_sha256": source.content_sha256,
            "license": source.license,
        },
        "separation": separated,
        "counts": {
            "prompts": len(prompts),
            "kept": len(kept),
            "samples": len(samples),
            "reasons": dict(sorted(reasons.items())),
        },
        "usage": spent,
        "seconds": round(time.monotonic() - started, 2),
        "samples_sha256": sha(samples_bytes),
        "train_sha256": sha(train_bytes),
    }
    record_bytes = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    (output / "distillation.json").write_bytes(record_bytes)
    # A resumed run replaces its partial output.
    for stale in ("train.jsonl", "train.manifest.json"):
        (output / stale).unlink(missing_ok=True)
    if kept:
        write_dataset(
            output,
            "train",
            kept,
            source=f"distill:{config.teacher.model}",
            revision=sha(record_bytes),
            source_split="synthetic",
            license_name=f"{source.license}; teacher outputs: see distillation.json teacher_terms",
            purpose="train",
        )
    return record


def verify(output_dir) -> dict:
    """Re-derive every hash and check each kept example against an accepted raw sample."""
    output = Path(output_dir)
    record_bytes = (output / "distillation.json").read_bytes()
    record = json.loads(record_bytes)
    samples_bytes = (output / "samples.jsonl").read_bytes()
    if sha(samples_bytes) != record["samples_sha256"]:
        raise ValueError("samples.jsonl does not match its recorded hash")
    if not record["teacher_terms"] or len(record["teacher_terms"]) < 20:
        raise ValueError("Teacher terms of use are missing")
    if not record["separation"]["verified"]:
        raise ValueError("Separation was not verified")
    kept = 0
    if record["counts"]["kept"]:
        manifest = DatasetManifest.model_validate_json((output / "train.manifest.json").read_text())
        if manifest.source_revision != sha(record_bytes) or manifest.source_split != "synthetic":
            raise ValueError("Dataset manifest is not bound to this distillation record")
        train_bytes = (output / "train.jsonl").read_bytes()
        if (
            sha(train_bytes) != record["train_sha256"]
            or file_sha256(output / "train.jsonl") != manifest.content_sha256
        ):
            raise ValueError("train.jsonl does not match its recorded hash")
        accepted = {}
        for line in samples_bytes.decode().splitlines():
            sample = json.loads(line)
            if sample["accepted"]:
                accepted.setdefault(sample["prompt_id"], set()).add(sample["text"])
        for line in train_bytes.decode().splitlines():
            row = json.loads(line)
            if row["answer"] not in accepted.get(row["id"], set()):
                raise ValueError(f"Kept example {row['id']} is not an accepted teacher sample")
            kept += 1
    return {
        "status": "verified",
        "kept": kept,
        "provenance_sha256": sha(record_bytes),
        "teacher": record["teacher"],
        "checker_id": record["task"]["checker_id"],
        "verification": record["verification"],
        "teacher_terms_present": True,
        "separation_verified": True,
    }
