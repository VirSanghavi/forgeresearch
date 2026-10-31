"""Declarative answer checkers: an extractor reads an answer, a comparator judges it.

A checker never raises on model output. Malformed output scores as incorrect with a reason.
References are validated up front with `check_reference`, so a mapping error surfaces when a
dataset is registered rather than as silent zero scores.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
import unicodedata
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .scoring import extract_answer

# Patterns come from the task's operator and are trusted. Capping the text they see bounds
# polynomial backtracking on long outputs; it cannot rescue an exponential pattern.
MAX_MATCH_CHARS = 8000
NUMBER = re.compile(r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)")


class Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Extractor(Spec):
    kind: Literal["final_marker", "regex", "last_line", "whole", "json"]
    marker: str = Field(default="####", min_length=1, max_length=32)
    # forge_numeric_v1 is the training-core GSM8K grammar, kept byte-identical.
    grammar: Literal["forge_numeric_v1", "number", "text"] = "text"
    pattern: str | None = Field(default=None, max_length=1000)
    group: int = Field(default=1, ge=0, le=20)

    @model_validator(mode="after")
    def consistent(self):
        if self.kind == "regex":
            if not self.pattern:
                raise ValueError("A regex extractor needs a pattern")
            if re.compile(self.pattern).groups < self.group:
                raise ValueError("The regex extractor's group does not exist in its pattern")
        if self.grammar == "forge_numeric_v1" and (
            self.kind != "final_marker" or self.marker != "####"
        ):
            raise ValueError("forge_numeric_v1 reads only a final '####' line")
        return self


class FieldCheck(Spec):
    path: str = Field(pattern=r"^(/[^/]*)*$")
    compare: Literal["exact", "casefold", "numeric"] = "exact"


class Comparator(Spec):
    kind: Literal["numeric", "exact", "regex_match", "json_fields", "python"]
    abs_tol: float = Field(default=0, ge=0)
    rel_tol: float = Field(default=0, ge=0, le=1)
    normalize: tuple[Literal["strip", "casefold", "whitespace", "nfkc"], ...] = ("strip",)
    json_schema: dict | None = None
    fields: tuple[FieldCheck, ...] = ()
    path: str | None = None
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    timeout_s: float = Field(default=5, gt=0, le=60)

    @model_validator(mode="after")
    def consistent(self):
        if self.kind == "python" and not (self.path and self.sha256):
            raise ValueError("A python comparator needs a path and its sha256")
        if self.kind != "json_fields" and (self.json_schema or self.fields):
            raise ValueError("Schemas and field checks belong to the json_fields comparator")
        return self


class CheckerSpec(Spec):
    extract: Extractor
    compare: Comparator
    # "same": the reference goes through the extractor too (GSM8K references end in '#### N').
    reference: Literal["same", "whole"] = "same"

    @model_validator(mode="after")
    def compatible(self):
        if self.compare.kind == "json_fields" and self.extract.kind != "json":
            raise ValueError("The json_fields comparator needs the json extractor")
        if self.extract.kind == "json" and self.compare.kind not in ("json_fields", "python"):
            raise ValueError("The json extractor pairs with json_fields or python")
        return self


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _number(text: str) -> Decimal | None:
    match = NUMBER.fullmatch(text.strip())
    if not match:
        return None
    try:
        return Decimal(match.group(0).replace(",", ""))
    except InvalidOperation:
        return None


def _json(text: str):
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*\n(.*?)\n```", text[: MAX_MATCH_CHARS * 4], re.DOTALL)
    candidates = [text, fenced.group(1) if fenced else None]
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if starts:
        start = min(starts)
        end = max(text.rfind("}"), text.rfind("]"))
        if end > start:
            candidates.append(text[start : end + 1])
    for candidate in candidates:
        if candidate:
            try:
                return json.loads(candidate)
            except (json.JSONDecodeError, RecursionError):
                continue
    return None


def shown(value) -> str:
    return "" if value is None else value if isinstance(value, str) else canonical(value)


def _pointer(value, path: str):
    for part in path.split("/")[1:]:
        key = part.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            value = value[int(key)]
        elif isinstance(value, dict):
            value = value[key]
        else:
            raise KeyError(path)
    return value


class Checker:
    def __init__(self, spec: CheckerSpec | dict):
        self.spec = spec if isinstance(spec, CheckerSpec) else CheckerSpec.model_validate(spec)
        digest = hashlib.sha256(canonical(self.spec.model_dump(mode="json")).encode()).hexdigest()
        self.checker_id = f"{self.spec.extract.kind}-{self.spec.compare.kind}@{digest[:12]}"
        if self.spec.compare.kind == "python":
            self._verify_python()

    # ---- extraction -------------------------------------------------------------------------

    def extract(self, text: str):
        """The answer read from text: a string, a parsed JSON value, or None."""
        e = self.spec.extract
        if e.kind == "final_marker":
            if e.grammar == "forge_numeric_v1":
                value = extract_answer(text)
                return str(value) if value is not None else None
            pattern = rf"(?:^|\n)[ \t]*{re.escape(e.marker)}[ \t]*(.*?)[ \t\r\n]*$"
            match = re.search(pattern, text[-MAX_MATCH_CHARS:], re.DOTALL)
            if not match or not match.group(1).strip() or "\n" in match.group(1).strip():
                return None
            answer = match.group(1).strip()
        elif e.kind == "regex":
            match = re.search(e.pattern, text[:MAX_MATCH_CHARS])
            if not match or match.group(e.group) is None:
                return None
            answer = match.group(e.group).strip()
        elif e.kind == "last_line":
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            answer = lines[-1] if lines else ""
        elif e.kind == "whole":
            answer = text.strip()
        else:
            return _json(text)
        if not answer:
            return None
        if e.grammar == "number":
            value = _number(answer)
            return str(value) if value is not None else None
        return answer

    def _reference(self, reference: str):
        if self.spec.compare.kind == "regex_match":
            return reference
        if self.spec.reference == "whole":
            return _json(reference) if self.spec.extract.kind == "json" else reference.strip()
        return self.extract(reference)

    # ---- comparison -------------------------------------------------------------------------

    def _normalize(self, text: str) -> str:
        for step in self.spec.compare.normalize:
            if step == "strip":
                text = text.strip()
            elif step == "casefold":
                text = text.casefold()
            elif step == "whitespace":
                text = " ".join(text.split())
            else:
                text = unicodedata.normalize("NFKC", text)
        return text

    def _numeric(self, answer, expected) -> bool:
        try:
            a, b = Decimal(str(answer).replace(",", "")), Decimal(str(expected).replace(",", ""))
        except InvalidOperation:
            return False
        c = self.spec.compare
        if not c.abs_tol and not c.rel_tol:
            return a == b
        return float(abs(a - b)) <= max(c.abs_tol, c.rel_tol * float(abs(b)))

    def _schema_errors(self, value) -> bool:
        schema = self.spec.compare.json_schema
        if not schema:
            return False
        import jsonschema

        return any(True for _ in jsonschema.Draft202012Validator(schema).iter_errors(value))

    def _fields(self, answer, expected) -> dict:
        results = {}
        for check in self.spec.compare.fields:
            try:
                a, b = _pointer(answer, check.path), _pointer(expected, check.path)
            except (KeyError, IndexError, ValueError, TypeError):
                results[check.path] = False
                continue
            if check.compare == "numeric":
                results[check.path] = self._numeric(a, b)
            elif check.compare == "casefold" and isinstance(a, str) and isinstance(b, str):
                results[check.path] = a.strip().casefold() == b.strip().casefold()
            else:
                results[check.path] = canonical(a) == canonical(b)
        return results

    def _verify_python(self):
        c = self.spec.compare
        if hashlib.sha256(Path(c.path).read_bytes()).hexdigest() != c.sha256:
            raise ValueError("The python checker file does not match its pinned sha256")

    def _python(self, output: str, reference: str) -> dict:
        """Operator code in a separate isolated interpreter. Not a security boundary."""
        c = self.spec.compare
        self._verify_python()
        runner = (
            "import importlib.util, json, sys\n"
            "spec = importlib.util.spec_from_file_location('forge_checker', sys.argv[1])\n"
            "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)\n"
            "job = json.loads(sys.stdin.read())\n"
            "result = module.check(job['output'], job['reference'])\n"
            "if isinstance(result, bool): result = {'correct': result}\n"
            "print(json.dumps({'correct': bool(result['correct']),"
            " 'answer': str(result.get('answer', ''))[:2000]}))\n"
        )
        with tempfile.TemporaryDirectory() as folder:
            try:
                done = subprocess.run(
                    [sys.executable, "-I", "-c", runner, str(Path(c.path).resolve())],
                    input=json.dumps({"output": output, "reference": reference}),
                    capture_output=True,
                    text=True,
                    timeout=c.timeout_s,
                    env={},
                    cwd=folder,
                    check=False,
                )
                result = (
                    json.loads(done.stdout.strip().splitlines()[-1])
                    if done.returncode == 0
                    else None
                )
            except (subprocess.TimeoutExpired, json.JSONDecodeError, IndexError):
                result = None
        if result is None:
            return {"answer": "", "correct": False, "reason": "checker_error"}
        reason = "correct" if result["correct"] else "incorrect_answer"
        return {"answer": result["answer"], "correct": result["correct"], "reason": reason}

    # ---- public API -------------------------------------------------------------------------

    def check_reference(self, reference: str) -> None:
        """Raise ValueError when a reference cannot be scored by this checker."""
        kind = self.spec.compare.kind
        if kind == "regex_match":
            try:
                re.compile(reference)
            except re.error as error:
                raise ValueError(f"Reference is not a valid pattern: {error}") from None
            return
        if kind == "python":
            return
        expected = self._reference(reference)
        if expected is None:
            raise ValueError("Reference has no answer this checker can read")
        if (
            kind == "numeric"
            and _number(str(expected)) is None
            and self.spec.extract.grammar != "forge_numeric_v1"
        ):
            raise ValueError("Reference answer is not a number")
        if kind == "json_fields" and self._schema_errors(expected):
            raise ValueError("Reference does not satisfy the task's JSON schema")

    def format_ok(self, output: str) -> bool:
        if self.spec.compare.kind == "python":
            return bool(output.strip())
        answer = self.extract(output)
        if answer is None:
            return False
        return not (self.spec.compare.kind == "json_fields" and self._schema_errors(answer))

    def score(self, output: str, reference: str) -> dict:
        """{answer, expected, correct, reason[, detail]} with string answer/expected fields."""
        kind = self.spec.compare.kind
        if kind == "python":
            return {"expected": reference[:2000], **self._python(output, reference)}
        expected = self._reference(reference)
        answer = self.extract(output)
        if answer is None:
            return {
                "answer": "",
                "expected": shown(expected),
                "correct": False,
                "reason": "invalid_answer_format",
            }
        if kind == "numeric":
            correct = expected is not None and self._numeric(answer, expected)
        elif kind == "exact":
            correct = expected is not None and self._normalize(answer) == self._normalize(expected)
        elif kind == "regex_match":
            try:
                correct = re.fullmatch(expected, answer[:MAX_MATCH_CHARS]) is not None
            except re.error:
                correct = False
        else:
            if self._schema_errors(answer):
                return {
                    "answer": shown(answer),
                    "expected": shown(expected),
                    "correct": False,
                    "reason": "schema_violation",
                }
            fields = self._fields(answer, expected) if self.spec.compare.fields else None
            correct = all(fields.values()) if fields else canonical(answer) == canonical(expected)
            result = {"answer": shown(answer), "expected": shown(expected), "correct": correct}
            result["reason"] = "correct" if correct else "incorrect_answer"
            if fields is not None:
                result["detail"] = {"fields": fields}
            return result
        return {
            "answer": shown(answer),
            "expected": shown(expected),
            "correct": bool(correct),
            "reason": "correct" if correct else "incorrect_answer",
        }
