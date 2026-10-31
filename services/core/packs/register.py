"""Turn a company's file into an immutable, validated task pack.

Rows are mapped with simple `{column}` templates, deduplicated by normalized question and split
by a salted hash into sealed, development and training pools. Every reference must pass the
task's own checker, so a mapping mistake is caught here instead of showing up as zero scores.
Sealed rows never leave this folder except through the final evaluator.
"""

import csv
import hashlib
import io
import json
import re
import secrets

from forge_training.checkers import Checker, CheckerSpec
from forge_training.data import question_key
from forge_training.tasks import TaskSpec
from pydantic import BaseModel, ConfigDict, Field, model_validator

from services.core.packs.gsm8k import PACK as GSM8K
from services.core.packs.spec import Capabilities, Display, Source, TaskPack

MAX_ROWS = 200_000
MIN_SEALED, MIN_DEV = 24, 12
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
PLACEHOLDER = re.compile(r"\{([A-Za-z0-9_ .-]{1,64})\}")


class Mapping(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str | None = None
    input_template: str = Field(default="{question}", min_length=3, max_length=4000)
    target_template: str | None = Field(default="{answer}", max_length=4000)
    chosen: str | None = Field(default=None, max_length=4000)
    rejected: str | None = Field(default=None, max_length=4000)


class Draft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=3, max_length=80)
    objective: str = Field(min_length=10, max_length=500)
    kind: str = Field(pattern=r"^(qa|extraction|preference)$")
    format: str = Field(default="jsonl", pattern=r"^(jsonl|csv)$")
    mapping: Mapping = Field(default_factory=Mapping)
    instruction: str = Field(min_length=1, max_length=4000)
    checker: CheckerSpec | None = None
    license: str = Field(min_length=1, max_length=200)
    # The operator's statement that this data may be used to train the model.
    data_terms: str = Field(min_length=20, max_length=2000)
    sealed_fraction: float = Field(default=0.2, ge=0.05, le=0.5)
    dev_fraction: float = Field(default=0.15, ge=0.05, le=0.5)
    max_new_tokens: int | None = Field(default=None, ge=16, le=4096)
    unit: str = Field(default="examples", min_length=1, max_length=24)

    @model_validator(mode="after")
    def consistent(self):
        if self.kind == "preference":
            if not (self.mapping.chosen and self.mapping.rejected):
                raise ValueError("A preference task maps a chosen and a rejected reply")
        elif not (self.checker and self.mapping.target_template):
            raise ValueError("Q&A and extraction tasks need a reference template and a checker")
        if self.checker and self.checker.compare.kind == "python":
            raise ValueError("Python checkers are registered from the command line only")
        if self.sealed_fraction + self.dev_fraction > 0.7:
            raise ValueError("Keep at least 30% of rows for training")
        return self


def parse(text, fmt):
    """Rows as {column: text}. Nested JSON values are kept as compact JSON."""
    if fmt == "csv":
        reader = csv.DictReader(io.StringIO(text))
        return [(i + 2, {k: v or "" for k, v in row.items() if k}) for i, row in enumerate(reader)]
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            rows.append((number, None))
            continue
        if not isinstance(value, dict):
            rows.append((number, None))
            continue
        rows.append(
            (
                number,
                {
                    k: v if isinstance(v, str) else json.dumps(v, separators=(",", ":"), ensure_ascii=False)
                    for k, v in value.items()
                },
            )
        )
    return rows


def render(template, row):
    missing = [name for name in PLACEHOLDER.findall(template) if name not in row]
    if missing:
        raise KeyError(", ".join(sorted(set(missing))))
    return PLACEHOLDER.sub(lambda m: row[m.group(1)], template).strip()


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "pack"


def task_for(draft, pack_id):
    checker = draft.checker or CheckerSpec.model_validate(
        {"extract": {"kind": "whole"}, "compare": {"kind": "exact"}, "reference": "whole"}
    )
    return TaskSpec(
        task_id=pack_id,
        instruction=draft.instruction,
        input_template=draft.mapping.input_template,
        target_template=draft.mapping.target_template or draft.mapping.chosen,
        prompt_version=f"{pack_id}-v1",
        checker=checker,
        max_new_tokens=draft.max_new_tokens,
    )


def build(draft, text):
    """Validate, map and split. Returns (pack spec, pools, report). Nothing is written."""
    if not isinstance(draft, Draft):
        draft = Draft.model_validate(draft)
    content_sha = hashlib.sha256(text.encode()).hexdigest()
    rows = parse(text, draft.format)
    if len(rows) > MAX_ROWS:
        raise ValueError(f"At most {MAX_ROWS} rows per task")
    checker = Checker(draft.checker) if draft.checker else None
    seen, examples, invalid, duplicates = set(), [], [], 0
    for line, row in rows:
        if row is None:
            invalid.append({"line": line, "reason": "not a JSON object"})
            continue
        try:
            question = render(draft.mapping.input_template, row)
            if draft.kind == "preference":
                chosen = render(draft.mapping.chosen, row)
                rejected = render(draft.mapping.rejected, row)
                if not chosen or not rejected or chosen == rejected:
                    raise ValueError("chosen and rejected must be different and non-empty")
                fields = {"answer": chosen, "chosen": chosen, "rejected": rejected}
            else:
                answer = render(draft.mapping.target_template, row)
                if not answer:
                    raise ValueError("empty reference")
                checker.check_reference(answer)
                fields = {"answer": answer}
        except KeyError as error:
            invalid.append({"line": line, "reason": f"missing column {error.args[0]}"})
            continue
        except ValueError as error:
            invalid.append({"line": line, "reason": str(error)[:200]})
            continue
        if not question:
            invalid.append({"line": line, "reason": "empty question"})
            continue
        key = question_key(question)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        given = row.get(draft.mapping.id) if draft.mapping.id else None
        row_id = (
            given
            if given and IDENTIFIER.match(given)
            else f"row-{hashlib.sha256(key.encode()).hexdigest()[:12]}"
        )
        examples.append({"id": row_id, "question": question, **fields})
    if len({e["id"] for e in examples}) != len(examples):
        raise ValueError("Row ids are not unique; map a unique id column or leave it unmapped")
    salt = secrets.token_hex(8)
    # Rank rows by a salted hash and cut at exact fractions: random membership, exact pool sizes.
    ranked = sorted(
        examples, key=lambda e: hashlib.sha256((salt + question_key(e["question"])).encode()).hexdigest()
    )
    sealed_n = round(draft.sealed_fraction * len(ranked))
    dev_n = round(draft.dev_fraction * len(ranked))
    pools = {
        "sealed": ranked[:sealed_n],
        "dev": ranked[sealed_n : sealed_n + dev_n],
        "train": ranked[sealed_n + dev_n :],
    }
    report = {
        "rows": len(rows),
        "valid": len(examples),
        "invalid": len(invalid),
        "invalid_examples": invalid[:20],
        "duplicates": duplicates,
        "split": {name: len(pool) for name, pool in pools.items()},
        "content_sha256": content_sha,
        "samples": pools["train"][:3],
        "problems": [],
    }
    if len(pools["sealed"]) < MIN_SEALED:
        report["problems"].append(
            f"Only {len(pools['sealed'])} sealed rows; at least {MIN_SEALED} are needed"
        )
    if len(pools["dev"]) < MIN_DEV:
        report["problems"].append(f"Only {len(pools['dev'])} development rows; at least {MIN_DEV} are needed")
    if not pools["train"]:
        report["problems"].append("No training rows")
    definition = draft.model_dump(mode="json")
    pack_id = f"{slug(draft.name)}-{hashlib.sha256((content_sha + json.dumps(definition, sort_keys=True)).encode()).hexdigest()[:8]}"
    pack = TaskPack(
        id=pack_id,
        kind=draft.kind,
        task=task_for(draft, pack_id),
        source=Source(
            kind="upload", revision=content_sha, license=draft.license, data_terms=draft.data_terms
        ),
        display=Display(
            name=draft.name,
            objective=draft.objective,
            benchmark_label=draft.name,
            scoring_label=f"{task_for(draft, pack_id).make_checker().checker_id} on the reference answer"
            if draft.kind != "preference"
            else "Held-out preference accuracy: chosen reply more likely than rejected",
            unit=draft.unit,
        ),
        # Uploaded tasks keep the four arithmetic questions as a general skill-retention check,
        # scored with the GSM8K task's own prompt and checker.
        regression=GSM8K.regression,
        regression_pack=GSM8K.id,
        regression_title="General arithmetic retention",
        regression_label="Four arithmetic questions scored with the GSM8K checker: a small retention check, not a broad safety evaluation.",
        detectors=("truncated", "user_reported"),
        limitations=(
            f"{draft.name}: an operator-supplied dataset; its quality and licensing are as attested at registration",
            "Development and sealed rows are a random split of the same file, not an independent benchmark",
            "Possible base model pretraining contamination",
        ),
        capabilities=Capabilities(
            terse_world=False,
            grpo=False,
            dpo=draft.kind == "preference",
            lm_eval_final=False,
            remote=draft.kind != "preference",
        ),
    )
    report["pack_id"] = pack_id
    report["salt"] = salt
    return pack, pools, report
