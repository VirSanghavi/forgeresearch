"""Pinned EleutherAI evaluation over explicitly supplied, verified rows.

The task's download method is replaced, never its prompt, filters, or metrics. No
TaskManager or named dataset loader can accidentally fetch the sealed test split.
"""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import distribution, version
from pathlib import Path
from types import SimpleNamespace

from .artifacts import directory_sha256, file_sha256, manifest_sha256, runtime_versions, write_json
from .config import ModelSpec
from .harness_config import HARNESS_VERSION, HarnessProtocol, ResourceLimits


def harness_identity() -> dict:
    if version("lm_eval") != HARNESS_VERSION:
        raise ValueError(f"Harness requires lm_eval=={HARNESS_VERSION}")
    package = distribution("lm_eval")
    sources = {
        str(p): file_sha256(Path(package.locate_file(p)))
        for p in package.files or []
        if str(p).startswith("lm_eval/") and str(p).endswith((".py", ".yaml"))
    }
    return {"version": HARNESS_VERSION, "source_sha256": manifest_sha256(sources)}


def task_path() -> Path:
    return Path(distribution("lm_eval").locate_file("lm_eval/tasks/gsm8k/gsm8k.yaml"))


def task_for(rows: list[dict], fewshot: list[dict], protocol: HarnessProtocol):
    import yaml
    from datasets import Dataset, DatasetDict
    from lm_eval.api.task import ConfigurableTask

    class SuppliedRowsTask(ConfigurableTask):
        def download(self, dataset_kwargs=None, **kwargs):
            self.dataset = DatasetDict(
                {
                    "test": Dataset.from_list(rows),
                    "train": Dataset.from_list(fewshot or rows[:1]),
                }
            )

    config = yaml.safe_load(task_path().read_text())
    config["num_fewshot"] = protocol.num_fewshot
    config["generation_kwargs"]["max_gen_toks"] = protocol.max_new_tokens
    return SuppliedRowsTask(config=config)


def normalize_samples(raw: dict, tokenizer, max_tokens: int, expected: list[dict]) -> list[dict]:
    groups = {}
    for item in raw["samples"]["gsm8k"]:
        group = groups.setdefault(item["doc_id"], {})
        if item["filter"] in group or item["exact_match"] not in (0, 1):
            raise ValueError("Duplicate filter or invalid harness score")
        group[item["filter"]] = item
    if set(groups) != set(range(len(expected))):
        raise ValueError("Incomplete harness sample coverage")
    samples = []
    for index, row in enumerate(expected):
        pair = groups[index]
        if set(pair) != {"strict-match", "flexible-extract"}:
            raise ValueError("Incomplete harness filter coverage")
        strict, flexible = pair["strict-match"], pair["flexible-extract"]
        if strict["doc"] != row or flexible["doc"] != row:
            raise ValueError("Harness evaluated different documents")
        if strict["resps"] != flexible["resps"] or strict["prompt_hash"] != flexible["prompt_hash"]:
            raise ValueError("Filters must score the same generation")
        response = strict["resps"][0][0]
        tokens = len(tokenizer.encode(response, add_special_tokens=False))
        samples.append(
            {
                "id": row["id"],
                "question": row["question"],
                "target": strict["target"],
                "prompt_sha256": strict["prompt_hash"],
                "prediction": response,
                "strict": bool(strict["exact_match"]),
                "flexible": bool(flexible["exact_match"]),
                "response_tokens": tokens,
                # Harness strips stop sequences; this is a conservative length diagnostic.
                "hit_limit": tokens >= max_tokens,
                "strict_extracted": strict["filtered_resps"][0],
                "flexible_extracted": flexible["filtered_resps"][0],
            }
        )
    for name, field in (("strict-match", "strict"), ("flexible-extract", "flexible")):
        if (
            abs(
                raw["results"]["gsm8k"][f"exact_match,{name}"]
                - sum(s[field] for s in samples) / len(samples)
            )
            > 1e-10
        ):
            raise ValueError("Harness aggregates do not match individual scores")
    return samples


def evaluate_side(plan_path: Path, side: str) -> dict:
    import torch
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    from lm_eval.utils import handle_non_serializable
    from peft import PeftModel

    from .modeling import load_base, seed_everything

    plan = json.loads(plan_path.read_text())
    root = plan_path.parent
    protocol = HarnessProtocol.model_validate(plan["protocol"])
    limits = ResourceLimits.model_validate(plan["limits"])
    identity = {**runtime_versions(), "lm_eval": harness_identity()}
    if identity != plan["runtime"] or file_sha256(task_path()) != plan["task_sha256"]:
        raise ValueError("Harness source/runtime changed after protocol freeze")
    for name, digest in plan["inputs"].items():
        if file_sha256(root / name) != digest:
            raise ValueError(f"Input changed after protocol freeze: {name}")
    rows = [json.loads(line) for line in (root / "evaluation.jsonl").read_text().splitlines()]
    fewshot = [json.loads(line) for line in (root / "fewshot.jsonl").read_text().splitlines()]
    if protocol.num_fewshot and len(fewshot) < protocol.num_fewshot:
        raise ValueError("Not enough independent few-shot examples")
    checkpoint = plan[side]
    if checkpoint and directory_sha256(root / checkpoint["path"]) != checkpoint["sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    seed_everything(protocol.seed)
    torch.set_num_threads(2)
    if protocol.device == "mps":
        torch.mps.set_per_process_memory_fraction(limits.mps_memory_fraction)
    config = SimpleNamespace(
        model=ModelSpec.model_validate(plan["model"]),
        device=protocol.device,
        precision=protocol.precision,
        max_model_parameters=protocol.max_model_parameters,
        attention_implementation="eager",
        device_memory_limit_mb=None,
    )
    model, tokenizer = load_base(config)
    if checkpoint:
        path = root / checkpoint["path"]
        adapter = json.loads((path / "adapter_config.json").read_text())
        if (
            adapter.get("base_model_name_or_path") != config.model.model_id
            or adapter.get("revision") != config.model.revision
        ):
            raise ValueError("Adapter base model identity mismatch")
        model = PeftModel.from_pretrained(model, str(path), is_trainable=False)
    model.eval()
    context = min(
        getattr(model.config, "max_position_embeddings", 4096),
        protocol.max_prompt_tokens + protocol.max_new_tokens,
    )

    class NoTruncationHFLM(HFLM):
        def generate_until(self, requests, disable_tqdm=False):
            for request in requests:
                length = len(self.tok_encode(request.args[0]))
                if length > min(
                    protocol.max_prompt_tokens, self.max_length - protocol.max_new_tokens
                ):
                    raise ValueError(
                        "Harness prompt exceeds context budget; refusing silent truncation"
                    )
            return super().generate_until(requests, disable_tqdm=disable_tqdm)

    lm = NoTruncationHFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=1,
        max_length=context,
        device=protocol.device,
        truncation=False,
    )
    raw = evaluator.simple_evaluate(
        model=lm,
        tasks=[task_for(rows, fewshot, protocol)],
        apply_chat_template=protocol.apply_chat_template,
        num_fewshot=protocol.num_fewshot,
        bootstrap_iters=0,
        log_samples=True,
        random_seed=protocol.seed,
        numpy_random_seed=protocol.seed,
        torch_random_seed=protocol.seed,
        fewshot_random_seed=protocol.seed,
        cache_requests=False,
        use_cache=None,
    )
    raw = json.loads(json.dumps(raw, default=handle_non_serializable))
    samples = normalize_samples(raw, tokenizer, protocol.max_new_tokens, rows)
    if checkpoint and directory_sha256(root / checkpoint["path"]) != checkpoint["sha256"]:
        raise ValueError("Checkpoint changed during evaluation")
    if {**runtime_versions(), "lm_eval": harness_identity()} != identity:
        raise ValueError("Runtime changed during evaluation")
    write_json(root / f"{side}.raw.json", raw)
    report = {
        "schema_version": 1,
        "role": plan["role"],
        "protocol": protocol.model_dump(),
        "runtime": identity,
        "task_sha256": plan["task_sha256"],
        "model": plan["model"],
        "dataset": plan["dataset"],
        "checkpoint_sha256": checkpoint["sha256"] if checkpoint else None,
        "raw_sha256": file_sha256(root / f"{side}.raw.json"),
        "samples": samples,
    }
    write_json(root / f"{side}.json", report)
    return {"status": "evaluated", "side": side, "num_examples": len(rows)}


def digest_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
