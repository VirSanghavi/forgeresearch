"""Completion-only supervised LoRA optimization with explicit local resource bounds."""

from __future__ import annotations

import inspect
import json
import math
import random
import time
from pathlib import Path

from .artifacts import EventLog, directory_manifest, manifest_sha256, utc_now, write_json
from .config import Example, ExperimentConfig
from .modeling import PROMPT_INSTRUCTION, load_base, parameter_sha256, prompt_for, seed_everything


def encode_completion(
    tokenizer, example: Example, max_length: int, instruction: str = PROMPT_INSTRUCTION
) -> dict:
    prompt = prompt_for(tokenizer, example.question, instruction)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(example.answer, add_special_tokens=False) + [
        tokenizer.eos_token_id
    ]
    if len(prompt_ids) + len(answer_ids) > max_length:
        raise ValueError(f"Training example exceeds sequence limit (no truncation): {example.id}")
    return {
        "input_ids": prompt_ids + answer_ids,
        "labels": [-100] * len(prompt_ids) + answer_ids,
    }


def collate_completion(records: list[dict], pad_id: int, device: str) -> dict:
    import torch

    width = max(len(record["input_ids"]) for record in records)
    ids, labels, masks = [], [], []
    for record in records:
        padding = width - len(record["input_ids"])
        ids.append(record["input_ids"] + [pad_id] * padding)
        labels.append(record["labels"] + [-100] * padding)
        masks.append([1] * len(record["input_ids"]) + [0] * padding)
    return {
        key: torch.tensor(value, dtype=torch.long, device=device)
        for key, value in (("input_ids", ids), ("labels", labels), ("attention_mask", masks))
    }


def completion_loss(model, batch: dict, *, completion_only_logits: bool):
    """Same token-mean causal loss, optionally projecting only supervised positions."""
    import torch
    from torch.nn import functional as F

    if not completion_only_logits:
        return model(**batch).loss
    labels = batch["labels"]
    shifted = F.pad(labels[:, 1:], (0, 1), value=-100)
    positions = torch.nonzero((shifted != -100).any(dim=0), as_tuple=True)[0]
    inputs = {key: value for key, value in batch.items() if key != "labels"}
    logits = model(**inputs, logits_to_keep=positions).logits
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        shifted[:, positions].reshape(-1),
        ignore_index=-100,
    )


def train(config: ExperimentConfig, rows: list[Example], output: Path, events: EventLog) -> dict:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model

    started = time.monotonic()
    seed_everything(config.seed)
    model, tokenizer = load_base(config)
    if config.training.completion_only_logits and (
        "logits_to_keep" not in inspect.signature(model.forward).parameters
    ):
        raise ValueError("completion_only_logits requires a model with logits_to_keep support")
    records = [
        encode_completion(tokenizer, row, config.training.max_sequence_length) for row in rows
    ]
    context_limit = getattr(model.config, "max_position_embeddings", None)
    if context_limit and any(len(record["input_ids"]) > context_limit for record in records):
        raise ValueError("Training data exceeds model context window")
    spec = config.training
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=spec.lora_rank,
            lora_alpha=spec.lora_alpha,
            lora_dropout=0.0,
            target_modules=spec.target_modules,
            bias="none",
            revision=config.model.revision,
        ),
    )
    # PEFT may infer a path from a loaded model; pin it explicitly for future reloads.
    model.peft_config["default"].base_model_name_or_path = config.model.model_id
    model.peft_config["default"].revision = config.model.revision
    model.config.use_cache = False
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("LoRA configuration did not create trainable parameters")
    before = parameter_sha256(model)
    optimizer = torch.optim.AdamW(parameters, lr=spec.learning_rate)
    generator = random.Random(config.seed)
    order = list(range(len(records)))
    generator.shuffle(order)
    cursor = 0
    consumed_tokens = 0
    supervised_tokens = 0
    losses = []
    model.train()
    events.emit(
        "TRAINING_STARTED",
        {
            "maxSteps": spec.max_steps,
            "trainableParameters": sum(parameter.numel() for parameter in parameters),
        },
    )
    metrics_path = output / "training-metrics.jsonl"
    optimization_started = time.monotonic()
    with metrics_path.open("x", encoding="utf-8") as metrics:
        for step in range(1, spec.max_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            microbatches = []
            for _ in range(spec.gradient_accumulation_steps):
                if time.monotonic() - started >= spec.max_seconds:
                    raise TimeoutError("Training deadline reached; no completed checkpoint emitted")
                selected = []
                for _ in range(spec.batch_size):
                    if cursor == len(order):
                        generator.shuffle(order)
                        cursor = 0
                    selected.append(records[order[cursor]])
                    cursor += 1
                token_count = sum(len(record["input_ids"]) for record in selected)
                if consumed_tokens + token_count > spec.max_training_tokens:
                    raise ValueError("Training token budget would be exceeded")
                targets = sum(sum(label != -100 for label in r["labels"][1:]) for r in selected)
                microbatches.append((selected, targets))
                consumed_tokens += token_count
            step_targets = sum(targets for _, targets in microbatches)
            for selected, targets in microbatches:
                if time.monotonic() - started >= spec.max_seconds:
                    raise TimeoutError("Training deadline reached; no completed checkpoint emitted")
                batch = collate_completion(selected, tokenizer.pad_token_id, config.device)
                loss = completion_loss(
                    model, batch, completion_only_logits=spec.completion_only_logits
                )
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                # Weight by supervised tokens across the *whole* optimizer batch.
                # Equal microbatch weighting overweights short answers.
                weight = targets / step_targets
                (loss * weight).backward()
                step_loss += loss.detach().item() * weight
            supervised_tokens += step_targets
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            if time.monotonic() - started > spec.max_seconds:
                raise TimeoutError("Training deadline exceeded; no completed checkpoint emitted")
            if not math.isfinite(step_loss):
                raise ValueError("Non-finite training metric")
            metric = {
                "step": step,
                "loss": step_loss,
                "training_tokens": consumed_tokens,
                "supervised_tokens": supervised_tokens,
                "elapsed_seconds": time.monotonic() - started,
                "timestamp": utc_now(),
            }
            metrics.write(json.dumps(metric, allow_nan=False) + "\n")
            metrics.flush()
            losses.append(step_loss)
            events.emit("TRAINING_PROGRESS", metric)
    optimization_seconds = time.monotonic() - optimization_started
    after = parameter_sha256(model)
    if after == before:
        raise ValueError("Optimization did not change trainable weights")
    checkpoint = output / "checkpoint"
    checkpoint.mkdir()
    model.save_pretrained(checkpoint, safe_serialization=True)
    # Tokenizer identity is pinned by the base reference. Keep checkpoint artifacts adapter-only.
    files = directory_manifest(checkpoint)
    summary = {
        "method": "sft_lora",
        "optimizer": {
            "name": "AdamW",
            "learning_rate": spec.learning_rate,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "max_gradient_norm": 1.0,
            "scheduler": "constant",
        },
        "optimizer_steps": len(losses),
        "training_tokens": consumed_tokens,
        "supervised_tokens": supervised_tokens,
        "loss_normalization": "supervised_tokens_per_optimizer_step",
        "completion_only_logits": spec.completion_only_logits,
        "attention_implementation": config.attention_implementation,
        "device_memory_limit_mb": config.device_memory_limit_mb,
        "optimization_seconds": optimization_seconds,
        "training_tokens_per_second": consumed_tokens / optimization_seconds,
        "initial_trainable_sha256": before,
        "final_trainable_sha256": after,
        "weights_changed": True,
        "checkpoint_sha256": manifest_sha256(files),
        "checkpoint_files": files,
        "checkpoint_path": str(checkpoint),
        "elapsed_seconds": time.monotonic() - started,
        "first_step_loss": losses[0],
        "last_step_loss": losses[-1],
        "cost_usd": None,
        "cost_status": "not_measured_by_local_worker",
    }
    write_json(output / "training.json", summary)
    events.emit(
        "TRAINING_COMPLETED",
        {
            "checkpointSha256": summary["checkpoint_sha256"],
            "optimizerSteps": len(losses),
            "weightsChanged": True,
        },
    )
    return summary
