"""Real Qwen LoRA optimization and deterministic, sample-level evaluation."""

import functools
import gc
import hashlib
import json
import math
import random
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model
from forge_training.config import Example
from forge_training.modeling import prompt_for
from forge_training.tasks import GSM8K_TASK
from forge_training.training import encode_completion, collate_completion
from services.core.contracts import MODEL_ID, MODEL_REVISION, model_spec  # noqa: F401
from services.core.packs.gsm8k import benchmark_data  # noqa: F401  (the original GSM8K loader)
from services.core.protocol import (  # noqa: F401  (re-exported for the runner)
    DATASET_REVISION,
    PROMPT_VERSION,
    collect,
    equal,
    extract,
    sha,
)

# Records per optimizer step for hard-example training. Every other strategy stays at one.
HARD_ACCUMULATION = 4


def device_name():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def device_info():
    """Provenance for the hardware that actually ran a job, recorded with every result."""
    import os
    import platform
    import subprocess

    kind = device_name()
    if kind == "cuda":
        name = torch.cuda.get_device_name(0)
    elif kind == "mps":
        try:
            name = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=2
            ).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            name = ""
        name = (name or "Apple silicon") + " / Metal"
    else:
        name = platform.processor() or platform.machine() or "CPU"
    return {
        "backend": os.environ.get("FORGE_EXECUTION_BACKEND", "local"),
        "type": kind,
        "name": name,
        "torch": torch.__version__,
        "cuda": torch.version.cuda if kind == "cuda" else None,
    }


def load(adapter=None, model_id=None):
    spec = model_spec(model_id)
    device = device_name()
    dtype = torch.float32 if device == "cpu" else getattr(torch, spec["dtype"])
    model = AutoModelForCausalLM.from_pretrained(
        spec["id"],
        revision=spec["revision"],
        dtype=dtype,
        attn_implementation="eager",
        trust_remote_code=False,
        use_safetensors=True,
        # Materialize weights on the GPU so a large model fits the container's RAM. Not on MPS:
        # Pythia segfaults or returns garbage when loaded that way.
        **({"device_map": {"": device}} if device == "cuda" else {}),
    )
    tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"], trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if spec["chat_template_kwargs"]:
        # Every prompt, for evaluation and for training targets, renders with the same template
        # settings (Qwen3.5: non-thinking mode).
        tokenizer.apply_chat_template = functools.partial(
            tokenizer.apply_chat_template, **spec["chat_template_kwargs"]
        )
    if adapter:
        model = PeftModel.from_pretrained(model, str(adapter))
    model.forge_model_id = spec["id"]
    return model.to(device), tokenizer


def unload(model):
    model.to("cpu")
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def predict(model, tokenizer, rows, task=GSM8K_TASK):
    """Greedy answers for a batch of rows. With one row this is exactly the original protocol.

    Scoring happens in protocol.collect, on this machine, with the task's checker.
    """
    texts = [prompt_for(tokenizer, row["question"], task.instruction) for row in rows]
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    limit = task.max_new_tokens or model_spec(getattr(model, "forge_model_id", None))["max_new_tokens"]
    tokens = model.generate(
        **inputs, do_sample=False, max_new_tokens=limit, pad_token_id=tokenizer.eos_token_id
    )
    stops = {tokenizer.eos_token_id}
    configured = getattr(model.generation_config, "eos_token_id", None)
    stops.update(
        configured if isinstance(configured, list) else [configured] if configured is not None else []
    )
    width = inputs["input_ids"].shape[1]
    predictions = []
    for row, generated in zip(rows, tokens[:, width:].tolist(), strict=True):
        end = next((i for i, token in enumerate(generated) if token in stops), None)
        if end is not None:
            generated = generated[: end + 1]
        output = tokenizer.decode(generated, skip_special_tokens=True)
        predictions.append(
            {
                "id": row["id"],
                "question": row["question"],
                "output": output,
                "finish_reason": "stop" if end is not None else "length",
            }
        )
    return predictions


def generate(model, tokenizer, rows, cancelled, batch_size=1, task=GSM8K_TASK):
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            if cancelled():
                raise InterruptedError("Campaign stopped")
            yield from predict(model, tokenizer, rows[start : start + batch_size], task)


def evaluate(
    model, tokenizer, rows, progress, cancelled, split="development", worlds=None, batch_size=1, pack=None
):
    from services.core import packs

    pack = pack or packs.get()
    started = time.monotonic()
    stream = generate(model, tokenizer, rows, cancelled, batch_size, pack.task)
    model_id = getattr(model, "forge_model_id", None)
    return collect(
        rows,
        stream,
        progress,
        split,
        worlds,
        started,
        device_info(),
        batch_size,
        model_id=model_id,
        pack=pack,
    )


def scheduled_rate(peak, step, steps):
    """Linear warmup over the first tenth of training, then cosine decay to zero."""
    warmup = max(1, steps // 10)
    if step < warmup:
        return peak * (step + 1) / warmup
    return peak * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))


def train(proposal, config, rows, folder, progress, cancelled, task=GSM8K_TASK):
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    device = device_info()
    spec = model_spec(config.get("model_id"))
    model, tokenizer = load(model_id=spec["id"])
    model = get_peft_model(
        model,
        LoraConfig(
            r=proposal["lora_rank"],
            lora_alpha=proposal["lora_rank"] * 2,
            lora_dropout=0.05,
            target_modules=spec["lora_targets"],
            task_type="CAUSAL_LM",
        ),
    )
    params = [p for p in model.parameters() if p.requires_grad]
    before = hashlib.sha256(b"".join(p.detach().float().cpu().numpy().tobytes() for p in params)).hexdigest()
    optimizer = torch.optim.AdamW(params, lr=proposal["learning_rate"])
    model.train()
    model.config.use_cache = False
    metrics = []
    sampled = list(rows)
    production = [r for r in sampled if r["id"].startswith("prod-")]
    sampled = [r for r in sampled if not r["id"].startswith("prod-")]
    if proposal["strategy"] == "targeted_sft":
        # Reweight longer, multi-step worked solutions using training data only.
        sampled.sort(key=lambda r: len(r["answer"]), reverse=True)
        sampled = sampled[: max(32, len(sampled) // 2)]
    hard = proposal["strategy"] == "hard_sft"
    # Hard-example rows arrive in their deliberate hard/easy order.
    if not hard:
        random.Random(config["seed"]).shuffle(sampled)
    random.Random(config["seed"]).shuffle(production)
    # Each step trains one record, so targeted and distilled runs see production failures first.
    if proposal["strategy"] in ("targeted_sft", "distill_sft", "hard_sft"):
        sampled = production + sampled
    else:
        sampled = sampled + production
        random.Random(config["seed"]).shuffle(sampled)
    records = []
    accepted = []
    excluded = []
    for row in sampled:
        try:
            example = Example(id=row["id"], question=row["question"], answer=row["answer"])
            records.append(encode_completion(tokenizer, example, 768, task.instruction))
            accepted.append(row)
        except ValueError:
            excluded.append(row["id"])
    if not records:
        raise ValueError("No complete training examples fit the sequence limit")
    accumulation = HARD_ACCUMULATION if hard else 1
    started = time.monotonic()
    for step in range(config["train_steps"]):
        if cancelled():
            raise InterruptedError("Campaign stopped")
        rate = proposal["learning_rate"]
        if hard:
            rate = scheduled_rate(rate, step, config["train_steps"])
            for group in optimizer.param_groups:
                group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for part in range(accumulation):
            record = records[(step * accumulation + part) % len(records)]
            batch = collate_completion([record], tokenizer.pad_token_id, str(model.device))
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss; experiment quarantined")
            (loss / accumulation).backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        optimizer.step()
        metric = {
            "step": step + 1,
            "loss": round(total / accumulation, 5),
            "elapsed": round(time.monotonic() - started, 2),
            "learning_rate": rate,
        }
        metrics.append(metric)
        progress(metric)
    after = hashlib.sha256(b"".join(p.detach().float().cpu().numpy().tobytes() for p in params)).hexdigest()
    folder.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    (folder / "training.json").write_text(
        json.dumps(
            {
                "proposal": proposal,
                "config": config,
                "metrics": metrics,
                "before_hash": before,
                "after_hash": after,
                "weights_changed": before != after,
                "train_ids": [r["id"] for r in accepted],
                "excluded_oversize_ids": excluded,
                "max_sequence_length": 768,
                "gradient_accumulation": accumulation,
                "prompt_version": task.prompt_version,
                "model_id": spec["id"],
                "model_revision": spec["revision"],
                "lora_targets": spec["lora_targets"],
                "device": device,
            },
            indent=2,
        )
    )
    del optimizer, params
    unload(model)
    return {
        "metrics": metrics,
        "before_hash": before,
        "after_hash": after,
        "weights_changed": before != after,
        "sha256": sha(folder / "adapter_model.safetensors"),
        "seconds": time.monotonic() - started,
        "max_sequence_length": 768,
        "lora_targets": spec["lora_targets"],
        "dtype": spec["dtype"],
        "device": device,
    }
