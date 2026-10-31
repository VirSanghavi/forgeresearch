"""Preference alignment: TRL DPO training and held-out preference accuracy.

A preference row is a prompt with a chosen and a rejected reply. Training uses TRL's DPOTrainer on
a LoRA adapter (the reference model is the same base with the adapter disabled, so no second copy
is loaded). Evaluation asks, for each held-out pair, whether the model assigns the chosen reply a
higher mean per-token log-probability than the rejected one. That is deterministic, costs no API
calls, has a meaningful baseline (the untrained model) and gives a per-example `correct`, so the
same paired-bootstrap verdict applies as for every other task.
"""

import json
import time

import torch

from services.core import protocol, training
from services.core.rl import parameter_hash

MAX_LENGTH = 768
LEARNING_RATE_CAP = 1e-4
TRL_VERSION = "1.16.0"


def _continuation_logprob(model, tokenizer, prompt, reply):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    reply_ids = tokenizer.encode(reply, add_special_tokens=False) + [tokenizer.eos_token_id]
    ids = (prompt_ids + reply_ids)[:MAX_LENGTH]
    if len(ids) <= len(prompt_ids):
        return float("-inf"), 0
    tensor = torch.tensor([ids], device=model.device)
    logits = model(input_ids=tensor).logits[0, :-1].float()
    targets = tensor[0, 1:]
    logprobs = torch.log_softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)
    scored = logprobs[len(prompt_ids) - 1 :]
    return float(scored.mean()), int(scored.numel())


def score_pairs(model, tokenizer, rows, progress, cancelled, split="development", pack=None):
    """Held-out preference accuracy with the same summary shape as every evaluation."""
    from forge_training.modeling import prompt_for

    started = time.monotonic()
    model.eval()
    results = []
    with torch.inference_mode():
        for i, row in enumerate(rows):
            if cancelled():
                raise InterruptedError("Campaign stopped")
            prompt = prompt_for(tokenizer, row["question"], pack.task.instruction)
            chosen, n_chosen = _continuation_logprob(model, tokenizer, prompt, row["chosen"])
            rejected, n_rejected = _continuation_logprob(model, tokenizer, prompt, row["rejected"])
            margin = chosen - rejected
            p = {
                "id": row["id"],
                "question": row["question"],
                "output": "",
                "answer": "chosen" if margin > 0 else "rejected",
                "expected": "chosen",
                "correct": margin > 0,
                "finish_reason": None,
                "margin": round(margin, 6),
                "logp_chosen": round(chosen, 6),
                "logp_rejected": round(rejected, 6),
                "tokens": {"chosen": n_chosen, "rejected": n_rejected},
                "environment_id": None,
            }
            results.append(p)
            progress(i + 1, len(rows), p)
    return protocol.summary(
        results,
        split,
        started,
        training.device_info(),
        model_id=getattr(model, "forge_model_id", None),
        pack=pack,
        decoding={"method": "teacher-forced mean token log-probability", "max_length": MAX_LENGTH},
    )


def train_dpo(
    proposal, config, rows, folder, progress, cancelled, pack, warm_start=None, loader=None, lora_targets=None
):
    """A bounded LoRA DPO run. Returns the same result shape as SFT and GRPO."""
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import TrainerCallback
    from trl import DPOConfig, DPOTrainer

    from services.core.contracts import model_spec

    torch.manual_seed(config["seed"])
    spec = model_spec(config.get("model_id"))
    model, tokenizer = (loader or training.load)(warm_start, config.get("model_id"))
    tokenizer.padding_side = "right"
    if warm_start:
        if model.peft_config["default"].r != proposal["lora_rank"]:
            raise ValueError(
                f"Approved warm-start adapter has LoRA rank {model.peft_config['default'].r}; requested rank "
                f"{proposal['lora_rank']}. Revise lora_rank to match the approved adapter."
            )
        for name, p in model.named_parameters():
            p.requires_grad_("lora_" in name)
    peft_config = (
        None
        if warm_start
        else LoraConfig(
            r=proposal["lora_rank"],
            lora_alpha=proposal["lora_rank"] * 2,
            lora_dropout=0.05,
            target_modules=lora_targets or spec["lora_targets"],
            task_type="CAUSAL_LM",
        )
    )
    learning_rate = min(proposal["learning_rate"], LEARNING_RATE_CAP)
    pairs = [r for r in rows if r.get("chosen") and r.get("rejected")]
    if not pairs:
        raise ValueError("No preference pairs to train on")
    data = Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": f"{pack.task.instruction}\n\n{r['question']}"}],
                "chosen": [{"role": "assistant", "content": r["chosen"]}],
                "rejected": [{"role": "assistant", "content": r["rejected"]}],
            }
            for r in pairs
        ]
    )
    metrics = []
    started = time.monotonic()

    class Telemetry(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs and state.global_step:
                m = {
                    "step": state.global_step,
                    "loss": float(logs["loss"]),
                    "elapsed": time.monotonic() - started,
                    "learning_rate": learning_rate,
                    "reward_accuracy": float(logs.get("rewards/accuracies", 0)),
                    "reward_margin": float(logs.get("rewards/margins", 0)),
                }
                metrics.append(m)
                progress(m)

        def on_step_begin(self, args, state, control, **kwargs):
            if cancelled():
                raise InterruptedError("DPO compute window stopped")

    args = DPOConfig(
        output_dir=str(folder / "trainer-state"),
        max_steps=config["train_steps"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=learning_rate,
        beta=0.1,
        loss_type=["sigmoid"],
        max_length=MAX_LENGTH,
        bf16=False,
        fp16=False,
        gradient_checkpointing=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
        seed=config["seed"],
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=data,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[Telemetry()],
    )
    before = parameter_hash(trainer.model)
    trainer.train()
    after = parameter_hash(trainer.model)
    if any(not torch.isfinite(p).all() for p in trainer.model.parameters() if p.requires_grad):
        raise ValueError("DPO produced nonfinite adapter parameters")
    if before == after:
        raise RuntimeError("DPO produced no weight change; checkpoint rejected")
    folder.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(folder))
    tokenizer.padding_side = "left"
    tokenizer.save_pretrained(folder)
    result = {
        "metrics": metrics,
        "before_hash": before,
        "after_hash": after,
        "weights_changed": True,
        "sha256": training.sha(folder / "adapter_model.safetensors"),
        "seconds": time.monotonic() - started,
        "algorithm": "TRL DPO",
        "trl_version": TRL_VERSION,
        "beta": 0.1,
        "pairs": len(pairs),
        "learning_rate": learning_rate,
        **(
            {"learning_rate_clamped_from": proposal["learning_rate"]}
            if learning_rate != proposal["learning_rate"]
            else {}
        ),
        "lora_targets": lora_targets or spec["lora_targets"],
        "warm_start": bool(warm_start),
        "device": training.device_info(),
    }
    (folder / "training.json").write_text(
        json.dumps(
            {"proposal": proposal, "config": config, "prompt_version": pack.task.prompt_version, **result},
            indent=2,
        )
    )
    training.unload(trainer.model)
    return result
