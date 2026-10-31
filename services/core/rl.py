"""Bounded TRL GRPO. Every scalar reward is returned by a real Terse world."""

import hashlib
import json
import time
import uuid
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from services.core import store, training
from services.core.actor_client import get_actor


def parameter_hash(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if p.requires_grad:
            h.update(name.encode())
            h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def train_grpo(
    campaign_id, experiment_id, proposal, config, rows, folder, progress, cancelled, warm_start=None
):
    torch.manual_seed(config["seed"])
    spec = training.model_spec(config.get("model_id"))
    model, tokenizer = training.load(warm_start, spec["id"])
    if warm_start:
        if model.peft_config["default"].r != proposal["lora_rank"]:
            raise ValueError(
                f"Approved warm-start adapter has LoRA rank {model.peft_config['default'].r}; requested rank {proposal['lora_rank']}. Revise lora_rank to match the approved adapter."
            )
        for name, p in model.named_parameters():
            p.requires_grad_("lora_" in name)
    peft_config = (
        None
        if warm_start
        else LoraConfig(
            r=proposal["lora_rank"],
            lora_alpha=proposal["lora_rank"] * 2,
            target_modules=spec["lora_targets"],
            task_type="CAUSAL_LM",
        )
    )
    metrics, trajectories = [], []
    started = time.monotonic()
    holder = {}

    def reward_function(prompts, completions, reference, task_id, **_kwargs):
        if cancelled():
            raise InterruptedError("GRPO cancelled at rollout boundary")
        rewards = []
        policy_hash = parameter_hash(holder["trainer"].model)
        for i, (prompt, completion, answer, source_id) in enumerate(
            zip(prompts, completions, reference, task_id, strict=True)
        ):
            output = completion[0]["content"] if isinstance(completion, list) else completion
            eid = f"{campaign_id}-rl-{uuid.uuid4().hex[:10]}"
            world = get_actor("RLWorldActor", eid)
            question = prompt[-1]["content"] if isinstance(prompt, list) else prompt
            world.reset(eid, question, training.extract(answer), config["seed"] + i)
            result = json.loads(
                world.step(
                    json.dumps(
                        {
                            "episode_id": eid,
                            "action_id": eid + "-answer",
                            "expected_version": 0,
                            "answer": training.extract(output),
                        }
                    )
                )
            )
            result.update(
                id=eid,
                model_output=output,
                policy_sha256=policy_hash,
                policy_step=holder["trainer"].state.global_step,
                source_id=source_id,
            )
            trajectories.append(result)
            rewards.append(float(result["reward"]))
        s = store.get(campaign_id)
        if s:
            store.patch(
                campaign_id, lambda s: s.setdefault("environments", []).extend(trajectories[-len(rewards) :])
            )
            store.event(
                campaign_id,
                "rollout",
                f"GRPO rollout: {sum(rewards):.0f}/{len(rewards)} exact-answer reward. Terse policy step {holder['trainer'].state.global_step}.",
                "trainer",
                experiment_id=experiment_id,
            )
        return rewards

    class Telemetry(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs and state.global_step:
                m = {
                    "step": state.global_step,
                    "loss": float(logs["loss"]),
                    "elapsed": time.monotonic() - started,
                    "learning_rate": proposal["learning_rate"],
                    "reward": float(logs.get("reward", 0)),
                }
                metrics.append(m)
                progress(m)

        def on_step_begin(self, args, state, control, **kwargs):
            if cancelled():
                raise InterruptedError("GRPO compute window stopped")

    data = Dataset.from_list(
        [
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": "Solve the problem. End with a final line in the form #### <number>.\n\n"
                        + r["question"],
                    }
                ],
                "reference": r["answer"],
                "task_id": r["id"],
            }
            for r in rows[:32]
        ]
    )
    args = GRPOConfig(
        output_dir=str(folder / "trainer-state"),
        max_steps=min(8, config["train_steps"]),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        num_generations=4,
        max_completion_length=160,
        learning_rate=proposal["learning_rate"],
        temperature=1.0,
        beta=0.02,
        bf16=False,
        fp16=False,
        gradient_checkpointing=False,
        use_vllm=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
        seed=config["seed"],
        dataloader_pin_memory=False,
    )
    trainer = GRPOTrainer(
        model=model,
        args=args,
        reward_funcs=reward_function,
        train_dataset=data,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[Telemetry()],
    )
    holder["trainer"] = trainer
    before = parameter_hash(trainer.model)
    trainer.train()
    after = parameter_hash(trainer.model)
    if any(not torch.isfinite(p).all() for p in trainer.model.parameters() if p.requires_grad):
        raise ValueError("GRPO produced nonfinite adapter parameters")
    folder.mkdir(exist_ok=True, parents=True)
    (folder / "trajectories.json").write_text(json.dumps(trajectories, indent=2))
    if before == after:
        raise RuntimeError("GRPO produced no weight change; rollouts preserved, checkpoint rejected")
    trainer.save_model(str(folder))
    tokenizer.save_pretrained(folder)
    result = {
        "metrics": metrics,
        "before_hash": before,
        "after_hash": after,
        "weights_changed": True,
        "sha256": training.sha(folder / "adapter_model.safetensors"),
        "seconds": time.monotonic() - started,
        "algorithm": "TRL GRPO",
        "trl_version": "1.16.0",
        "rollouts": len(trajectories),
        "policy_versions_observed": len({t["policy_sha256"] for t in trajectories}),
        "warm_start": bool(warm_start),
    }
    (folder / "training.json").write_text(
        json.dumps({"proposal": proposal, "config": config, **result}, indent=2)
    )
    training.unload(trainer.model)
    return result
