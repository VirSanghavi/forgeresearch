"""Group-relative policy optimization (pinned TRL) driven by Terse world transitions.

The trainer never receives reference answers: its training data are answer-free tasks,
and every reward is the RLWorldActor's reward for a submitted action, obtained through
the orchestrator's loopback world gateway, which signs the actor commands. Each update
records the adapter state that generated its rollouts and the state it produced, and
saves adapter snapshots, so lineage from rollout to update to the next rollout is
checkable from the files alone.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from importlib.metadata import version
from pathlib import Path

from .artifacts import (
    EventLog,
    directory_manifest,
    directory_sha256,
    manifest_sha256,
    utc_now,
    write_json,
)
from .config import ExperimentConfig, Task
from .data import load_tasks
from .modeling import load_base, parameter_sha256, prompt_for, seed_everything
from .scoring import extract_answer

TRL_VERSION = "0.24.0"
REWARD_SOURCE = "terse_world_transitions"
# Submitted when a completion has no final answer line; the world scores it 0.
NO_ANSWER = "none"
GATEWAY_URL, GATEWAY_TOKEN = "FORGE_WORLD_GATEWAY_URL", "FORGE_WORLD_GATEWAY_TOKEN"
TRL_METRICS = (
    "reward",
    "reward_std",
    "frac_reward_zero_std",
    "completions/mean_length",
    "completions/clipped_ratio",
    "entropy",
    "kl",
    "clip_ratio/region_mean",
    "num_tokens",
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def answer_action(completion: str) -> dict:
    """final_answer_v1: submit the final '#### <number>' value without grouping commas."""
    value = extract_answer(completion)
    answer = NO_ANSWER if value is None else format(value, "f")
    return {"type": "submit_answer", "answer": answer if len(answer) <= 64 else NO_ANSWER}


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def group_advantages(rewards: list[float], group: int, scale: str) -> list[float]:
    """TRL 0.24 GRPO advantages: reward minus its group mean, over the (unbiased) std + 1e-4."""
    batch_std = _std(rewards)
    advantages = []
    for start in range(0, len(rewards), group):
        chunk = rewards[start : start + group]
        mean = sum(chunk) / len(chunk)
        std = batch_std if scale == "batch" else _std(chunk)
        advantages += [
            (reward - mean) if scale == "none" else (reward - mean) / (std + 1e-4)
            for reward in chunk
        ]
    return advantages


def adapter_state_sha256(state: dict) -> str:
    """Identity of adapter tensors as saved: sorted names, shapes, dtypes and bytes."""
    import torch

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().contiguous().cpu()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def policy_sha256(model) -> str:
    from peft import get_peft_model_state_dict

    return adapter_state_sha256(get_peft_model_state_dict(model))


def saved_policy_sha256(directory: Path) -> str:
    from safetensors.torch import load_file

    return adapter_state_sha256(load_file(directory / "adapter_model.safetensors"))


def _release_cache(device: str) -> None:
    """Return cached accelerator blocks between generation and updates (bounds host memory)."""
    import torch

    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


class WorldGateway:
    """Client of the orchestrator's loopback world gateway; it holds no actor credentials."""

    def __init__(self, timeout_seconds: int):
        url, token = os.environ.get(GATEWAY_URL), os.environ.get(GATEWAY_TOKEN)
        if not url or not token:
            raise ValueError("GRPO rewards require the orchestrator's world gateway")
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "::1", "localhost"):
            raise ValueError("The world gateway must be a loopback HTTP address")
        self.url = url.rstrip("/") + "/v1/rollouts"
        self.token = token
        self.timeout = timeout_seconds
        # Loopback only: never route rollouts through an environment-configured proxy.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def score(self, payload: dict, deadline: float) -> dict:
        """Retries transport failures with identical bytes; gateway and actor deduplicate."""
        body = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
        delay = 0.25
        while True:
            request = urllib.request.Request(
                self.url,
                data=body,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json",
                },
            )
            remaining = deadline - time.monotonic()
            try:
                with self.opener.open(
                    request, timeout=max(1.0, min(self.timeout, remaining))
                ) as reply:
                    return json.loads(reply.read())
            except urllib.error.HTTPError as error:
                detail = error.read(2000).decode("utf-8", "replace")
                if error.code != 503 or time.monotonic() + delay >= deadline:
                    raise RuntimeError(
                        f"World gateway refused rollouts (HTTP {error.code}): {detail}"
                    ) from None
            except (urllib.error.URLError, OSError) as error:
                if time.monotonic() + delay >= deadline:
                    raise RuntimeError(f"World gateway unreachable: {error}") from None
            time.sleep(delay)
            delay = min(delay * 2, 5.0)


class WorldRollouts:
    """TRL reward function and lineage recorder. Each reward is a world transition's reward.

    TRL calls it once per update, right after generating with the current weights; the
    trainer then reports the advantages it computed from these rewards, and the update
    callback records the weights the optimizer produced.
    """

    def __init__(self, config: ExperimentConfig, output: Path, events: EventLog, started: float):
        self.__name__ = "terse_world_reward"
        self.config, self.events, self.started = config, events, started
        self.gateway = WorldGateway(config.grpo.request_timeout_seconds)
        self.model = None
        self.prompt_tokens: dict[str, int] = {}
        self.lineage: list[str] = []
        self.updates: list[dict] = []
        self.snapshots: dict[str, str] = {}
        self.pending: list[dict] | None = None
        self.consumed_tokens = 0
        self.output = output
        self.rollouts_path = output / "rollouts.jsonl"
        self.metrics_path = output / "training-metrics.jsonl"

    def deadline(self) -> float:
        return self.started + self.config.training.max_seconds

    def check_deadline(self) -> None:
        if time.monotonic() >= self.deadline():
            raise TimeoutError("Training deadline reached; no completed checkpoint emitted")

    def snapshot(self, update: int) -> None:
        """Adapter state after `update` (0: initial), saved for independent lineage checks."""
        directory = self.output / "lineage" / f"update-{update:05d}"
        directory.mkdir(parents=True)
        self.model.save_pretrained(directory, safe_serialization=True)
        saved = saved_policy_sha256(directory)
        if saved != self.lineage[update]:
            raise ValueError("Saved adapter snapshot differs from the in-memory policy")
        self.snapshots[str(update)] = saved

    def __call__(self, prompts, completions, completion_ids, task_id, trainer_state, **_) -> list:
        self.check_deadline()
        grpo = self.config.grpo
        update = trainer_state.global_step + 1
        if self.pending is not None or update != len(self.updates) + 1:
            raise RuntimeError("Each update must use exactly one freshly scored rollout batch")
        policy = policy_sha256(self.model)
        # The rollout lineage: these completions came from the weights the last update made.
        if policy != self.lineage[-1]:
            raise ValueError("Rollouts were not generated by the latest policy")
        for start in range(0, len(task_id), grpo.num_generations):
            if len(set(task_id[start : start + grpo.num_generations])) != 1:
                raise ValueError("A rollout group mixes tasks")
        tokens = sum(
            self.prompt_tokens[task] + len(ids)
            for task, ids in zip(task_id, completion_ids, strict=True)
        )
        if self.consumed_tokens + tokens > self.config.training.max_training_tokens:
            raise ValueError("Training token budget would be exceeded")
        records = [
            {
                "update": update,
                "index": index,
                "rollout_id": f"u{update:05d}-r{index:03d}",
                "group": index // grpo.num_generations,
                "task_id": task,
                "policy_sha256": policy,
                "prompt_sha256": sha256_text(prompt),
                "completion": text,
                "completion_sha256": sha256_text(text),
                "completion_token_ids": list(ids),
                "action": answer_action(text),
            }
            for index, (prompt, text, ids, task) in enumerate(
                zip(prompts, completions, completion_ids, task_id, strict=True)
            )
        ]
        response = self.gateway.score(
            {
                "schemaVersion": 1,
                "campaignId": self.config.campaign_id,
                "experimentId": self.config.experiment_id,
                "update": update,
                "policySha256": policy,
                "rollouts": [
                    {
                        "rolloutId": record["rollout_id"],
                        "taskId": record["task_id"],
                        "action": record["action"],
                        "completionSha256": record["completion_sha256"],
                    }
                    for record in records
                ],
            },
            self.deadline(),
        )
        results = response.get("results") if isinstance(response, dict) else None
        if not isinstance(results, list) or len(results) != len(records):
            raise ValueError("World gateway returned a different number of transitions")
        for record, result in zip(records, results, strict=True):
            fields = ("worldId", "episodeId", "actionId", "stateSha256")
            if (
                not isinstance(result, dict)
                or result.get("rolloutId") != record["rollout_id"]
                or type(result.get("reward")) is not int
                or result["reward"] not in (0, 1)
                or result.get("terminated") is not True
                or not all(isinstance(result.get(field), str) for field in fields)
                or type(result.get("stateVersion")) is not int
            ):
                raise ValueError("World gateway returned an invalid transition")
            record["transition"] = {
                key: result[key]
                for key in (*fields, "stateVersion", "reward", "terminated", "truncated")
            }
            record["transition"]["replayed"] = result.get("replayed") is True
            record["reward"] = result["reward"]
        self.consumed_tokens += tokens
        self.pending = records
        _release_cache(self.config.device)
        return [float(record["reward"]) for record in records]

    def record_advantages(self, advantages: list[float]) -> None:
        """Called by the trainer with the advantages it will optimize; must match the rewards."""
        records, self.pending = self.pending, None
        if records is None or len(advantages) != len(records):
            raise RuntimeError("Advantages do not correspond to the scored rollout batch")
        grpo = self.config.grpo
        rewards = [float(record["reward"]) for record in records]
        expected = group_advantages(rewards, grpo.num_generations, grpo.scale_rewards)
        if any(abs(a - b) > 1e-4 for a, b in zip(advantages, expected, strict=True)):
            raise ValueError("Trainer advantages do not follow from the world rewards")
        lines = []
        for record, advantage in zip(records, advantages, strict=True):
            record["advantage"] = advantage
            lines.append(json.dumps(record, sort_keys=True, allow_nan=False))
        with self.rollouts_path.open("a", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        groups = [
            rewards[start : start + grpo.num_generations]
            for start in range(0, len(rewards), grpo.num_generations)
        ]
        summary = {
            "update": records[0]["update"],
            "rollouts": len(records),
            "reward_sum": int(sum(rewards)),
            "reward_mean": sum(rewards) / len(rewards),
            "rewarded_groups": sum(1 for group in groups if max(group) > 0),
            "zero_signal_groups": sum(1 for group in groups if min(group) == max(group)),
            "rollouts_sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest(),
            "rollout_policy_sha256": records[0]["policy_sha256"],
            "training_tokens": self.consumed_tokens,
        }
        self.updates.append(summary)
        self.events.emit(
            "ROLLOUTS_SCORED",
            {
                "update": summary["update"],
                "rollouts": summary["rollouts"],
                "rewardSum": summary["reward_sum"],
                "rolloutPolicySha256": summary["rollout_policy_sha256"],
                "rolloutsSha256": summary["rollouts_sha256"],
                "rewardSource": REWARD_SOURCE,
            },
        )

    def updated(self, step: int) -> None:
        """After optimizer update `step`: the produced policy feeds the next rollout batch."""
        if step != len(self.updates) or len(self.lineage) != step:
            raise RuntimeError("Optimizer update without exactly one scored rollout batch")
        after = policy_sha256(self.model)
        self.lineage.append(after)
        self.updates[-1]["policy_sha256_after"] = after
        every = self.config.grpo.snapshot_every
        if every and (step % every == 0 or step == self.config.training.max_steps):
            self.snapshot(step)
        _release_cache(self.config.device)

    def logged(self, step: int, logs: dict) -> None:
        if not self.updates or self.updates[-1]["update"] != step:
            raise RuntimeError("Training metrics do not match the latest update")
        loss, grad_norm = logs.get("loss"), logs.get("grad_norm")
        if not all(
            isinstance(value, int | float) and math.isfinite(value) for value in (loss, grad_norm)
        ):
            raise ValueError("Non-finite GRPO training metric")
        update = self.updates[-1]
        # TRL's own reward statistic must be the world rewards it was handed.
        if abs(float(logs.get("reward", math.nan)) - update["reward_mean"]) > 1e-6:
            raise ValueError("Trainer reward differs from the world rewards")
        metric = {
            "step": step,
            "loss": loss,
            "grad_norm": grad_norm,
            "learning_rate": logs.get("learning_rate"),
            "rollouts": update["rollouts"],
            "reward_sum": update["reward_sum"],
            "reward_mean": update["reward_mean"],
            "rewarded_groups": update["rewarded_groups"],
            "zero_signal_groups": update["zero_signal_groups"],
            "training_tokens": update["training_tokens"],
            "rollout_policy_sha256": update["rollout_policy_sha256"],
            "policy_sha256_after": update["policy_sha256_after"],
            "trl": {key: logs[key] for key in TRL_METRICS if key in logs},
            "elapsed_seconds": time.monotonic() - self.started,
            "timestamp": utc_now(),
        }
        with self.metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metric, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
        self.events.emit("TRAINING_PROGRESS", metric)


def _trainer_class():
    from trl import GRPOTrainer

    class ForgeGRPOTrainer(GRPOTrainer):
        """Reports the exact advantages each update optimizes (pinned TRL internals)."""

        forge_rollouts: WorldRollouts

        def _generate_and_score_completions(self, inputs):
            output = super()._generate_and_score_completions(inputs)
            self.forge_rollouts.record_advantages(output["advantages"].tolist())
            return output

    return ForgeGRPOTrainer


def _callback(rollouts: WorldRollouts):
    from transformers import TrainerCallback

    class LineageCallback(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            rollouts.check_deadline()

        def on_step_end(self, args, state, control, **kwargs):
            rollouts.updated(state.global_step)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                rollouts.logged(state.global_step, logs)

    return LineageCallback()


def _load_parent(model, config: ExperimentConfig, path: Path, sha256: str):
    """Continue an existing adapter, which must match this run's base model and LoRA settings."""
    from peft import PeftModel

    if directory_sha256(path) != sha256:
        raise ValueError("Starting adapter checksum mismatch")
    settings = json.loads((path / "adapter_config.json").read_text())
    spec = config.training
    if (
        settings.get("base_model_name_or_path") != config.model.model_id
        or settings.get("revision") != config.model.revision
    ):
        raise ValueError("Starting adapter belongs to a different base model")
    if (
        settings.get("r") != spec.lora_rank
        or settings.get("lora_alpha") != spec.lora_alpha
        or sorted(settings.get("target_modules") or []) != sorted(spec.target_modules)
        or settings.get("lora_dropout") != 0.0
    ):
        raise ValueError("Starting adapter's LoRA settings differ from the training settings")
    return PeftModel.from_pretrained(model, str(path), is_trainable=True)


def train_grpo(config: ExperimentConfig, tasks: list[Task], output: Path, events: EventLog) -> dict:
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from trl import GRPOConfig

    if version("trl") != TRL_VERSION:
        raise ValueError(f"GRPO requires the pinned trl=={TRL_VERSION}")
    started = time.monotonic()
    spec, grpo = config.training, config.grpo
    rollouts = WorldRollouts(config, output, events, started)
    seed_everything(config.seed)
    model, tokenizer = load_base(config)
    if "token_type_ids" in tokenizer.model_input_names:
        # TRL 0.24 generates with them but computes the loss without them.
        raise ValueError("GRPO needs a tokenizer without token type IDs (on-policy loss)")
    context_limit = getattr(model.config, "max_position_embeddings", None)
    rows = []
    for task in tasks:
        prompt = prompt_for(tokenizer, task.question)
        length = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        # TRL would silently truncate long prompts; refuse them instead.
        if length > grpo.max_prompt_tokens:
            raise ValueError(f"Training prompt exceeds token limit (no truncation): {task.id}")
        if context_limit and length + grpo.max_completion_tokens > context_limit:
            raise ValueError(f"Rollout would exceed model context window: {task.id}")
        rollouts.prompt_tokens[task.id] = length
        rows.append({"prompt": prompt, "task_id": task.id})
    parent = grpo.init_adapter
    if parent is None:
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
    else:
        model = _load_parent(model, config, Path(parent.path), parent.sha256)
    model.peft_config["default"].base_model_name_or_path = config.model.model_id
    model.peft_config["default"].revision = config.model.revision
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("LoRA configuration did not create trainable parameters")
    before = parameter_sha256(model)
    rollouts.model = model
    rollouts.lineage.append(policy_sha256(model))
    # Lineage starts exactly at the parent's saved weights.
    if parent is not None and rollouts.lineage[0] != saved_policy_sha256(Path(parent.path)):
        raise ValueError("Loaded starting adapter differs from its saved weights")
    rollouts.snapshot(0)
    args = GRPOConfig(
        output_dir=str(output / "trainer-scratch"),
        seed=config.seed,
        data_seed=config.seed,
        max_steps=spec.max_steps,
        per_device_train_batch_size=spec.batch_size,
        gradient_accumulation_steps=spec.gradient_accumulation_steps,
        # One rollout batch per update, used once: rollouts are always on-policy.
        steps_per_generation=spec.gradient_accumulation_steps,
        num_iterations=1,
        num_generations=grpo.num_generations,
        max_prompt_length=grpo.max_prompt_tokens,
        max_completion_length=grpo.max_completion_tokens,
        temperature=grpo.temperature,
        top_p=grpo.top_p,
        beta=grpo.beta,
        epsilon=grpo.epsilon,
        loss_type=grpo.loss_type,
        scale_rewards=grpo.scale_rewards,
        learning_rate=spec.learning_rate,
        lr_scheduler_type="constant",
        warmup_steps=0,
        weight_decay=0.0,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        gradient_checkpointing=grpo.gradient_checkpointing,
        bf16=config.precision == "bfloat16",
        fp16=False,
        use_cpu=config.device == "cpu",
        disable_dropout=True,
        shuffle_dataset=True,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        disable_tqdm=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        log_completions=False,
    )
    if args.device.type != config.device:
        raise ValueError(f"Trainer selected {args.device.type}, not the configured {config.device}")
    trainer = _trainer_class()(
        model=model,
        reward_funcs=[rollouts],
        args=args,
        train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
        callbacks=[_callback(rollouts)],
    )
    trainer.forge_rollouts = rollouts
    events.emit(
        "TRAINING_STARTED",
        {
            "method": "grpo",
            "maxSteps": spec.max_steps,
            "rolloutsPerUpdate": args.generation_batch_size,
            "numGenerations": grpo.num_generations,
            "rewardSource": REWARD_SOURCE,
            "trainableParameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        },
    )
    trainer.train()
    rollouts.check_deadline()
    if trainer.state.global_step != spec.max_steps or len(rollouts.updates) != spec.max_steps:
        raise ValueError("GRPO stopped before completing its configured updates")
    scratch = output / "trainer-scratch"
    if scratch.exists():
        if any(scratch.iterdir()):
            raise ValueError("The trainer wrote unexpected files into the run directory")
        scratch.rmdir()
    after = parameter_sha256(model)
    if after == before:
        raise ValueError("Optimization did not change trainable weights")
    final = policy_sha256(model)
    if final != rollouts.lineage[-1]:
        raise ValueError("Final policy differs from the last recorded update")
    checkpoint = output / "checkpoint"
    checkpoint.mkdir()
    model.save_pretrained(checkpoint, safe_serialization=True)
    if saved_policy_sha256(checkpoint) != final:
        raise ValueError("Saved checkpoint differs from the final policy")
    files = directory_manifest(checkpoint)
    rewards = [update["reward_sum"] for update in rollouts.updates]
    summary = {
        "method": "grpo",
        "trainer": {"library": "trl", "version": TRL_VERSION, "class": "GRPOTrainer"},
        "reward_source": REWARD_SOURCE,
        "reward_version": grpo.reward_version,
        "action_parser": grpo.action_parser,
        "grpo": grpo.model_dump(),
        "rollouts_per_update": args.generation_batch_size,
        "prompts_per_update": args.generation_batch_size // grpo.num_generations,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": spec.learning_rate,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
            "max_gradient_norm": 1.0,
            "scheduler": "constant",
        },
        "optimizer_steps": len(rollouts.updates),
        "rollouts": sum(update["rollouts"] for update in rollouts.updates),
        "rewarded_rollouts": sum(rewards),
        "first_update_reward_mean": rollouts.updates[0]["reward_mean"],
        "last_update_reward_mean": rollouts.updates[-1]["reward_mean"],
        "updates": rollouts.updates,
        "starting_checkpoint_sha256": parent.sha256 if parent is not None else None,
        "lineage": {
            "initial_policy_sha256": rollouts.lineage[0],
            "final_policy_sha256": final,
            "snapshots": rollouts.snapshots,
        },
        "training_tokens": rollouts.consumed_tokens,
        "initial_trainable_sha256": before,
        "final_trainable_sha256": after,
        "weights_changed": True,
        "checkpoint_sha256": manifest_sha256(files),
        "checkpoint_files": files,
        "checkpoint_path": str(checkpoint),
        "elapsed_seconds": time.monotonic() - started,
        "cost_usd": None,
        "cost_status": "not_measured_by_local_worker",
    }
    write_json(output / "training.json", summary)
    events.emit(
        "TRAINING_COMPLETED",
        {
            "checkpointSha256": summary["checkpoint_sha256"],
            "optimizerSteps": len(rollouts.updates),
            "rollouts": summary["rollouts"],
            "rewardedRollouts": summary["rewarded_rollouts"],
            "weightsChanged": True,
        },
    )
    return summary


def verify_rollouts(output: Path, config: ExperimentConfig, training: dict) -> dict:
    """Re-derive actions, rewards, advantages and lineage of a completed GRPO run from files.

    Rewards are checked against the recorded world transitions here; the orchestrator
    separately checks those transitions against the world actors' own journals.
    """
    grpo = config.grpo
    if grpo is None or training.get("reward_source") != REWARD_SOURCE:
        raise ValueError("Run is not a world-rewarded GRPO run")
    tasks, _ = load_tasks(config.train)
    task_ids = {task.id for task in tasks}
    per_update = training["rollouts_per_update"]
    if per_update != config.training.batch_size * config.training.gradient_accumulation_steps:
        raise ValueError("Rollouts per update differ from the configuration")
    lines = (output / "rollouts.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    steps = config.training.max_steps
    if len(records) != steps * per_update or len(training["updates"]) != steps:
        raise ValueError("Rollout count differs from the configured updates")
    lineage = training["lineage"]
    policy = lineage["initial_policy_sha256"]
    parent = grpo.init_adapter
    if training.get("starting_checkpoint_sha256") != (parent.sha256 if parent else None):
        raise ValueError("Training metadata names a different starting adapter")
    if parent is not None and saved_policy_sha256(Path(parent.path)) != policy:
        raise ValueError("Lineage does not start at the starting adapter's weights")
    for number, update in enumerate(training["updates"], 1):
        batch = records[(number - 1) * per_update : number * per_update]
        if update["update"] != number or update["rollout_policy_sha256"] != policy:
            raise ValueError(f"Update {number} rollouts were not generated by the prior policy")
        rewards = []
        for index, record in enumerate(batch):
            transition = record["transition"]
            if (
                record["update"] != number
                or record["index"] != index
                or record["rollout_id"] != f"u{number:05d}-r{index:03d}"
                or record["group"] != index // grpo.num_generations
                or record["policy_sha256"] != policy
                or record["task_id"] not in task_ids
                or record["completion_sha256"] != sha256_text(record["completion"])
                or record["action"] != answer_action(record["completion"])
                or record["reward"] not in (0, 1)
                or transition["reward"] != record["reward"]
                or transition["terminated"] is not True
            ):
                raise ValueError(f"Rollout {record.get('rollout_id')} is inconsistent")
            rewards.append(float(record["reward"]))
        for start in range(0, per_update, grpo.num_generations):
            if (
                len({record["task_id"] for record in batch[start : start + grpo.num_generations]})
                != 1
            ):
                raise ValueError(f"Update {number} mixes tasks within a group")
        expected = group_advantages(rewards, grpo.num_generations, grpo.scale_rewards)
        if any(
            abs(record["advantage"] - value) > 1e-4
            for record, value in zip(batch, expected, strict=True)
        ):
            raise ValueError(f"Update {number} advantages do not follow from its rewards")
        digest = hashlib.sha256(
            "\n".join(lines[(number - 1) * per_update : number * per_update]).encode()
        ).hexdigest()
        if update["rollouts_sha256"] != digest or update["reward_sum"] != int(sum(rewards)):
            raise ValueError(f"Update {number} summary differs from its rollouts")
        policy = update["policy_sha256_after"]
    if policy != lineage["final_policy_sha256"]:
        raise ValueError("Lineage does not end at the final policy")
    metrics = [
        json.loads(line) for line in (output / "training-metrics.jsonl").read_text().splitlines()
    ]
    if [(m["step"], m["reward_sum"], m["policy_sha256_after"]) for m in metrics] != [
        (u["update"], u["reward_sum"], u["policy_sha256_after"]) for u in training["updates"]
    ]:
        raise ValueError("Training metrics differ from the recorded updates")
    expected_snapshots = {"0": lineage["initial_policy_sha256"]} | {
        str(u["update"]): u["policy_sha256_after"]
        for u in training["updates"]
        if grpo.snapshot_every and (u["update"] % grpo.snapshot_every == 0 or u["update"] == steps)
    }
    if lineage["snapshots"] != expected_snapshots:
        raise ValueError("Lineage snapshots differ from the recorded updates")
    saved = sorted(path.name for path in (output / "lineage").iterdir())
    if saved != [f"update-{int(update):05d}" for update in sorted(expected_snapshots, key=int)]:
        raise ValueError("Lineage snapshot directories differ from the record")
    for update, digest in expected_snapshots.items():
        if saved_policy_sha256(output / "lineage" / f"update-{int(update):05d}") != digest:
            raise ValueError(f"Lineage snapshot {update} differs from its recorded policy")
    if saved_policy_sha256(output / "checkpoint") != lineage["final_policy_sha256"]:
        raise ValueError("Checkpoint is not the final policy of the lineage")
    return {
        "updates": steps,
        "rollouts": len(records),
        "rewarded_rollouts": sum(record["reward"] for record in records),
        "snapshots": len(expected_snapshots),
    }
