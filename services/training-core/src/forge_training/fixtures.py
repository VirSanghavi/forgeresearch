"""Offline tiny models for mechanics tests, never benchmark evidence."""

from __future__ import annotations

from pathlib import Path

from .artifacts import directory_sha256, write_json
from .config import Example, Task
from .data import write_dataset


def _tiny_model(**tokenizer_options):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    vocabulary = ["<pad>", "<eos>", "<unk>", "Question", "Answer", ":", "####", ".", "+", "?"]
    vocabulary += [str(i) for i in range(32)]
    vocabulary += "Solve the problem End with a final line in form number What is plus".split()
    vocabulary = list(dict.fromkeys(vocabulary))
    backend = Tokenizer(
        WordLevel(dict(zip(vocabulary, range(len(vocabulary)), strict=True)), unk_token="<unk>")
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
        **tokenizer_options,
    )
    torch.manual_seed(7)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(vocabulary),
            n_positions=256,
            n_ctx=256,
            n_embd=32,
            n_layer=1,
            n_head=2,
            bos_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    )
    return model, tokenizer


def _sums(name: str, numbers) -> list[Example]:
    return [
        Example(id=f"{name}-{i}", question=f"What is {i} plus 1?", answer=f"#### {i + 1}")
        for i in numbers
    ]


def _write_synthetic(directory: Path, name: str, rows, purpose: str | None = None) -> None:
    write_dataset(
        directory,
        name,
        rows,
        source="forge-offline-fixture",
        revision="generator-v1",
        source_split="synthetic",
        license_name="CC0-1.0",
        purpose=purpose,
    )


def make_smoke_fixture(output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=False)
    model_dir = output / "base-model"
    model_dir.mkdir()
    model, tokenizer = _tiny_model()
    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir, safe_serialization=True)
    data_dir = output / "data"
    data_dir.mkdir()
    for name, numbers in (("train", range(1, 5)), ("development", range(5, 7))):
        _write_synthetic(data_dir, name, _sums(name, numbers))
    config = {
        "campaign_id": "mechanics-smoke",
        "experiment_id": "tiny-lora",
        "model": {
            "source": "local",
            "model_status": "fixture",
            "model_id": "base-model",
            "revision": directory_sha256(model_dir),
        },
        "train": {"records": "data/train.jsonl", "manifest": "data/train.manifest.json"},
        "development": {
            "records": "data/development.jsonl",
            "manifest": "data/development.manifest.json",
        },
        "training": {
            "max_steps": 3,
            "gradient_accumulation_steps": 1,
            "max_sequence_length": 128,
            "lora_rank": 4,
            "lora_alpha": 8,
            "target_modules": ["c_attn"],
            "max_seconds": 120,
        },
        "evaluation": {"max_new_tokens": 8, "max_prompt_tokens": 128, "max_seconds": 120},
        "device": "cpu",
        "precision": "float32",
        "output_dir": "run",
    }
    config_path = output / "experiment.json"
    write_json(config_path, config)
    return config_path


# Train, development and regression problems of the GRPO fixture. The warm-started model
# answers each "What is i plus 1?" with one of i, i + 1 or i + 2, about a third each, and
# greedily answers 2 and 4 wrongly; those form the development set.
GRPO_TRAIN, GRPO_DEVELOPMENT, GRPO_REGRESSION = (1, 3, 5, 7), (2, 4), (6, 8)


def make_grpo_fixture(output: Path) -> Path:
    """A tiny model warm-started on the answer format, so world rewards vary within groups.

    Random weights never produce a valid final line, so every world reward would be zero
    and group-relative advantages would carry no signal. The warm start is full-parameter
    training on '#### k' answers for k in {i, i + 1, i + 2}; it is part of the fixture's base
    model, not campaign training. Mechanics only: this is not evidence about useful models.
    """
    import torch

    from .modeling import prompt_for

    output.mkdir(parents=True, exist_ok=False)
    model_dir = output / "base-model"
    model_dir.mkdir()
    # GPT-2 embeds token type IDs; TRL passes them when generating but not in its loss, so
    # a tokenizer that emits them would sample from a different policy than it optimizes.
    model, tokenizer = _tiny_model(model_input_names=["input_ids", "attention_mask"])
    sequences = []
    for i in range(1, 9):
        prompt = tokenizer.encode(
            prompt_for(tokenizer, f"What is {i} plus 1?"), add_special_tokens=False
        )
        for k in (i, i + 1, i + 2):
            answer = tokenizer.encode(f"#### {k}", add_special_tokens=False)
            answer.append(tokenizer.eos_token_id)
            sequences.append((prompt + answer, [-100] * len(prompt) + answer))
    width = max(len(ids) for ids, _ in sequences)
    batch = {
        "input_ids": torch.tensor([ids + [0] * (width - len(ids)) for ids, _ in sequences]),
        "labels": torch.tensor(
            [labels + [-100] * (width - len(labels)) for _, labels in sequences]
        ),
        "attention_mask": torch.tensor(
            [[1] * len(ids) + [0] * (width - len(ids)) for ids, _ in sequences]
        ),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    for _ in range(400):
        loss = model(**batch).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir, safe_serialization=True)
    data_dir = output / "data"
    data_dir.mkdir()
    train = _sums("train", GRPO_TRAIN)
    # The allowed partition, with answers, stays with the task service; the trainer reads
    # only its answer-free projection.
    _write_synthetic(data_dir, "train", train)
    _write_synthetic(
        data_dir, "tasks", [Task(id=row.id, question=row.question) for row in train], "train"
    )
    _write_synthetic(data_dir, "development", _sums("development", GRPO_DEVELOPMENT))
    _write_synthetic(data_dir, "regression", _sums("regression", GRPO_REGRESSION), "development")
    config = {
        "campaign_id": "grpo-mechanics",
        "experiment_id": "tiny-grpo",
        "model": {
            "source": "local",
            "model_status": "fixture",
            "model_id": "base-model",
            "revision": directory_sha256(model_dir),
        },
        "train": {"records": "data/tasks.jsonl", "manifest": "data/tasks.manifest.json"},
        "development": {
            "records": "data/development.jsonl",
            "manifest": "data/development.manifest.json",
        },
        "training": {
            "method": "grpo",
            "learning_rate": 0.01,
            "max_steps": 6,
            "batch_size": 4,
            "gradient_accumulation_steps": 2,
            "lora_rank": 4,
            "lora_alpha": 8,
            "target_modules": ["c_attn"],
            "max_seconds": 120,
            "max_training_tokens": 100000,
        },
        "grpo": {
            "num_generations": 4,
            "max_completion_tokens": 8,
            "max_prompt_tokens": 128,
            "max_rollouts": 256,
            "request_timeout_seconds": 30,
        },
        "evaluation": {"max_new_tokens": 8, "max_prompt_tokens": 128, "max_seconds": 120},
        "device": "cpu",
        "precision": "float32",
        "output_dir": "run",
    }
    config_path = output / "experiment.json"
    write_json(config_path, config)
    return config_path


# Questions whose answers are all 9 ("What is a plus 9 - a?"), split so that train,
# development and regression never share a question. The sealed final evaluator writes its
# own two questions (a = 0 and a = 9) and checks they appear in none of these.
CONSTANT_TRAIN, CONSTANT_DEVELOPMENT, CONSTANT_REGRESSION = (1, 2, 3, 4), (5, 6), (7, 8)


def _nines(name: str, numbers) -> list[Example]:
    return [
        Example(id=f"{name}-{a}", question=f"What is {a} plus {9 - a}?", answer="#### 9")
        for a in numbers
    ]


def make_constant_fixture(output: Path) -> Path:
    """A random tiny model and a task supervised training reliably improves.

    Every answer is 9, so a short SFT run lifts development accuracy from 0 to 1 on any
    platform; campaigns on it promote a checkpoint, which the freeze and sealed final
    evaluation need. Mechanics only: learning a constant is not evidence about useful models.
    """
    output.mkdir(parents=True, exist_ok=False)
    model_dir = output / "base-model"
    model_dir.mkdir()
    # Without token type IDs, training, evaluation and the final harness see the same inputs.
    model, tokenizer = _tiny_model(model_input_names=["input_ids", "attention_mask"])
    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir, safe_serialization=True)
    data_dir = output / "data"
    data_dir.mkdir()
    _write_synthetic(data_dir, "train", _nines("train", CONSTANT_TRAIN))
    _write_synthetic(data_dir, "development", _nines("development", CONSTANT_DEVELOPMENT))
    _write_synthetic(
        data_dir, "regression", _nines("regression", CONSTANT_REGRESSION), "development"
    )
    config = {
        "campaign_id": "constant-mechanics",
        "experiment_id": "constant-sft",
        "model": {
            "source": "local",
            "model_status": "fixture",
            "model_id": "base-model",
            "revision": directory_sha256(model_dir),
        },
        "train": {"records": "data/train.jsonl", "manifest": "data/train.manifest.json"},
        "development": {
            "records": "data/development.jsonl",
            "manifest": "data/development.manifest.json",
        },
        "training": {
            "learning_rate": 0.01,
            "max_steps": 48,
            "gradient_accumulation_steps": 1,
            "max_sequence_length": 128,
            "lora_rank": 4,
            "lora_alpha": 8,
            "target_modules": ["c_attn"],
            "max_seconds": 120,
        },
        "evaluation": {"max_new_tokens": 8, "max_prompt_tokens": 128, "max_seconds": 120},
        "device": "cpu",
        "precision": "float32",
        "output_dir": "run",
    }
    config_path = output / "experiment.json"
    write_json(config_path, config)
    return config_path
