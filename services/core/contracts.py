from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

import json
import os

from services.core.model_catalog import CATALOG

# Models added by name in the command center, kept beside the rest of the lab's data.
REGISTERED = Path(__file__).resolve().parents[2] / "data" / "models.json"
# Above this a model trains on a cloud GPU only: a 16 GB laptop swaps hard beyond about 1B.
LOCAL_PARAMETER_LIMIT = int(os.environ.get("FORGE_LOCAL_MODEL_MAX_PARAMETERS", 1_200_000_000))
# The largest model that fits the 80 GB cloud GPU with room for LoRA and generation.
MODEL_PARAMETER_LIMIT = 20_000_000_000

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"

# Every model a campaign may train, with everything that differs between them. The 0.5B entry
# is exactly the behaviour Forge has always had.
MODELS = {
    MODEL_ID: {
        "revision": MODEL_REVISION,
        "label": "Qwen2.5 · 0.5B",
        "parameters": "494M",
        "dtype": "float16",
        "lora_targets": ["q_proj", "v_proj"],
        "chat_template_kwargs": {},
        "max_new_tokens": 256,
        "remote_only": False,
    },
    # Hybrid text tower: 24 gated delta (linear attention) layers and 8 full-attention layers.
    # q_proj/v_proj exist only in the 8 full-attention layers, so LoRA also targets the
    # linear-attention input projection to adapt every layer. Weights ship in bfloat16; float16
    # is not used. Thinking is off: the chat template's documented non-thinking mode emits an
    # empty <think></think>, and the same "#### <number>" scorer reads the direct answer.
    "Qwen/Qwen3.5-4B": {
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "label": "Qwen3.5 · 4B",
        "parameters": "4.66B",
        "dtype": "bfloat16",
        "lora_targets": ["q_proj", "v_proj", "in_proj_qkv"],
        "chat_template_kwargs": {"enable_thinking": False},
        "max_new_tokens": 512,
        "remote_only": True,
    },
}
# Specs that arrived with a remote job; the GPU container has no local registry.
_ADOPTED = {}


def registered_models():
    try:
        return json.loads(REGISTERED.read_text())
    except FileNotFoundError:
        return {}


def all_models():
    """Hand-written entries win over the generated catalog, which wins over added models."""
    found = {}
    for source, custom in ((registered_models(), True), (CATALOG, False)):
        for model_id, spec in source.items():
            count = spec["parameter_count"]
            found[model_id] = spec | {"remote_only": count > LOCAL_PARAMETER_LIMIT, "custom": custom}
    return found | {model_id: spec | {"custom": False} for model_id, spec in MODELS.items()}


def model_spec(model_id=None):
    model_id = model_id or MODEL_ID
    return (_ADOPTED.get(model_id) or all_models()[model_id]) | {"id": model_id}


def adopt_model(spec):
    _ADOPTED[spec["id"]] = {k: v for k, v in spec.items() if k != "id"}


def register_model(spec):
    """Remember a model at its pinned commit. Adding it again never moves the pin."""
    current = registered_models()
    if spec["id"] in all_models():
        return model_spec(spec["id"])
    current[spec["id"]] = {k: v for k, v in spec.items() if k != "id"}
    temporary = REGISTERED.with_suffix(".tmp")
    temporary.write_text(json.dumps(current, indent=2))
    temporary.replace(REGISTERED)
    return model_spec(spec["id"])


DEFAULT_MODEL_ID = os.environ.get("FORGE_MODEL_ID", MODEL_ID)
if DEFAULT_MODEL_ID not in all_models():
    raise ValueError(f"FORGE_MODEL_ID is not a known model: {DEFAULT_MODEL_ID!r}")


class CampaignConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(default="Mathematical reasoning", min_length=3, max_length=80)
    model_id: str = DEFAULT_MODEL_ID
    benchmark: Literal["gsm8k-subset"] = "gsm8k-subset"
    # The task this campaign improves. Built-in: gsm8k; uploaded packs are checked by the API.
    task_pack: str = Field(default="gsm8k", pattern=r"^[a-z0-9][a-z0-9-]{1,63}$")
    final_backend: Literal["shared_scorer", "lm_eval_azure"] = "shared_scorer"
    objective: str = Field(
        default="Improve multi-step mathematical reasoning without benchmark leakage.",
        min_length=10,
        max_length=500,
    )
    max_seconds: int = Field(default=1800, ge=120, le=14400)
    max_experiments: int = Field(default=2, ge=1, le=4)
    train_steps: int = Field(default=24, ge=4, le=200)
    dev_samples: int = Field(default=12, ge=4, le=100)
    final_samples: int = Field(default=12, ge=4, le=100)
    seed: int = Field(default=42, ge=0, le=2147483647)
    strategies: list[Literal["broad_sft", "targeted_sft", "distill_sft", "hard_sft", "grpo", "dpo"]] = Field(
        default_factory=lambda: ["broad_sft", "targeted_sft"], min_length=1
    )
    # A teacher dataset prepared for this task pack (teacher distillation).
    synthetic_dataset: str | None = Field(default=None, pattern=r"^syn-[0-9a-f]{8}$")
    production_scan: str | None = Field(default=None, pattern=r"^scan-[0-9a-f]{8}$")

    @field_validator("model_id")
    @classmethod
    def known_model(cls, value):
        if value not in all_models():
            raise ValueError(f"{value} is not in the catalog. Add it first.")
        return value


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    campaign_id: str
    sender: Literal["director", "data", "trainer", "evaluator", "governance", "hephaestus"]
    recipient: Literal["director", "data", "trainer", "evaluator", "governance", "hephaestus"]
    category: str
    text: str
    experiment_id: str | None = None
    # "dm": a 1:1 handoff in the recipient's Relay inbox. "room": a decision or announcement
    # in the campaign's research room; with reply_to, a thread reply under that room message.
    route: Literal["dm", "room"] = "dm"
    reply_to: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]{1,64}$")


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")
    episode_id: str
    action_id: str = Field(min_length=1, max_length=100)
    expected_version: int = Field(ge=0)
    answer: str = Field(max_length=100)
