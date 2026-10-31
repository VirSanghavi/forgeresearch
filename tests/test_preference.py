"""DPO and preference accuracy on a tiny offline model: real optimizer steps, no download."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("trl")

from forge_training.fixtures import _tiny_model  # noqa: E402
from services.core import packs  # noqa: E402
from services.core.packs.spec import TaskPack  # noqa: E402
from services.core.preference import score_pairs, train_dpo  # noqa: E402

TEMPLATE = (
    "{% for m in messages %}{{ m['content'] }} {% endfor %}{% if add_generation_prompt %}Answer : {% endif %}"
)


def tiny(*_args):
    model, tokenizer = _tiny_model()
    tokenizer.chat_template = TEMPLATE
    model.forge_model_id = None
    return model, tokenizer


def preference_pack():
    base = packs.get().model_dump(mode="json")
    return TaskPack.model_validate(
        base
        | {
            "id": "tiny-prefs",
            "kind": "preference",
            "capabilities": base["capabilities"] | {"dpo": True, "terse_world": False, "grpo": False},
        }
    )


ROWS = [
    {
        "id": f"p-{i}",
        "question": f"What is {i} plus 1 ?",
        "answer": f"#### {i + 1}",
        "chosen": f"#### {i + 1}",
        "rejected": f"#### {i + 7}",
    }
    for i in range(1, 9)
]


def test_preference_accuracy_is_a_per_pair_comparison():
    model, tokenizer = tiny()
    result = score_pairs(model, tokenizer, ROWS[:4], lambda *a: None, lambda: False, pack=preference_pack())
    assert result["n"] == 4 and set(result["predictions"][0]) >= {"margin", "logp_chosen", "logp_rejected"}
    for p in result["predictions"]:
        assert p["correct"] == (p["logp_chosen"] > p["logp_rejected"])
    assert result["decoding"]["method"].startswith("teacher-forced")


def test_dpo_updates_weights_and_records_reward_margins(tmp_path):
    pack = preference_pack()
    config = {"seed": 7, "train_steps": 12, "model_id": None}
    proposal = {"strategy": "dpo", "learning_rate": 0.001, "lora_rank": 8}
    seen = []
    result = train_dpo(
        proposal,
        config,
        ROWS,
        tmp_path / "exp",
        seen.append,
        lambda: False,
        pack,
        loader=tiny,
        lora_targets=["c_attn"],
    )
    assert result["weights_changed"] and result["algorithm"] == "TRL DPO"
    assert result["learning_rate"] == 1e-4 and result["learning_rate_clamped_from"] == 0.001
    assert len(seen) == 12 and all("reward_margin" in m for m in seen)
    assert (tmp_path / "exp" / "adapter_model.safetensors").exists()


def test_dpo_refuses_data_without_pairs(tmp_path):
    with pytest.raises(ValueError, match="No preference pairs"):
        train_dpo(
            {"strategy": "dpo", "learning_rate": 1e-4, "lora_rank": 8},
            {"seed": 7, "train_steps": 2, "model_id": None},
            [{"id": "x", "question": "q", "answer": "a"}],
            tmp_path / "exp",
            lambda m: None,
            lambda: False,
            preference_pack(),
            loader=tiny,
            lora_targets=["c_attn"],
        )
