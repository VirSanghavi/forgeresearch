from __future__ import annotations

import pytest
import torch

from forge_training.artifacts import EventLog
from forge_training.config import Example, ExperimentConfig, load_config
from forge_training.data import write_dataset
from forge_training.evaluation import compare, evaluate
from forge_training.fixtures import make_smoke_fixture
from forge_training.training import collate_completion, completion_loss, train


@pytest.fixture(scope="module")
def fixture_config(tmp_path_factory):
    return load_config(make_smoke_fixture(tmp_path_factory.mktemp("performance") / "fixture"))


def changed(config, **changes):
    return ExperimentConfig.model_validate(config.model_dump() | changes)


@pytest.mark.parametrize("architecture", ["gpt2", "qwen2"])
def test_selected_logits_preserve_loss_and_gradients(architecture):
    from transformers import GPT2Config, GPT2LMHeadModel, Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(4)
    if architecture == "gpt2":
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=32,
                n_positions=32,
                n_embd=16,
                n_layer=1,
                n_head=2,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
            )
        )
    else:
        model = Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
            )
        )
    batch = collate_completion(
        [
            {"input_ids": [1, 2, 3, 4, 5], "labels": [-100, -100, -100, 4, 5]},
            {"input_ids": [1, 2, 6], "labels": [-100, 2, 6]},
        ],
        0,
        "cpu",
    )
    full = completion_loss(model, batch, completion_only_logits=False)
    full.backward()
    gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    selected = completion_loss(model, batch, completion_only_logits=True)
    selected.backward()
    torch.testing.assert_close(full, selected)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, gradients[name], atol=1e-6, rtol=1e-4)


@pytest.mark.integration
def test_accumulation_matches_a_full_variable_length_batch(fixture_config, tmp_path):
    from safetensors.torch import load_file

    rows = [
        Example(id="short", question="What is 1 plus 1?", answer="#### 2"),
        Example(id="long", question="What is 2 plus 1?", answer="2 plus 1 is 3. #### 3"),
    ]
    states = []
    for batch_size, accumulation in [(2, 1), (1, 2)]:
        output = tmp_path / f"batch-{batch_size}"
        output.mkdir()
        config = changed(
            fixture_config,
            training=fixture_config.training.model_dump()
            | {
                "max_steps": 1,
                "batch_size": batch_size,
                "gradient_accumulation_steps": accumulation,
                "completion_only_logits": True,
            },
        )
        summary = train(config, rows, output, EventLog(output / "events.jsonl", "test", "test"))
        assert summary["supervised_tokens"] > 0
        states.append(load_file(output / "checkpoint/adapter_model.safetensors"))
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], atol=1e-6, rtol=1e-4)


@pytest.mark.integration
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_real_batched_generation_matches_serial(fixture_config, tmp_path, attention):
    rows = [
        Example(id="a", question="What is 1 plus 1?", answer="#### 2"),
        Example(id="b", question="2?", answer="#### 2"),
        Example(id="c", question="What is 3 plus 1 plus 1?", answer="#### 5"),
    ]
    write_dataset(
        tmp_path,
        "development",
        rows,
        source="fixture",
        revision="v1",
        source_split="synthetic",
        license_name="CC0",
    )
    config = changed(
        fixture_config,
        attention_implementation=attention,
        development={
            "records": str(tmp_path / "development.jsonl"),
            "manifest": str(tmp_path / "development.manifest.json"),
        },
    )
    serial = evaluate(config)
    batched = evaluate(
        changed(config, evaluation=config.evaluation.model_dump() | {"batch_size": 2})
    )
    assert batched["samples"] == serial["samples"]
    assert batched["protocol"]["attention_implementation"] == attention
    with pytest.raises(ValueError, match="different protocol"):
        compare(serial, batched | {"checkpoint_sha256": "a" * 64}, 42)


@pytest.mark.parametrize("shared_pad_eos", [False, True])
def test_batch_eos_padding_counts_and_order(fixture_config, monkeypatch, tmp_path, shared_pad_eos):
    import forge_training.evaluation as evaluation
    from forge_training.modeling import load_base

    model, tokenizer = load_base(fixture_config)
    if shared_pad_eos:
        tokenizer.pad_token = tokenizer.eos_token
    rows = [
        Example(id="short", question="1?", answer="#### 2"),
        Example(id="long", question="What is 2 plus 1?", answer="#### 3"),
        Example(id="tail", question="4?", answer="#### 4"),
    ]
    write_dataset(
        tmp_path,
        "development",
        rows,
        source="fixture",
        revision="v1",
        source_split="synthetic",
        license_name="CC0",
    )
    calls = []

    def generate(**kwargs):
        ids, masks = kwargs["input_ids"], kwargs["attention_mask"]
        assert (masks[:, -1] == 1).all()
        calls.append(len(ids))
        if len(ids) == 2:
            assert masks[0, 0] == 0 and masks[1, 0] == 1
            suffix = [
                [tokenizer.eos_token_id] + [tokenizer.pad_token_id] * 7,
                [tokenizer.convert_tokens_to_ids("3")] * 8,
            ]
        else:
            suffix = [[tokenizer.eos_token_id]]
        return torch.cat((ids, torch.tensor(suffix)), dim=1)

    monkeypatch.setattr(model, "generate", generate)
    monkeypatch.setattr(evaluation, "load_base", lambda config: (model, tokenizer))
    config = changed(
        fixture_config,
        development={
            "records": str(tmp_path / "development.jsonl"),
            "manifest": str(tmp_path / "development.manifest.json"),
        },
        evaluation=fixture_config.evaluation.model_dump() | {"batch_size": 2},
    )
    report = evaluate(config)
    assert calls == [2, 1]
    assert [s["id"] for s in report["samples"]] == [r.id for r in rows]
    assert [s["generated_tokens"] for s in report["samples"]] == [1, 8, 1]
    assert [s["hit_generation_limit"] for s in report["samples"]] == [False, True, False]


def test_batch_deadline_never_emits_a_partial_score(fixture_config, monkeypatch):
    import forge_training.evaluation as evaluation
    from forge_training.modeling import load_base

    model, tokenizer = load_base(fixture_config)
    elapsed = 0
    monkeypatch.setattr(evaluation.time, "monotonic", lambda: elapsed)
    monkeypatch.setattr(evaluation, "load_base", lambda config: (model, tokenizer))

    def generate(**kwargs):
        nonlocal elapsed
        elapsed = fixture_config.evaluation.max_seconds + 1
        return kwargs["input_ids"]

    monkeypatch.setattr(model, "generate", generate)
    with pytest.raises(TimeoutError, match="deadline expired"):
        evaluate(fixture_config)


@pytest.mark.parametrize(
    "change",
    [
        {"evaluation": {"batch_size": 0}},
        {"evaluation": {"batch_size": 65}},
        {"evaluation": {"batch_size": True}},
        {"attention_implementation": "flash_attention_2"},
    ],
)
def test_performance_options_are_bounded(fixture_config, change):
    with pytest.raises(ValueError):
        changed(fixture_config, **change)


@pytest.mark.parametrize("limit", ["prompt", "context"])
def test_all_prompts_are_checked_before_generation(fixture_config, monkeypatch, limit):
    import forge_training.evaluation as evaluation
    from forge_training.modeling import load_base

    model, tokenizer = load_base(fixture_config)
    config = fixture_config
    if limit == "prompt":
        config = changed(
            config, evaluation=config.evaluation.model_dump() | {"max_prompt_tokens": 16}
        )
    else:
        model.config.max_position_embeddings = 16
    monkeypatch.setattr(evaluation, "load_base", lambda config: (model, tokenizer))

    def must_not_generate(**kwargs):
        pytest.fail("Generation started before checking the prompt/context bound")

    monkeypatch.setattr(model, "generate", must_not_generate)
    with pytest.raises(ValueError, match="prompt exceeds|context window"):
        evaluate(config)


def test_mps_allocator_limit_uses_recommended_capacity(fixture_config, monkeypatch):
    from forge_training.modeling import configure_device_memory

    fractions = []
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 8 * 1024**3)
    monkeypatch.setattr(torch.mps, "set_per_process_memory_fraction", fractions.append)
    config = changed(fixture_config, device="mps", device_memory_limit_mb=4096)
    configure_device_memory(config)
    assert fractions == [0.5]
    with pytest.raises(ValueError, match="exceeds"):
        configure_device_memory(changed(config, device_memory_limit_mb=16384))
    assert fractions == [0.5]


def test_cpu_cannot_claim_a_gpu_memory_limit(fixture_config):
    from forge_training.modeling import configure_device_memory

    with pytest.raises(ValueError, match="MPS or CUDA"):
        configure_device_memory(changed(fixture_config, device_memory_limit_mb=4096))


@pytest.mark.parametrize("supported", [False, True])
def test_mps_bfloat16_checks_the_os_version(fixture_config, monkeypatch, supported):
    from forge_training.modeling import validate_device

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_macos_or_newer", lambda major, minor: supported)
    config = changed(fixture_config, device="mps", precision="bfloat16")
    if supported:
        validate_device(config)
    else:
        with pytest.raises(ValueError, match="macOS 14"):
            validate_device(config)
