import json
import subprocess

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from services import api
from services.core import contracts, store
from services.core.model_catalog import CATALOG
from services.core.model_inspect import Unsupported, lora_targets

LOCAL = "http://127.0.0.1:8000"
SPEC = {
    "id": "someone/small-7m",
    "revision": "b" * 40,
    "label": "small 7m",
    "family": "someone",
    "architecture": "llama",
    "parameters": "7M",
    "parameter_count": 7_000_000,
    "dtype": "bfloat16",
    "lora_targets": ["q_proj", "v_proj"],
    "chat_template_kwargs": {},
    "max_new_tokens": 256,
    "chat_template": True,
    "license": "mit",
    "gated": False,
}


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setattr(contracts, "REGISTERED", tmp_path / "models.json")
    monkeypatch.setattr(store, "DATA", tmp_path)
    monkeypatch.setenv("FORGE_BACKEND", "local")
    return tmp_path / "models.json"


def linear(*paths):
    return [(path, "Linear") for path in paths]


@pytest.mark.parametrize(
    "modules, expected",
    [
        (
            linear(
                "model.layers.0.self_attn.q_proj",
                "model.layers.0.self_attn.v_proj",
                "model.layers.0.self_attn.k_proj",
                "model.layers.0.self_attn.o_proj",
                "model.layers.0.mlp.up_proj",
            ),
            ["q_proj", "v_proj"],
        ),
        (
            linear("h.0.self_attention.query_key_value", "h.0.self_attention.dense", "h.0.mlp.dense_h_to_4h"),
            ["query_key_value"],
        ),
        (
            [
                ("transformer.h.0.attn.c_attn", "Conv1D"),
                ("transformer.h.0.attn.c_proj", "Conv1D"),
                ("transformer.h.0.mlp.c_fc", "Conv1D"),
            ],
            ["c_attn"],
        ),
        (linear("layers.0.attention.wq", "layers.0.attention.wv", "layers.0.attention.wo"), ["wq", "wv"]),
        (linear("layers.0.attn.mixer_a", "layers.0.attn.out_proj"), ["mixer_a"]),
        (linear("vision_tower.layers.0.attn.q_proj", "text.layers.0.attn.query"), ["query"]),
    ],
)
def test_lora_targets_read_each_architecture_naming(modules, expected):
    assert lora_targets(modules) == expected


def test_lora_targets_refuse_architectures_without_attention():
    with pytest.raises(Unsupported):
        lora_targets(linear("encoder.fc1", "lm_head"))


def test_every_catalog_entry_is_pinned_and_trainable():
    assert len(CATALOG) >= 40
    assert len({spec["family"] for spec in CATALOG.values()}) >= 12
    for model_id, spec in CATALOG.items():
        assert len(spec["revision"]) == 40, model_id
        assert spec["lora_targets"], model_id
        assert spec["parameter_count"] <= contracts.MODEL_PARAMETER_LIMIT, model_id
        assert contracts.CampaignConfig(model_id=model_id).model_id == model_id


def test_half_precision_is_used_only_where_the_weights_ship_in_it():
    assert CATALOG["bigscience/bloomz-560m"]["dtype"] == "float32"  # Stored in float16
    assert CATALOG["distilbert/distilgpt2"]["dtype"] == "float32"
    assert CATALOG["HuggingFaceTB/SmolLM2-135M-Instruct"]["dtype"] == "bfloat16"


def test_the_hand_written_models_keep_their_exact_settings():
    assert contracts.model_spec(None)["revision"] == contracts.MODEL_REVISION
    assert contracts.model_spec(None)["dtype"] == "float16"
    assert contracts.model_spec("Qwen/Qwen3.5-4B")["remote_only"] is True


def test_an_added_model_is_pinned_once_and_campaigns_accept_it(registry):
    with pytest.raises(ValidationError):
        contracts.CampaignConfig(model_id=SPEC["id"])
    assert contracts.register_model(SPEC)["revision"] == "b" * 40
    assert contracts.register_model(SPEC | {"revision": "c" * 40})["revision"] == "b" * 40
    assert contracts.CampaignConfig(model_id=SPEC["id"]).model_id == SPEC["id"]
    assert contracts.model_spec(SPEC["id"])["remote_only"] is False
    assert contracts.register_model(SPEC | {"id": "someone/huge", "parameter_count": 9_000_000_000})[
        "remote_only"
    ]


def test_the_api_adds_a_model_by_url_and_reports_refusals(registry, monkeypatch):
    calls = []

    def run(command, **_):
        calls.append(command[-1])
        answer = {"error": "This model has no safetensors weights"} if "pickle" in command[-1] else SPEC
        return subprocess.CompletedProcess(command, 0, json.dumps(answer) + "\n", "")

    monkeypatch.setattr(api.subprocess, "run", run)
    with TestClient(api.app, base_url=LOCAL) as c:
        added = c.post("/models", json={"repo": "https://huggingface.co/someone/small-7m"})
        assert added.status_code == 201 and added.json()["custom"] is True
        assert calls == ["someone/small-7m"]
        assert any(m["id"] == SPEC["id"] for m in c.get("/models").json())
        refused = c.post("/models", json={"repo": "someone/pickle"})
        assert refused.status_code == 422 and "safetensors" in refused.json()["detail"]
        for bad in ("../../etc/passwd", "no-owner", "a/b; rm -rf"):
            assert c.post("/models", json={"repo": bad}).status_code == 422
        assert len(calls) == 2


def test_model_problems_explain_size_and_gating(registry, monkeypatch):
    small = contracts.register_model(SPEC)
    big = contracts.model_spec("Qwen/Qwen3.5-4B")
    gated = small | {"gated": True}
    assert api.model_problem(small, "local") is None
    assert "cloud GPU" in api.model_problem(big, "local")
    assert api.model_problem(big, "modal") is None
    monkeypatch.setattr(api, "get_token", lambda: None)
    assert "HF_TOKEN" in api.model_problem(gated, "local")
    assert "local backend" in api.model_problem(gated, "modal")
    monkeypatch.setattr(api, "get_token", lambda: "token")
    assert api.model_problem(gated, "local") is None


def test_remote_jobs_carry_the_spec_for_a_container_without_the_registry(registry):
    from services.core import compute

    contracts.register_model(SPEC)

    class Client:
        jobs = []

        def call(self, job, _cancelled):
            self.jobs.append(job)
            raise InterruptedError
            yield

    backend = compute.Remote(Client())
    model, tokenizer = backend.load(model_id=SPEC["id"])
    row = {"id": "a", "question": "1 + 1?", "answer": "2"}
    with pytest.raises(InterruptedError):
        backend.evaluate(model, tokenizer, [row], lambda *_: None, lambda: False)
    carried = Client.jobs[0]["model"]
    assert carried["id"] == SPEC["id"] and carried["lora_targets"] == ["q_proj", "v_proj"]
    registry.unlink()
    contracts.adopt_model(carried)
    assert contracts.model_spec(SPEC["id"])["revision"] == "b" * 40
    contracts._ADOPTED.clear()
