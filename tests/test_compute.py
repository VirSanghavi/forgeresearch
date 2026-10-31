"""The backend switch, exercised with a fake remote GPU: no Modal calls and no GPU spend."""

import hashlib
import json
import queue
from pathlib import Path

import pytest

from services.core import compute, contracts, protocol

H100 = {
    "backend": "modal",
    "type": "cuda",
    "name": "NVIDIA H100 80GB HBM3",
    "torch": "2.14.1",
    "cuda": "13.0",
}
ROWS = [
    {"id": f"train-{i}", "question": f"What is {i} plus {i}?", "answer": f"#### {2 * i}"} for i in range(1, 4)
]


class FakeRemote:
    """Answers like the Modal function: stream events, then one result. Records every job."""

    def __init__(self, answers=None, files=None, sha=None, device=H100):
        self.jobs, self.answers, self.files, self.sha, self.device = [], answers or {}, files, sha, device
        self.closed = self.abandoned = False

    def call(self, job, cancelled):
        self.jobs.append(job)
        finished = False
        try:
            yield ("device", self.device)
            if job["kind"] == "evaluate":
                for row in job["rows"]:
                    if cancelled():
                        raise InterruptedError("Campaign stopped")
                    answer = self.answers.get(row["id"], protocol.extract(row["answer"]))
                    # Like the Modal function: raw output only. Scoring happens on this machine.
                    yield (
                        "prediction",
                        {
                            "id": row["id"],
                            "question": row["question"],
                            "output": f"#### {answer}",
                            "finish_reason": "stop",
                        },
                    )
                finished = True
                yield ("result", {"device": self.device, "seconds": 1.5, "count": len(job["rows"])})
            else:
                metrics = [{"step": s, "loss": 1.0 / s, "elapsed": s, "learning_rate": 1e-4} for s in (1, 2)]
                for metric in metrics:
                    if cancelled():
                        raise InterruptedError("Campaign stopped")
                    yield ("metric", metric)
                result = {
                    "metrics": metrics,
                    "before_hash": "a",
                    "after_hash": "b",
                    "weights_changed": True,
                    "sha256": self.sha or hashlib.sha256(self.files["adapter_model.safetensors"]).hexdigest(),
                    "seconds": 2.0,
                    "max_sequence_length": 768,
                    "device": self.device,
                }
                finished = True
                yield ("result", {"result": result, "files": self.files, "device": self.device})
        finally:
            self.abandoned = self.abandoned or not finished

    def close(self):
        self.closed = True


def checkpoint_files(device=H100, weights=b"lora-weights"):
    return {
        "adapter_model.safetensors": weights,
        "adapter_config.json": b'{"r": 8}',
        "training.json": json.dumps({"device": device, "metrics": []}).encode(),
    }


def never():
    return False


def test_backend_defaults_to_local_and_rejects_unknown_values(monkeypatch):
    monkeypatch.delenv("FORGE_BACKEND", raising=False)
    assert compute.configured() == "local"
    monkeypatch.setenv("FORGE_BACKEND", " Modal ")
    assert compute.configured() == "modal"
    assert compute.describe()["label"] == "Modal NVIDIA H100"
    monkeypatch.setenv("FORGE_BACKEND", "tpu")
    with pytest.raises(ValueError, match="FORGE_BACKEND must be one of"):
        compute.configured()


def test_remote_evaluation_streams_progress_and_keeps_worlds_local():
    remote = FakeRemote(answers={"train-2": "5"})
    seen, worlds, hardware = [], [], []
    backend = compute.Remote(remote, on_hardware=hardware.append)
    model, tokenizer = backend.load()

    def world(row, answer, i):
        worlds.append((row["id"], answer, i))
        return {"id": f"env-{i}", "reward": 1 if row["id"] != "train-3" else 0}

    result = backend.evaluate(
        model,
        tokenizer,
        ROWS,
        lambda done, total, p: seen.append((done, total, p["id"])),
        never,
        worlds=world,
    )
    assert seen == [(1, 3, "train-1"), (2, 3, "train-2"), (3, 3, "train-3")]
    assert worlds == [("train-1", "2", 0), ("train-2", "5", 1), ("train-3", "6", 2)]
    # Terse world rewards decide correctness, exactly as in local evaluation.
    assert [p["correct"] for p in result["predictions"]] == [True, True, False]
    assert [p["environment_id"] for p in result["predictions"]] == ["env-0", "env-1", "env-2"]
    assert result["correct"] == 2 and result["n"] == 3 and result["split"] == "development"
    assert result["device"] == H100 and result["gpu_seconds"] == 1.5
    assert result["decoding"] == {"do_sample": False, "max_new_tokens": 256, "batch_size": 1}
    assert result["model_revision"] == contracts.MODEL_REVISION
    assert result["dataset_revision"] == protocol.DATASET_REVISION
    assert all(p["finish_reason"] == "stop" for p in result["predictions"])
    # Planned hardware first, then the measured GPU exactly once.
    assert [h["name"] for h in hardware] == [None, "NVIDIA H100 80GB HBM3"]
    assert remote.jobs[0]["adapter"] is None
    assert set(remote.jobs[0]["rows"][0]) == {"id", "question", "answer"}
    # The prompt travels with the job; the checker stays local.
    assert remote.jobs[0]["task"]["task_id"] == "gsm8k"
    assert [p["answer"] for p in result["predictions"]] == ["2", "5", "6"]
    assert [p["expected"] for p in result["predictions"]] == ["2", "4", "6"]


def test_remote_evaluation_sends_the_saved_adapter_bytes(tmp_path):
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
    (tmp_path / "adapter_config.json").write_text("{}")
    remote = FakeRemote()
    backend = compute.Remote(remote)
    model, tokenizer = backend.load(tmp_path)
    backend.evaluate(model, tokenizer, ROWS[:1], lambda *_: None, never)
    adapter = remote.jobs[0]["adapter"]
    assert adapter["files"] == {"adapter_model.safetensors": b"weights", "adapter_config.json": b"{}"}
    assert adapter["sha256"] == hashlib.sha256(b"weights").hexdigest()


def test_remote_training_saves_the_same_checkpoint_layout(tmp_path):
    remote = FakeRemote(files=checkpoint_files())
    metrics = []
    backend = compute.Remote(remote)
    result = backend.train(
        {"strategy": "broad_sft"}, {"seed": 1}, ROWS, tmp_path / "exp-001", metrics.append, never
    )
    folder = tmp_path / "exp-001"
    assert [m["step"] for m in metrics] == [1, 2]
    assert sorted(p.name for p in folder.iterdir()) == [
        "adapter_config.json",
        "adapter_model.safetensors",
        "training.json",
    ]
    assert protocol.sha(folder / "adapter_model.safetensors") == result["sha256"]
    assert result["weights_changed"] and result["device"]["name"] == "NVIDIA H100 80GB HBM3"
    assert backend.hardware == H100


def test_remote_checkpoint_that_changed_in_transfer_is_rejected(tmp_path):
    remote = FakeRemote(files=checkpoint_files(), sha="0" * 64)
    with pytest.raises(RuntimeError, match="checksum changed in transfer"):
        compute.Remote(remote).train({}, {}, ROWS, tmp_path, lambda m: None, never)


def test_remote_file_names_cannot_escape_the_campaign_folder(tmp_path):
    files = checkpoint_files() | {"../escape.json": b"{}"}
    with pytest.raises(RuntimeError, match="Refusing remote file name"):
        compute.Remote(FakeRemote(files=files)).train({}, {}, ROWS, tmp_path / "exp", lambda m: None, never)
    assert not (tmp_path / "escape.json").exists()


def test_remote_training_record_must_name_the_device_that_ran_it(tmp_path):
    files = checkpoint_files(device=H100 | {"name": "Some other GPU"})
    with pytest.raises(RuntimeError, match="disagrees with the reported device"):
        compute.Remote(FakeRemote(files=files)).train({}, {}, ROWS, tmp_path, lambda m: None, never)


@pytest.mark.parametrize("kind", ["evaluate", "train"])
def test_stop_during_a_remote_job_interrupts_the_campaign(tmp_path, kind):
    remote = FakeRemote(files=checkpoint_files())
    backend = compute.Remote(remote)
    calls = {"n": 0}

    def stop_after_first():
        calls["n"] += 1
        return calls["n"] > 1

    with pytest.raises(InterruptedError):
        if kind == "evaluate":
            model, tokenizer = backend.load()
            backend.evaluate(model, tokenizer, ROWS, lambda *_: None, stop_after_first)
        else:
            backend.train({}, {}, ROWS, tmp_path, lambda m: None, stop_after_first)
    assert remote.abandoned
    assert not (tmp_path / "adapter_model.safetensors").exists()


def test_short_remote_stream_fails_instead_of_scoring_a_partial_evaluation():
    class Short(FakeRemote):
        def call(self, job, cancelled):
            yield from super().call(job | {"rows": job["rows"][:2]}, cancelled)

    with pytest.raises(RuntimeError, match="returned 2 of 3 predictions"):
        backend = compute.Remote(Short())
        model, tokenizer = backend.load()
        backend.evaluate(model, tokenizer, ROWS, lambda *_: None, never)


def test_out_of_order_remote_prediction_is_rejected():
    class Shuffled(FakeRemote):
        def call(self, job, cancelled):
            yield from super().call(job | {"rows": job["rows"][::-1]}, cancelled)

    with pytest.raises(ValueError, match="arrived for row"):
        backend = compute.Remote(Shuffled())
        model, tokenizer = backend.load()
        backend.evaluate(model, tokenizer, ROWS, lambda *_: None, never)


def test_unavailable_modal_fails_clearly_unless_fallback_is_configured(monkeypatch):
    pytest.importorskip("modal")
    from services.core import remote

    class Broken:
        def open(self):
            raise RuntimeError("no Modal credentials")

    monkeypatch.setattr(remote, "Session", Broken)
    monkeypatch.setenv("FORGE_BACKEND", "modal")
    monkeypatch.delenv("FORGE_BACKEND_FALLBACK", raising=False)
    with pytest.raises(RuntimeError, match="Modal H100 backend unavailable: no Modal credentials"):
        compute.open_backend()
    notices, hardware = [], []
    monkeypatch.setenv("FORGE_BACKEND_FALLBACK", "local")
    pytest.importorskip("torch")  # The fallback measures the local device.
    backend = compute.open_backend(on_hardware=hardware.append, on_notice=notices.append)
    assert backend.name == "local"
    assert "runs on the local device instead" in notices[0]
    assert hardware[0]["backend"] == "local"


def test_local_backend_is_the_default_and_never_touches_modal(monkeypatch):
    import sys

    pytest.importorskip("torch")
    monkeypatch.delenv("FORGE_BACKEND", raising=False)
    monkeypatch.delitem(sys.modules, "services.core.remote", raising=False)
    backend = compute.open_backend()
    assert backend.name == "local" and backend.hardware["backend"] == "local"
    assert "services.core.remote" not in sys.modules


def test_runner_routes_every_model_call_through_the_backend():
    source = Path(compute.__file__).with_name("runner.py").read_text()
    for name in ("load", "evaluate", "train", "unload"):
        assert f"training.{name}(" not in source, name
    assert "backend.close()" in source


class FakeCall:
    def __init__(self, ready_after, result):
        self.polls, self.ready_after, self.result, self.cancelled = 0, ready_after, result, None

    def get(self, timeout=None):
        self.polls += 1
        if self.polls < self.ready_after:
            raise TimeoutError()
        return self.result

    def cancel(self, terminate_containers=False):
        self.cancelled = terminate_containers


class FakeQueue:
    def __init__(self, items):
        self.items = list(items)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def get_many(self, n, block=True, timeout=None):
        batch, self.items = self.items[:1], self.items[1:]
        if not batch and block:
            raise queue.Empty()
        return batch


def session_with(monkeypatch, items, call):
    pytest.importorskip("modal")
    from services.core import remote

    events = FakeQueue(items)
    monkeypatch.setattr(remote.modal.Queue, "ephemeral", staticmethod(lambda: events))
    monkeypatch.setattr(remote.execute, "spawn", lambda job, q: call)
    session = remote.Session(poll_seconds=0)
    session._context = object()
    return session


def test_modal_session_delivers_every_event_before_the_result(monkeypatch):
    call = FakeCall(ready_after=2, result={"device": H100})
    session = session_with(monkeypatch, [("metric", 1), ("metric", 2), ("metric", 3)], call)
    assert list(session.call({"kind": "train"}, never)) == [
        ("metric", 1),
        ("metric", 2),
        ("metric", 3),
        ("result", {"device": H100}),
    ]
    assert call.cancelled is None


def test_modal_session_cancels_the_gpu_call_when_the_campaign_stops(monkeypatch):
    call = FakeCall(ready_after=99, result=None)
    session = session_with(monkeypatch, [("metric", 1)], call)
    with pytest.raises(InterruptedError):
        list(session.call({"kind": "train"}, lambda: True))
    assert call.cancelled is True


def test_modal_session_cancels_the_gpu_call_when_the_caller_abandons_it(monkeypatch):
    call = FakeCall(ready_after=99, result=None)
    session = session_with(monkeypatch, [("prediction", 1), ("prediction", 2)], call)
    stream = session.call({"kind": "evaluate"}, never)
    assert next(stream) == ("prediction", 1)
    stream.close()
    assert call.cancelled is True


def test_modal_session_refuses_calls_before_it_is_open():
    pytest.importorskip("modal")
    from services.core import remote

    with pytest.raises(RuntimeError, match="not open"):
        list(remote.Session().call({"kind": "evaluate"}, never))


def test_remote_gpu_is_named_before_the_first_prediction_arrives():
    names = []
    backend = compute.Remote(FakeRemote())
    model, tokenizer = backend.load()
    backend.evaluate(model, tokenizer, ROWS, lambda *_: names.append(backend.hardware["name"]), never)
    assert names == ["NVIDIA H100 80GB HBM3"] * 3


def test_default_model_is_unchanged_and_the_4b_is_described_by_its_own_spec():
    assert contracts.CampaignConfig().model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    small = contracts.model_spec(None)
    assert small["lora_targets"] == ["q_proj", "v_proj"] and small["max_new_tokens"] == 256
    assert small["dtype"] == "float16" and small["chat_template_kwargs"] == {}
    big = contracts.model_spec("Qwen/Qwen3.5-4B")
    assert big["revision"] == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    assert big["dtype"] == "bfloat16" and big["remote_only"]
    assert big["chat_template_kwargs"] == {"enable_thinking": False}
    assert "in_proj_qkv" in big["lora_targets"]  # 24 of its 32 layers are linear attention
    assert contracts.CampaignConfig(model_id="Qwen/Qwen3.5-4B").model_id == "Qwen/Qwen3.5-4B"
    with pytest.raises(ValueError):
        contracts.CampaignConfig(model_id="Qwen/Qwen3.5-9B")


def test_remote_jobs_carry_the_campaign_model_and_record_its_protocol():
    remote = FakeRemote()
    backend = compute.Remote(remote)
    model, tokenizer = backend.load(model_id="Qwen/Qwen3.5-4B")
    result = backend.evaluate(model, tokenizer, ROWS, lambda *_: None, never)
    assert remote.jobs[0]["model_id"] == "Qwen/Qwen3.5-4B"
    assert result["model_id"] == "Qwen/Qwen3.5-4B"
    assert result["model_revision"] == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    assert result["decoding"] == {
        "do_sample": False,
        "max_new_tokens": 512,
        "batch_size": 1,
        "enable_thinking": False,
    }
