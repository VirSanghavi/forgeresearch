"""forge-distill against a fake OpenAI-compatible teacher: no model, no network beyond loopback."""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from forge_training.config import DatasetSpec, Example
from forge_training.data import write_dataset
from forge_training.distill import DistillConfig, key_hash, run, verify
from forge_training.tasks import GSM8K_TASK


class FakeTeacher:
    """Replies from a script keyed by question; counts requests and records headers."""

    def __init__(self, script):
        self.script, self.requests, self.headers = script, [], []
        teacher = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, body):
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/v1/models":
                    self._send({"data": [{"id": "teacher-4b"}]})
                elif self.path == "/api/tags":
                    self._send({"models": [{"name": "teacher-4b", "digest": "d" * 12}]})
                elif self.path == "/api/ps":
                    self._send({"models": []})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/api/generate":
                    return self._send({"done": True})
                teacher.headers.append(dict(self.headers))
                teacher.requests.append(body)
                question = body["messages"][-1]["content"].split("\n\n", 1)[1]
                replies = teacher.script[question]
                k = body["seed"]
                text, finish = replies[min(k, len(replies) - 1)]
                self._send(
                    {
                        "choices": [{"message": {"content": text}, "finish_reason": finish}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
                        "system_fingerprint": "fp-1",
                    }
                )

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


QUESTIONS = {
    "What is 2 plus 2?": [("I guess 5\n#### 5", "stop"), ("2 + 2 = 4\n#### 4", "stop")],
    "What is 3 times 3?": [
        ("<think>hmm</think>3 * 3 = 9\n#### 9", "stop"),
        ("Nine\n#### 9", "stop"),
    ],
    "What is 10 minus 7?": [("10 - 7 = ", "length"), ("no final line", "stop")],
}
ANSWERS = {
    "What is 2 plus 2?": "#### 4",
    "What is 3 times 3?": "#### 9",
    "What is 10 minus 7?": "#### 3",
}


@pytest.fixture
def teacher(monkeypatch, tmp_path):
    monkeypatch.setenv("FORGE_MODEL_LOCK", str(tmp_path / "model.lock"))
    monkeypatch.delenv("FORGE_MODEL_LOCK_HELD", raising=False)
    fake = FakeTeacher(QUESTIONS)
    yield fake
    fake.close()


def dataset(folder, name, questions, purpose="train"):
    spec = DatasetSpec(
        records=str(folder / f"{name}.jsonl"), manifest=str(folder / f"{name}.manifest.json")
    )
    if Path(spec.records).exists():
        return spec
    folder.mkdir(parents=True, exist_ok=True)
    rows = [
        Example(id=f"{name}-{i}", question=q, answer=ANSWERS.get(q, "#### 0"))
        for i, q in enumerate(questions)
    ]
    write_dataset(
        folder,
        name,
        rows,
        source="test",
        revision="r1",
        source_split="train",
        license_name="MIT",
        purpose=purpose,
    )
    return DatasetSpec(
        records=str(folder / f"{name}.jsonl"), manifest=str(folder / f"{name}.manifest.json")
    )


def config(tmp_path, teacher, **changes):
    base = {
        "teacher": {
            "kind": "ollama",
            "base_url": teacher.url,
            "model": "teacher-4b",
            "teacher_terms": "Apache-2.0 model run locally; outputs may train other models.",
            "max_retries": 0,
        },
        "task": GSM8K_TASK.model_dump(mode="json"),
        "prompts": dataset(tmp_path / "data", "train", list(QUESTIONS)).model_dump(),
        "sampling": {"n": 2, "seed": 0},
        "output_dir": str(tmp_path / "out"),
    }
    return DistillConfig.model_validate(base | changes)


def test_keeps_only_checker_verified_solutions_and_records_provenance(teacher, tmp_path):
    record = run(config(tmp_path, teacher))
    out = tmp_path / "out"
    kept = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
    assert [r["question"] for r in kept] == ["What is 2 plus 2?", "What is 3 times 3?"]
    assert kept[0]["answer"] == "2 + 2 = 4\n#### 4"
    # Reasoning is stripped, and the shorter of two correct solutions is kept.
    assert kept[1]["answer"] == "Nine\n#### 9"
    assert record["counts"]["reasons"] == {
        "correct": 3,
        "incorrect_answer": 1,
        "invalid_answer_format": 1,
        "truncated": 1,
    }
    assert record["reasoning_returned"] is True
    assert record["teacher"]["digest"] == "d" * 12 and record["teacher_unloaded"] is True
    assert record["status"] == "complete" and record["separation"]["verified"]
    manifest = json.loads((out / "train.manifest.json").read_text())
    assert manifest["source_split"] == "synthetic" and manifest["source"] == "distill:teacher-4b"
    assert verify(out)["status"] == "verified"


def test_refuses_before_any_request_when_a_prompt_is_excluded(teacher, tmp_path):
    dev = dataset(tmp_path / "dev", "development", ["What is 3 times 3?"], purpose="development")
    with pytest.raises(ValueError, match="excluded set"):
        run(config(tmp_path, teacher, exclude=[dev.model_dump()]))
    keys = tmp_path / "sealed.keys"
    keys.write_text(key_hash("what is 10 MINUS 7?") + "\n")
    with pytest.raises(ValueError, match="excluded set"):
        run(config(tmp_path, teacher, exclude_keys=[str(keys)]))
    assert teacher.requests == []


def test_budget_stop_then_resume_repeats_no_request(teacher, tmp_path):
    first = run(config(tmp_path, teacher, budget={"max_requests": 3}))
    assert first["status"] == "budget_exhausted" and len(teacher.requests) == 3
    second = run(config(tmp_path, teacher))
    assert second["status"] == "complete"
    assert second["usage"]["cached"] == 3 and len(teacher.requests) == 6
    assert verify(tmp_path / "out")["kept"] == 2
    with pytest.raises(ValueError, match="finished distillation"):
        run(config(tmp_path, teacher))


def test_api_key_is_sent_but_never_written(teacher, tmp_path, monkeypatch):
    monkeypatch.setenv("TEACHER_KEY", "sk-very-secret-value")
    cfg = config(tmp_path, teacher)
    cfg = DistillConfig.model_validate(
        cfg.model_dump() | {"teacher": cfg.teacher.model_dump() | {"api_key_env": "TEACHER_KEY"}}
    )
    run(cfg)
    assert teacher.headers[0]["Authorization"] == "Bearer sk-very-secret-value"
    for path in (tmp_path / "out").rglob("*"):
        if path.is_file():
            assert "sk-very-secret-value" not in path.read_text()


def test_verify_detects_tampering(teacher, tmp_path):
    run(config(tmp_path, teacher))
    out = tmp_path / "out"
    rows = (out / "train.jsonl").read_text().replace("#### 4", "#### 5")
    (out / "train.jsonl").write_text(rows)
    with pytest.raises(ValueError, match="train.jsonl"):
        verify(out)


def test_a_held_model_lock_refuses_a_local_teacher(teacher, tmp_path):
    from forge_training.locking import acquire_model_lock

    held = acquire_model_lock("training")
    try:
        # Simulate another process: this one did not inherit the holder's mark.
        os.environ.pop("FORGE_MODEL_LOCK_HELD", None)
        with pytest.raises(RuntimeError, match="Another model job"):
            run(config(tmp_path, teacher))
    finally:
        held.close()
    assert teacher.requests == []


def test_cli_round_trip(teacher, tmp_path):
    from forge_training.distill_cli import main

    cfg = config(tmp_path, teacher)
    path = tmp_path / "distill.json"
    path.write_text(json.dumps(cfg.model_dump(mode="json")))
    assert main(["run", str(path)]) == 0
    assert main(["verify", str(tmp_path / "out")]) == 0
    assert main(["verify", str(Path(tmp_path / "missing"))]) == 2


class ReasoningEndpoint:
    """Refuses sampling fields the way OpenAI reasoning models do; otherwise answers."""

    def __init__(
        self, refuse=r"Unsupported parameter: 'max_tokens' is not supported with this model."
    ):
        self.requests = []
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                endpoint.requests.append(body)
                if {"max_tokens", "temperature", "top_p"} & set(body):
                    data, code = json.dumps({"error": {"message": refuse}}).encode(), 400
                else:
                    reply = {
                        "choices": [{"message": {"content": "#### 4"}, "finish_reason": "stop"}]
                    }
                    data, code = json.dumps(reply).encode(), 200
                self.send_response(code)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


def test_reasoning_models_fall_back_to_their_default_sampling_once():
    from forge_training.distill import Sampling, Teacher, TeacherClient

    endpoint = ReasoningEndpoint()
    try:
        client = TeacherClient(
            Teacher(
                kind="openai_compatible",
                base_url=endpoint.url,
                model="o-model",
                max_retries=0,
                teacher_terms="Test endpoint; outputs may train other models.",
            )
        )
        first = client.complete(
            [{"role": "user", "content": "2+2"}], Sampling(n=1, max_tokens=64), 0
        )
        second = client.complete(
            [{"role": "user", "content": "2+2"}], Sampling(n=1, max_tokens=64), 1
        )
    finally:
        endpoint.server.shutdown()
    assert first["text"] == second["text"] == "#### 4"
    assert first["sampling"] == "model defaults (temperature, top_p refused)"
    # One refused request, then every request uses the model's defaults with the same token cap.
    assert len(endpoint.requests) == 3
    assert all(r["max_completion_tokens"] == 64 for r in endpoint.requests[1:])
    assert not ({"temperature", "top_p", "max_tokens"} & set(endpoint.requests[2]))


def test_other_bad_requests_still_fail():
    import pytest

    from forge_training.distill import Sampling, Teacher, TeacherClient, TeacherHTTPError

    endpoint = ReasoningEndpoint(refuse="Invalid model name")
    try:
        client = TeacherClient(
            Teacher(
                kind="openai_compatible",
                base_url=endpoint.url,
                model="o-model",
                max_retries=0,
                teacher_terms="Test endpoint; outputs may train other models.",
            )
        )
        with pytest.raises(TeacherHTTPError, match="HTTP 400"):
            client.complete([{"role": "user", "content": "2+2"}], Sampling(n=1), 0)
    finally:
        endpoint.server.shutdown()
    assert len(endpoint.requests) == 1
