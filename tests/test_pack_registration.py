"""Uploaded task packs, chunked uploads and teacher datasets, end to end through the API."""

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from services.core import distill, packs, store

TERMS = "Our own support tickets; we may use them to train internal models."


def qa_csv(n=200, broken=3, duplicates=2):
    lines = ["ticket_id,question,answer"]
    for i in range(n):
        lines.append(f"t{i},How many items are in order {i}?,#### {i % 7}")
    for i in range(broken):
        lines.append(f"b{i},Broken row {i}?,no marker")
    for i in range(duplicates):
        lines.append(f"d{i},How many items are in order {i}?,#### {i % 7}")
    return "\n".join(lines) + "\n"


QA_DRAFT = {
    "name": "Order counts",
    "objective": "Answer order-count questions with a final #### line.",
    "kind": "qa",
    "format": "csv",
    "mapping": {"id": "ticket_id", "input_template": "{question}", "target_template": "{answer}"},
    "instruction": "Answer the question. End with a final line in the form #### <number>.",
    "checker": {"extract": {"kind": "final_marker", "grammar": "number"}, "compare": {"kind": "numeric"}},
    "license": "Proprietary",
    "data_terms": TERMS,
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    from services import api

    monkeypatch.setattr(store, "DATA", tmp_path)
    monkeypatch.setattr(store, "campaigns", lambda: [])
    return TestClient(api.app, base_url="http://127.0.0.1:8000")


def test_preview_then_register_a_qa_pack(client):
    preview = client.post("/packs/preview", json={"draft": QA_DRAFT, "text": qa_csv()})
    assert preview.status_code == 200, preview.text
    report = preview.json()
    assert report["valid"] == 200 and report["invalid"] == 3 and report["duplicates"] == 2
    assert sum(report["split"].values()) == 200 and report["problems"] == []
    assert "salt" not in report
    assert not list((store.DATA / "packs").glob("*")) if (store.DATA / "packs").exists() else True

    registered = client.post("/packs", json={"draft": QA_DRAFT, "text": qa_csv()})
    assert registered.status_code == 201, registered.text
    pack_id = registered.json()["id"]
    assert pack_id.startswith("order-counts-")
    assert [p["id"] for p in client.get("/packs").json()] == ["gsm8k", pack_id]
    detail = client.get(f"/packs/{pack_id}").json()
    assert detail["capabilities"] == {
        "terse_world": False,
        "grpo": False,
        "dpo": False,
        "lm_eval_final": False,
        "remote": True,
    }
    # Samples come from training rows only; development and sealed questions are never shown.
    held = {r["question"] for r in packs.read(pack_id, "sealed") + packs.read(pack_id, "dev")}
    assert not {s["question"] for s in detail["samples"]} & held
    assert detail["counts"]["sealed_remaining"] == detail["counts"]["sealed"]
    # Registering the same content again returns the same immutable pack.
    assert client.post("/packs", json={"draft": QA_DRAFT, "text": qa_csv()}).json()["id"] == pack_id


def test_too_little_data_is_refused_with_reasons(client):
    response = client.post("/packs", json={"draft": QA_DRAFT, "text": qa_csv(n=20)})
    assert response.status_code == 422 and "sealed rows" in response.json()["detail"]


def test_registered_pack_splits_feed_campaigns_and_never_reuse_sealed_rows(client, monkeypatch):
    pack_id = client.post("/packs", json={"draft": QA_DRAFT, "text": qa_csv()}).json()["id"]
    pack = packs.get(pack_id)
    config = {"seed": 42, "dev_samples": 8, "final_samples": 4}
    train, dev, sealed = packs.data(pack, config)
    assert len(dev) == 8 and len(train) == len(packs.read(pack_id, "train"))
    first = sealed()
    monkeypatch.setattr(
        store,
        "campaigns",
        lambda: [{"config": {"task_pack": pack_id}, "sealed_ids": [r["id"] for r in first]}],
    )
    assert not {r["id"] for r in sealed()} & {r["id"] for r in first}
    checker = pack.task.make_checker()
    assert checker.score("Seven minus one\n#### 6", "#### 6")["correct"]


def test_extraction_and_preference_packs(client):
    rows = [
        json.dumps({"text": f"Refund {i} dollars please", "label": {"intent": "refund", "amount": i}})
        for i in range(200)
    ]
    extraction = QA_DRAFT | {
        "name": "Ticket intents",
        "kind": "extraction",
        "format": "jsonl",
        "mapping": {"input_template": "Ticket: {text}", "target_template": "{label}"},
        "instruction": "Return JSON with intent and amount.",
        "checker": {
            "extract": {"kind": "json"},
            "compare": {
                "kind": "json_fields",
                "json_schema": {"type": "object", "required": ["intent", "amount"]},
                "fields": [{"path": "/intent"}, {"path": "/amount", "compare": "numeric"}],
            },
            "reference": "whole",
        },
    }
    response = client.post("/packs", json={"draft": extraction, "text": "\n".join(rows)})
    assert response.status_code == 201, response.text
    assert response.json()["kind"] == "extraction"

    pairs = [
        json.dumps({"prompt": f"Greet customer {i}", "good": f"Hello {i}, how can I help?", "bad": "What."})
        for i in range(200)
    ]
    preference = {k: v for k, v in QA_DRAFT.items() if k != "checker"} | {
        "name": "Polite replies",
        "kind": "preference",
        "format": "jsonl",
        "mapping": {
            "input_template": "{prompt}",
            "target_template": None,
            "chosen": "{good}",
            "rejected": "{bad}",
        },
        "instruction": "Reply to the customer.",
    }
    response = client.post("/packs", json={"draft": preference, "text": "\n".join(pairs)})
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["capabilities"]["dpo"] and not body["capabilities"]["remote"]
    assert packs.read(body["id"], "train")[0]["rejected"] == "What."


def test_chunked_upload_assembles_and_checks_hashes(client):
    text = qa_csv()
    data = text.encode()
    start = client.post(
        "/uploads", json={"name": "orders.csv", "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    )
    upload_id = start.json()["id"]
    half = len(text) // 2
    for index, chunk in enumerate([text[:half], text[half:]]):
        sha = hashlib.sha256(chunk.encode()).hexdigest()
        assert (
            client.post(
                f"/uploads/{upload_id}/parts", json={"index": index, "text": chunk, "sha256": sha}
            ).status_code
            == 200
        )
    bad = client.post(f"/uploads/{upload_id}/parts", json={"index": 2, "text": "x", "sha256": "0" * 64})
    assert bad.status_code == 422
    assert client.post(f"/uploads/{upload_id}/complete", json={"parts": 2}).status_code == 200
    preview = client.post("/packs/preview", json={"draft": QA_DRAFT, "upload_id": upload_id})
    assert preview.json()["valid"] == 200
    assert client.post("/packs/preview", json={"draft": QA_DRAFT}).status_code == 422
    assert client.post("/uploads/../x/complete", json={"parts": 1}).status_code == 404


def test_campaign_launch_checks_methods_against_the_task(client):
    pack_id = client.post("/packs", json={"draft": QA_DRAFT, "text": qa_csv()}).json()["id"]
    refused = client.post("/campaigns", json={"task_pack": pack_id, "strategies": ["dpo"]})
    assert refused.status_code == 422 and "does not apply" in refused.json()["detail"]
    refused = client.post("/campaigns", json={"task_pack": pack_id, "strategies": ["distill_sft"]})
    assert refused.status_code == 422 and "teacher dataset" in refused.json()["detail"]
    refused = client.post(
        "/campaigns",
        json={"task_pack": pack_id, "strategies": ["distill_sft"], "synthetic_dataset": "syn-00000000"},
    )
    assert refused.status_code == 404
    refused = client.post("/campaigns", json={"task_pack": pack_id, "strategies": ["grpo"]})
    assert refused.status_code == 422
    labels = {s["id"]: s["label"] for s in client.get("/strategies").json()}
    assert labels["distill_sft"] == "Teacher distillation" and labels["dpo"] == "DPO alignment"


class Teacher:
    def __init__(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, body):
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._send({"data": [{"id": "tiny"}]} if self.path == "/v1/models" else {"models": []})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                question = body["messages"][-1]["content"].split("\n\n", 1)[-1]
                number = int(question.split("order ")[1].rstrip("?")) % 7 if "order " in question else 0
                self._send(
                    {
                        "choices": [
                            {"message": {"content": f"Counted.\n#### {number}"}, "finish_reason": "stop"}
                        ],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 5},
                    }
                )

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


def test_teacher_dataset_is_verified_and_filtered_for_the_campaign(client, tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_MODEL_LOCK", str(tmp_path / "model.lock"))
    monkeypatch.delenv("FORGE_MODEL_LOCK_HELD", raising=False)
    pack_id = client.post("/packs", json={"draft": QA_DRAFT, "text": qa_csv()}).json()["id"]
    teacher = Teacher()
    spec = {
        "kind": "vllm",
        "base_url": teacher.url,
        "model": "tiny",
        "teacher_terms": TERMS,
        "max_retries": 0,
    }
    probe = client.post("/distill/probe", json={"teacher": spec})
    assert probe.status_code == 200 and probe.json()["model_listed"]
    started = client.post(
        f"/packs/{pack_id}/teacher-data",
        json={"teacher": spec, "sampling": {"n": 1}, "budget": {"max_prompts": 10}},
    )
    assert started.status_code == 201, started.text
    syn_id = started.json()["id"]
    import time

    for _ in range(200):
        job = client.get(f"/packs/{pack_id}/teacher-data/{syn_id}").json()
        if job["status"] != "running":
            break
        time.sleep(0.1)
    teacher.server.shutdown()
    assert job["status"] == "completed", job
    assert job["counts"]["kept"] == 10
    pack = packs.get(pack_id)
    train, dev, _ = packs.data(pack, {"seed": 42, "dev_samples": 12, "final_samples": 4})
    from forge_training.data import question_key

    protected = {question_key(train[0]["question"])}
    rows, evidence = distill.campaign_rows(pack, syn_id, train, protected)
    assert (
        evidence["provenance_verified"]
        and evidence["checker_matches_pack"]
        and evidence["teacher_terms_present"]
    )
    assert evidence["verification"] == "reference" and evidence["separation_verified"]
    assert len(rows) + evidence["overlap_dropped"] == 10
    assert all(r["id"].startswith("syn-") and r["answer"].startswith("Counted.") for r in rows)
    # The synthetic set never touches development or sealed questions.
    dev_keys = {question_key(r["question"]) for r in dev}
    assert not {question_key(r["question"]) for r in rows} & dev_keys
