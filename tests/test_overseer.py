"""The production overseer on fixtures: intake, free checks, budgeted judging, drift and hand-off."""

import json
import threading
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from services.core import production, store
from services.core.overseer import db, judge, review, traces, worker

TERMS = "Our own support traffic; we may use it to train internal models."
DRAFT = {
    "name": "Order counts",
    "objective": "Answer order-count questions with a final #### line.",
    "kind": "qa",
    "format": "csv",
    "mapping": {"input_template": "{question}", "target_template": "{answer}"},
    "instruction": "Answer the question. End with a final line in the form #### <number>.",
    "checker": {"extract": {"kind": "final_marker", "grammar": "number"}, "compare": {"kind": "numeric"}},
    "license": "Proprietary",
    "data_terms": TERMS,
}


@pytest.fixture
def lab(tmp_path, monkeypatch):
    from services import api

    monkeypatch.setattr(store, "DATA", tmp_path)
    monkeypatch.setattr(store, "campaigns", lambda: [])
    for name in ("FORGE_OVERSEER_JUDGE_URL", "FORGE_OVERSEER_JUDGE_MODEL", "FORGE_OVERSEER_SEND_TRACES"):
        monkeypatch.delenv(name, raising=False)
    traces._sealed_cache.clear()
    client = TestClient(api.app, base_url="http://127.0.0.1:8000")
    rows = "question,answer\n" + "".join(
        f"How many items are in order {i}?,#### {i % 7}\n" for i in range(200)
    )
    pack_id = client.post("/packs", json={"draft": DRAFT, "text": rows}).json()["id"]
    return client, pack_id


def trace(i, output=None, **fields):
    return {
        "id": f"t{i}",
        "prompt": f"Customer question {i}: how many boxes?",
        "output": output or f"Counted.\n#### {i}",
        **fields,
    }


def test_intake_is_idempotent_and_sealed_questions_are_kept_as_flags_only(lab):
    client, pack_id = lab
    from services.core import packs

    sealed = packs.read(pack_id, "sealed")[0]["question"]
    batch = [trace(1), trace(2), {"id": "s1", "prompt": sealed, "output": "#### 1"}, {"id": "bad"}]
    first = client.post(
        "/overseer/traces", json={"source": "web", "task_pack": pack_id, "traces": batch}
    ).json()
    assert first["accepted"] == 3 and len(first["rejected"]) == 1
    again = client.post(
        "/overseer/traces", json={"source": "web", "task_pack": pack_id, "traces": batch[:2]}
    ).json()
    assert again == {"accepted": 0, "duplicates": 2, "rejected": []}
    stored = db.rows("select payload, sealed_overlap from traces where trace_id = 's1'")[0]
    assert stored["sealed_overlap"] == 1 and sealed not in stored["payload"]
    with pytest.raises(ValueError):
        review.label("web:s1", "#### 1", "me")


def test_free_checks_flag_failures_without_any_model(lab):
    client, pack_id = lab
    batch = [
        trace(1),
        trace(2, output="I'm sorry, I can't help with that."),
        trace(3, output="no final line here"),
        trace(4, finish_reason="length"),
        trace(5, output=("buy more boxes now " * 30) + "\n#### 3"),
        trace(6, output="Ответ: коробок много, точно не знаю"),
        trace(7, feedback={"rating": -1, "correction": "It was 9"}),
        trace(8, latency_ms=9000.0),
    ]
    client.post("/overseer/traces", json={"source": "web", "task_pack": pack_id, "traces": batch})
    import os

    os.environ["FORGE_OVERSEER_LATENCY_SLO_MS"] = "2000"
    try:
        assert worker.check_new() == 8
    finally:
        os.environ.pop("FORGE_OVERSEER_LATENCY_SLO_MS")
    kinds = {}
    for s in db.rows("select key, kind from signals"):
        kinds.setdefault(s["key"], set()).add(s["kind"])
    assert "web:t1" not in kinds
    assert "refusal" in kinds["web:t2"] and "format_failure" in kinds["web:t2"]
    assert kinds["web:t3"] == {"format_failure"}
    assert "truncated" in kinds["web:t4"]
    assert "repetition" in kinds["web:t5"]
    assert "language_mismatch" in kinds["web:t6"]
    assert "user_reported" in kinds["web:t7"]
    assert "latency_outlier" in kinds["web:t8"]
    summary = client.get("/overseer/summary", params={"task_pack": pack_id}).json()
    assert summary["traces"] == 8 and summary["flagged"] == 7
    assert summary["judge"]["ready"] is False and summary["negative_feedback_rate"] == 1 / 8


class FakeJudge:
    def __init__(self, verdict="fail"):
        self.calls = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                outer.calls += 1
                self.rfile.read(int(self.headers["Content-Length"]))
                content = json.dumps(
                    {
                        "verdict": verdict,
                        "refusal": False,
                        "format_ok": False,
                        "failure_kinds": ["wrong_count"],
                        "confidence": 0.8,
                        "rationale": "Count is off.",
                    }
                )
                body = json.dumps(
                    {
                        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 400, "completion_tokens": 60},
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


def test_judge_runs_on_a_sample_within_a_hard_budget_and_caches(lab, monkeypatch):
    client, pack_id = lab
    fake = FakeJudge()
    monkeypatch.setenv("FORGE_OVERSEER_JUDGE_URL", fake.url)
    monkeypatch.setenv("FORGE_OVERSEER_JUDGE_MODEL", "judge-small")
    monkeypatch.setenv("FORGE_OVERSEER_SAMPLE_RATE", "1.0")
    monkeypatch.setenv("FORGE_OVERSEER_PRICE_IN_PER_M", "1000")
    monkeypatch.setenv("FORGE_OVERSEER_PRICE_OUT_PER_M", "1000")
    monkeypatch.setenv("FORGE_OVERSEER_DAILY_USD", "1.5")
    client.post(
        "/overseer/traces",
        json={"source": "web", "task_pack": pack_id, "traces": [trace(i) for i in range(10)]},
    )
    worker.check_new()
    result = worker.judge_eligible()
    # Each call reserves about $0.4 at these prices; the third would cross the $1.50 cap.
    assert result["judged"] == 3 and result["skipped"] == "Daily judge budget reached"
    assert fake.calls == 3 and judge.spend()["calls"] == 3 and judge.spend()["usd"] <= 1.5
    stored = json.loads(db.rows("select result from judgments limit 1")[0]["result"])
    assert stored["verdict"] == "fail" and stored["hallucination"] is None and "answer" not in stored
    # The same input is answered from cache without spending again.
    again = judge.judge(
        "web:t0-copy",
        db.payload(db.rows("select * from traces where key = 'web:t0'")[0]),
        __import__("services.core.packs", fromlist=["get"]).get(pack_id),
    )
    assert again["verdict"] == "fail" and fake.calls == 3
    summary = client.get("/overseer/summary", params={"task_pack": pack_id}).json()
    assert summary["judged"]["random_sample"] == 3 and summary["judged"]["sample_failure_rate"] == 1.0
    fake.server.shutdown()


def test_hosted_judges_need_an_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("FORGE_OVERSEER_JUDGE_URL", "https://judge.example.com/v1")
    monkeypatch.setenv("FORGE_OVERSEER_JUDGE_MODEL", "small")
    monkeypatch.delenv("FORGE_OVERSEER_SEND_TRACES", raising=False)
    ready, reason = judge.available()
    assert not ready and "FORGE_OVERSEER_SEND_TRACES" in reason
    monkeypatch.setenv("FORGE_OVERSEER_SEND_TRACES", "1")
    assert judge.available() == (True, None)


def test_sampling_is_deterministic():
    picks = [judge.sampled(f"web:t{i}", "day-a", 0.2) for i in range(1000)]
    assert picks == [judge.sampled(f"web:t{i}", "day-a", 0.2) for i in range(1000)]
    assert 150 < sum(picks) < 250


def test_drift_alerts_on_a_jump_against_the_trailing_week(lab):
    client, pack_id = lab
    now = datetime.now(UTC)
    old = (now - timedelta(days=3)).isoformat()
    traces.ingest([trace(i) for i in range(100)], "web", pack_id, received_at=old)
    traces.ingest(
        [trace(1000 + i, output="no final line") if i < 30 else trace(1000 + i) for i in range(100)],
        "web",
        pack_id,
        received_at=now.isoformat(),
    )
    worker.check_new()
    assert worker.alerts(now) >= 1
    kinds = {
        a["kind"] for a in client.get("/overseer/summary", params={"task_pack": pack_id}).json()["alerts"]
    }
    assert "format_failure" in kinds


def test_reviewer_labels_go_back_to_the_lab_as_a_production_scan(lab):
    client, pack_id = lab
    batch = [
        trace(1, output="no final line", feedback={"correction": "#### 99"}),
        trace(2),
        trace(3, output="#### 5"),
    ]
    client.post("/overseer/traces", json={"source": "web", "task_pack": pack_id, "traces": batch})
    worker.check_new()
    queue = client.get("/overseer/queue", params={"task_pack": pack_id}).json()
    assert queue[0]["key"] == "web:t1" and {s["kind"] for s in queue[0]["signals"]} >= {
        "format_failure",
        "user_reported",
    }
    assert client.post("/overseer/export", json={"task_pack": pack_id, "name": "none yet"}).status_code == 422
    assert (
        client.post("/overseer/labels", json={"key": "web:t1", "answer": "not a number"}).status_code == 422
    )
    assert client.post("/overseer/labels", json={"key": "web:t1", "answer": "#### 4"}).status_code == 200
    client.post("/overseer/labels", json={"key": "web:t3", "answer": "#### 6"})
    exported = client.post("/overseer/export", json={"task_pack": pack_id, "name": "Overseer week"})
    assert exported.status_code == 201, exported.text
    scan = exported.json()
    assert scan["source"] == "overseer" and scan["task_pack"] == pack_id and scan["traces"] == 3
    assert scan["audit"]["labeled"] == 2 and scan["audit"]["actually_wrong"] == 2
    report = production.load_scan(scan["id"])
    train, held = production.campaign_rows(
        report,
        production.load_labels(scan["id"]),
        __import__("services.core.packs", fromlist=["get"]).get(pack_id),
    )
    # The reviewer's label is the answer, never the user's correction. The wrong answer that no
    # detector flagged (t3) is counted as missed by the audit, not silently trained on.
    assert {r["answer"] for r in train + held} == {"#### 4"}
    assert scan["audit"]["missed"] == 1
    refused = client.post("/campaigns", json={"production_scan": scan["id"]})
    assert refused.status_code == 422 and "different task" in refused.json()["detail"]


def test_supabase_pull_marks_rows_pulled(lab):
    client, pack_id = lab

    class Remote:
        def __init__(self):
            self.rows = [
                {
                    "id": n,
                    "source": "app",
                    "pack_id": pack_id,
                    "payload": trace(n),
                    "received_at": datetime.now(UTC).isoformat(),
                }
                for n in range(3)
            ]
            self.patched = []

        def get(self, path, params):
            class R:
                def __init__(self, body):
                    self.body = body

                def raise_for_status(self):
                    pass

                def json(self):
                    return self.body

            return R([r for r in self.rows if r["id"] not in self.patched])

        def patch(self, path, params, json):
            self.patched += [int(x) for x in params["id"][4:-1].split(",")]
            return type("R", (), {"raise_for_status": lambda self: None})()

    from services.core.overseer import pull

    remote = Remote()
    assert pull.pull(remote) == {"pulled": 3, "rejected": 0}
    assert remote.patched == [0, 1, 2]
    assert db.rows("select count(*) as n from traces where source = 'app'")[0]["n"] == 3
