"""Autonomous traffic connections: the real agent process, against real HTTP servers.

Only the SuperServe SDK is a double, so these establish the contract with it, not live account access.
"""

import json
import os
import shutil
import stat
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient

from services.core import store
from services.core.overseer import agent, connections, db, pull, traces, vms

SECRET = "real-source-credential-do-not-leak"
TOKEN = "fo_real-ingest-token-do-not-leak"
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


def wait_for(check, seconds=20):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        found = check()
        if found:
            return found
        time.sleep(0.05)
    raise AssertionError("Timed out waiting for the agent")


class Server:
    """A real HTTP server whose reply is a function of the request."""

    def __init__(self, respond):
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self):
                length = int(self.headers.get("content-length") or 0)
                body = self.rfile.read(length) if length else b""
                outer.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                status, reply, headers = respond(self.path, body)
                self.send_response(status)
                for key, value in headers.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(reply)))
                self.end_headers()
                self.wfile.write(reply)

            do_GET = do_POST = _answer

            def log_message(self, *_):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def servers():
    made = []

    def make(respond):
        made.append(Server(respond))
        return made[-1]

    yield make
    for server in made:
        server.close()


def rows(n, start=1):
    return [
        {
            "id": i,
            "created_at": f"2030-01-01T00:00:{i:02d}+00:00",
            "request": {"model": "m1", "messages": [{"role": "user", "content": f"How many boxes in {i}?"}]},
            "response": {
                "choices": [
                    {"message": {"role": "assistant", "content": f"#### {i}"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        }
        for i in range(start, start + n)
    ]


def cursor_source(all_rows, page=2):
    def respond(path, _):
        since = parse_qs(urlsplit(path).query).get("since", ["0"])[0]
        newer = [r for r in all_rows if r["id"] > int(since)][:page]
        return 200, json.dumps({"data": newer}).encode(), {}

    return respond


def collecting_sink(store_to, status=lambda: 202):
    def respond(_, body):
        batch = json.loads(body)["traces"]
        current = status()
        if current != 202:
            return current, b"{}", {}
        fresh = [t for t in batch if t["id"] not in {s["id"] for s in store_to}]
        store_to.extend(fresh)
        reply = {"accepted": len(fresh), "duplicates": len(batch) - len(fresh), "rejected": []}
        return 202, json.dumps(reply).encode(), {}

    return respond


def run_agent(tmp_path, config, env=None):
    """Runs a copy of the file alone, in isolated mode: proof it needs nothing from this repository."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(agent.__file__, tmp_path / "agent.py")
    (tmp_path / "config.json").write_text(json.dumps(config))
    process = subprocess.Popen(
        [sys.executable, "-I", "agent.py", str(tmp_path)],
        cwd=tmp_path,
        env={"PATH": os.environ["PATH"], **(env or {})},
    )
    return process


def stop_agent(tmp_path, process):
    (tmp_path / "stop").touch()
    process.wait(timeout=10)


def state_of(tmp_path):
    try:
        return json.loads((tmp_path / "state.json").read_text())
    except (OSError, ValueError):
        return {}


def config_for(source, sink, **fields):
    return {
        "source": "conn-test",
        "task_pack": "gsm8k",
        "url": f"{source.url}/logs",
        "sink_url": f"{sink.url}/ingest",
        "cursor_param": "since",
        "start_cursor": 0,
        "poll_seconds": 1,
        **fields,
    }


# Working out the shape of a customer's records.


def test_discovers_openai_shaped_logs_with_no_configuration():
    records = rows(3)
    mapping = agent.discover(records)
    trace = agent.to_trace(records[0], mapping)
    assert trace["id"] == "1"
    assert trace["prompt"] == "How many boxes in 1?"
    assert trace["output"] == "#### 1"
    assert trace["finish_reason"] == "stop"
    assert trace["model_version"] == "m1"
    assert trace["tokens"] == {"prompt": 5, "completion": 2}
    assert trace["timestamp"].startswith("2030")


def test_reads_tracer_messages_held_as_json_text():
    record = {
        "id": "g1",
        "input": json.dumps(
            [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": [{"type": "text", "text": "second"}]},
            ]
        ),
        "output": json.dumps({"role": "assistant", "content": "done"}),
    }
    trace = agent.to_trace(record, agent.discover([record]))
    assert (trace["prompt"], trace["output"]) == ("second", "done")


def test_an_extraction_answer_that_is_json_stays_exactly_as_logged():
    answer = '{"name": "Ada", "text": "not a message", "total": 3}'
    record = {"id": "x", "prompt": "Extract the order", "output": answer}
    assert agent.to_trace(record, agent.discover([record]))["output"] == answer


def test_unknown_shape_asks_for_a_mapping_and_reports_names_not_values():
    records = [{"id": 1, "payload": {"customer_text": "PRIVATE VALUE", "bot_text": "PRIVATE ANSWER"}}]
    assert agent.discover(records) is None
    names = agent.field_names(records)
    assert "payload.customer_text" in names
    assert "PRIVATE" not in json.dumps(names)
    mapping = agent.discover(records, {"prompt": "payload.customer_text", "output": "payload.bot_text"})
    assert agent.to_trace(records[0], mapping)["prompt"] == "PRIVATE VALUE"


def test_records_are_found_in_the_usual_places_only():
    assert agent.find_records([{"a": 1}]) == [{"a": 1}]
    assert agent.find_records({"results": [{"a": 1}]}) == [{"a": 1}]
    assert agent.find_records({"deep": {"down": [{"a": 1}]}}, "deep.down") == [{"a": 1}]
    assert agent.find_records({"nothing": 1}) == []


def test_a_uuid_is_never_chosen_as_the_cursor(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"url": "https://x.test/?since={cursor}"}))
    worker = agent.Agent(str(tmp_path))
    assert worker.cursor_field([{"id": "7f3a-uuid"}]) is None
    assert worker.cursor_field([{"id": 7}]) == "id"
    assert worker.cursor_field([{"id": "7f3a-uuid", "created_at": "2030-01-01"}]) == "created_at"


def test_time_cursors_read_slightly_behind_and_others_are_untouched():
    assert agent.rewind("2030-01-01T00:10:00Z", 300).startswith("2030-01-01T00:05:00")
    assert agent.rewind(42, 300) == 42
    assert agent.rewind("not-a-time", 300) == "not-a-time"


def test_metadata_is_trimmed_to_what_the_ingest_function_accepts():
    record = {
        "prompt": "p",
        "output": "o",
        "metadata": {f"k{i}": i for i in range(30)} | {"nested": {"a": 1}},
    }
    trace = agent.to_trace(record, agent.discover([record]))
    assert len(trace["metadata"]) == 16
    assert "nested" not in trace["metadata"]


# The real agent process.


def test_the_agent_drains_a_backlog_then_picks_up_new_traffic(tmp_path, servers):
    all_rows, received = rows(5), []
    source = servers(cursor_source(all_rows))
    sink = servers(collecting_sink(received))
    process = run_agent(tmp_path, config_for(source, sink))
    try:
        wait_for(lambda: len(received) == 5)
        all_rows.extend(rows(2, start=6))
        wait_for(lambda: len(received) == 7)
        assert sorted(int(t["id"]) for t in received) == list(range(1, 8))
        wait_for(lambda: state_of(tmp_path).get("cursor") == 7)
    finally:
        stop_agent(tmp_path, process)
    assert state_of(tmp_path)["status"] == "stopped"


def test_the_cursor_holds_while_the_sink_refuses_so_nothing_is_lost(tmp_path, servers):
    received, sink_status = [], {"now": 429}
    source = servers(cursor_source(rows(3)))
    sink = servers(collecting_sink(received, lambda: sink_status["now"]))
    process = run_agent(tmp_path, config_for(source, sink))
    try:
        wait_for(lambda: state_of(tmp_path).get("status") == "capped")
        assert state_of(tmp_path)["cursor"] == 0
        assert received == []
        sink_status["now"] = 202
        wait_for(lambda: len(received) == 3, seconds=30)
    finally:
        stop_agent(tmp_path, process)
    assert state_of(tmp_path)["cursor"] == 3


def test_restarting_resumes_from_the_cursor_and_never_loses_or_duplicates(tmp_path, servers):
    received = []
    source = servers(cursor_source(rows(4)))
    sink = servers(collecting_sink(received))
    process = run_agent(tmp_path, config_for(source, sink))
    wait_for(lambda: len(received) == 4)
    # The cursor is saved just after the sink accepts a page. A kill in between repeats that page,
    # which the sink ignores, and is covered below; this checks a resume from a saved cursor.
    wait_for(lambda: state_of(tmp_path).get("cursor") == 4)
    process.kill()
    process.wait()
    before = len(source.requests)
    process = run_agent(tmp_path, config_for(source, sink))
    try:
        wait_for(lambda: len(source.requests) > before)
        assert "since=4" in source.requests[before]["path"]
    finally:
        stop_agent(tmp_path, process)
    assert len(received) == 4


def test_a_kill_before_the_cursor_is_saved_repeats_a_page_without_duplicates(tmp_path, servers):
    received = []
    source = servers(cursor_source(rows(4)))
    sink = servers(collecting_sink(received))
    (tmp_path).mkdir(parents=True, exist_ok=True)
    # As if the agent died after the sink took everything, with its cursor still at the first page.
    (tmp_path / "state.json").write_text(json.dumps({"cursor": 2, "mapping": None}))
    received.extend({"id": str(i)} for i in range(1, 5))
    process = run_agent(tmp_path, config_for(source, sink))
    try:
        wait_for(lambda: state_of(tmp_path).get("cursor") == 4)
    finally:
        stop_agent(tmp_path, process)
    assert "since=2" in source.requests[0]["path"]
    assert sorted(int(t["id"]) for t in received) == [1, 2, 3, 4]
    assert state_of(tmp_path)["duplicates"] >= 2


def test_a_second_copy_exits_rather_than_competing(tmp_path, servers):
    source = servers(cursor_source(rows(1)))
    sink = servers(collecting_sink([]))
    first = run_agent(tmp_path, config_for(source, sink))
    try:
        wait_for(lambda: state_of(tmp_path).get("status") == "syncing")
        second = run_agent(tmp_path, config_for(source, sink))
        assert second.wait(timeout=10) == 0
        assert first.poll() is None
    finally:
        stop_agent(tmp_path, first)


def test_bad_credentials_are_reported_without_the_reply_or_the_secret(tmp_path, servers):
    source = servers(lambda *_: (401, SECRET.encode(), {}))
    sink = servers(collecting_sink([]))
    config = config_for(source, sink, auth={"type": "bearer"})
    process = run_agent(tmp_path, config, {"FORGE_SOURCE_SECRET": SECRET})
    try:
        wait_for(lambda: state_of(tmp_path).get("status") == "source_auth_failed")
    finally:
        stop_agent(tmp_path, process)
    assert source.requests[0]["headers"]["Authorization"] == f"Bearer {SECRET}"
    assert SECRET not in (tmp_path / "state.json").read_text()


@pytest.mark.parametrize(
    ("auth", "header", "expected"),
    [
        ({"type": "basic", "username": "pk"}, "Authorization", "Basic cGs6c2VjcmV0"),
        ({"type": "api-key", "header": "apikey"}, "apikey", "secret"),
        ({"type": "api-key", "header": "X-Key", "prefix": "Token "}, "X-Key", "Token secret"),
    ],
)
def test_each_credential_form_is_sent_as_the_source_expects(monkeypatch, auth, header, expected):
    monkeypatch.setenv("FORGE_SOURCE_SECRET", "secret")
    request = agent.source_request({"url": "https://x.test/logs", "auth": auth}, None, None)
    assert request.get_header(header.capitalize()) == expected


def test_a_redirect_is_never_followed_with_the_credential(tmp_path, servers):
    elsewhere = servers(lambda *_: (200, b"[]", {}))
    source = servers(lambda *_: (302, b"", {"Location": f"{elsewhere.url}/stolen"}))
    sink = servers(collecting_sink([]))
    config = config_for(source, sink, auth={"type": "bearer"})
    process = run_agent(tmp_path, config, {"FORGE_SOURCE_SECRET": SECRET})
    try:
        wait_for(lambda: state_of(tmp_path).get("status") == "source_error")
    finally:
        stop_agent(tmp_path, process)
    assert elsewhere.requests == []


def test_page_walks_commit_the_cursor_only_after_the_last_page(tmp_path, servers):
    # Newest first, as Langfuse may return them: committing mid-walk would skip the older pages.
    pages = {
        None: {
            "data": [{"id": "b", "prompt": "p2", "output": "o2", "startTime": "2030-01-02"}],
            "meta": {"cursor": "n"},
        },
        "n": {"data": [{"id": "a", "prompt": "p1", "output": "o1", "startTime": "2030-01-01"}], "meta": {}},
    }

    def respond(path, _):
        return 200, json.dumps(pages[parse_qs(urlsplit(path).query).get("cursor", [None])[0]]).encode(), {}

    received = []
    source = servers(respond)
    sink = servers(collecting_sink(received))
    config = config_for(
        source,
        sink,
        cursor_param="from",
        start_cursor="2030-01-01",
        cursor_field="startTime",
        page_cursor_param="cursor",
        page_cursor_path="meta.cursor",
    )
    process = run_agent(tmp_path, config)
    try:
        wait_for(lambda: len(received) == 2)
        wait_for(lambda: state_of(tmp_path).get("cursor") == "2030-01-02")
    finally:
        stop_agent(tmp_path, process)


# A sandbox's secrets work through an https:// proxy, which urllib can't speak, so curl carries the
# requests there. Plain http isn't sent to that proxy, so these reach the servers through real curl.
CURL = {"HTTPS_PROXY": "https://proxy.invalid:9"}


def test_the_agent_works_end_to_end_through_curl(tmp_path, servers):
    received = []
    source = servers(cursor_source(rows(5)))
    sink = servers(collecting_sink(received))
    config = config_for(source, sink, auth={"type": "bearer"})
    process = run_agent(
        tmp_path, config, {**CURL, "FORGE_SOURCE_SECRET": SECRET, "FORGE_INGEST_TOKEN": TOKEN}
    )
    try:
        wait_for(lambda: len(received) == 5)
        wait_for(lambda: state_of(tmp_path).get("cursor") == 5)
    finally:
        stop_agent(tmp_path, process)
    assert source.requests[0]["headers"]["Authorization"] == f"Bearer {SECRET}"
    assert sink.requests[0]["headers"]["Authorization"] == f"Bearer {TOKEN}"
    assert sink.requests[0]["headers"]["Content-type"] == "application/json"
    assert state_of(tmp_path)["accepted"] == 5


@pytest.mark.parametrize(
    ("answer", "status"),
    [(401, "source_auth_failed"), (403, "source_auth_failed"), (500, "source_error"), (302, "source_error")],
)
def test_curl_reports_a_source_the_same_way_urllib_does(tmp_path, servers, answer, status):
    elsewhere = servers(lambda *_: (200, b"[]", {}))
    source = servers(lambda *_: (answer, b"", {"Location": f"{elsewhere.url}/stolen"}))
    sink = servers(collecting_sink([]))
    process = run_agent(tmp_path, config_for(source, sink, auth={"type": "bearer"}), CURL)
    try:
        wait_for(lambda: state_of(tmp_path).get("status") == status)
    finally:
        stop_agent(tmp_path, process)
    assert elsewhere.requests == []


def test_curl_holds_the_cursor_when_the_sink_refuses(tmp_path, servers):
    source = servers(cursor_source(rows(2)))
    sink = servers(lambda *_: (429, b"{}", {}))
    process = run_agent(tmp_path, config_for(source, sink), CURL)
    try:
        wait_for(lambda: state_of(tmp_path).get("status") == "capped")
    finally:
        stop_agent(tmp_path, process)
    assert state_of(tmp_path)["cursor"] == 0


def test_an_unreachable_host_through_curl_is_an_error_not_a_crash(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "https://proxy.invalid:9")
    request = agent.urllib.request.Request("http://127.0.0.1:9/nothing")
    with pytest.raises(OSError):
        agent.http_call(request)


def test_curl_never_sees_a_credential_or_url_as_an_argument(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "https://proxy.invalid:9")
    monkeypatch.setenv("FORGE_SOURCE_SECRET", SECRET)
    seen = {}

    def fake_run(command, **options):
        seen["command"] = command
        return SimpleNamespace(returncode=0, stdout=b"[]\n200")

    monkeypatch.setattr(agent.subprocess, "run", fake_run)
    url = f"https://x.test/logs?key={SECRET}"
    request = agent.source_request({"url": url, "auth": {"type": "bearer"}}, None, None)
    assert agent.http_call(request) == (200, b"[]")
    assert SECRET not in " ".join(seen["command"])
    assert "x.test" not in " ".join(seen["command"])


def test_header_values_cannot_inject_curl_options():
    quoted = agent._curl_quote('a"b\\c\nurl = "https://elsewhere.test"')
    assert "\n" not in quoted
    assert quoted == '"a\\"b\\\\curl = \\"https://elsewhere.test\\""'


# Through the lab: connect once, and traffic arrives.


@pytest.fixture
def lab(tmp_path, monkeypatch, servers):
    from services import api

    monkeypatch.setattr(store, "DATA", tmp_path)
    monkeypatch.setattr(store, "campaigns", lambda: [])
    monkeypatch.setitem(vms.PROVIDERS, "superserve", vms.SuperServeVM())
    monkeypatch.delenv("SUPERSERVE_API_KEY", raising=False)
    traces._sealed_cache.clear()
    client = TestClient(api.app, base_url="http://127.0.0.1:8000")
    text = "question,answer\n" + "".join(
        f"How many items are in order {i}?,#### {i % 7}\n" for i in range(200)
    )
    pack_id = client.post("/packs", json={"draft": DRAFT, "text": text}).json()["id"]

    def sink(_, body):
        batch = json.loads(body)
        return (
            202,
            json.dumps(traces.ingest(batch["traces"], batch["source"], batch["task_pack"])).encode(),
            {},
        )

    monkeypatch.setenv("FORGE_API_URL", servers(sink).url)
    yield SimpleNamespace(client=client, pack=pack_id, data=tmp_path)
    for connection in connections.listing()["connections"]:
        vms.PROVIDERS["local"].remove({"id": connection["id"]})


def connect(lab, source, **fields):
    body = {
        "name": "Support bot",
        "task_pack": lab.pack,
        "kind": "http",
        "url": f"{source.url}/logs",
        "cursor_param": "since",
        "start_cursor": "0",
        "data_terms": TERMS,
        **fields,
    }
    reply = lab.client.post("/overseer/connections", json=body)
    assert reply.status_code == 201, reply.text
    connection_id = reply.json()["id"]
    # The API's minimum is 15 seconds; these tests don't wait that long.
    row = connections._get(connection_id)
    config = {**json.loads(row["config"]), "poll_seconds": 1}
    vms.PROVIDERS["local"].write_config(row, config)
    connections._update(connection_id, config=json.dumps(config))
    return connection_id


def stored_traces():
    return db.rows("select trace_id from traces order by trace_id")


def test_connecting_once_brings_traffic_in_with_no_further_steps(lab, servers):
    source = servers(cursor_source(rows(4)))
    connection_id = connect(lab, source)
    wait_for(lambda: len(stored_traces()) == 4)
    wait_for(lambda: agent_state(lab, connection_id).get("cursor") == 4)
    assert lab.client.post("/overseer/cycle").json()["checked"] == 4
    listed = lab.client.get("/overseer/connections", params={"task_pack": lab.pack}).json()
    connection = listed["connections"][0]
    assert connection["status"] == "syncing"
    assert connection["counts"]["accepted"] == 4
    assert connection["mapping"]["prompt"] == "request.messages"
    assert connection["host"] == "127.0.0.1"
    assert lab.client.get("/overseer/summary", params={"task_pack": lab.pack}).json()["traces"] == 4


def agent_state(lab, connection_id):
    return state_of(lab.data / "overseer-agents" / connection_id)


def test_the_credential_only_exists_in_a_private_file(lab, servers):
    source = servers(cursor_source(rows(1)))
    connection_id = connect(lab, source, auth_type="bearer", secret=SECRET)
    wait_for(lambda: len(stored_traces()) == 1)
    lab.client.post("/overseer/cycle")
    assert source.requests[0]["headers"]["Authorization"] == f"Bearer {SECRET}"
    assert SECRET not in json.dumps(lab.client.get("/overseer/connections").json())
    assert SECRET not in json.dumps(db.rows("select * from connections"))
    directory = lab.data / "overseer-agents" / connection_id
    for name in ("config.json", "state.json", "agent.log"):
        assert SECRET not in (directory / name).read_text()
    assert stat.S_IMODE((directory / "secrets.json").stat().st_mode) == 0o600
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    lab.client.post(f"/overseer/connections/{connection_id}/disconnect")
    assert not directory.exists()


def test_a_query_string_key_never_reaches_the_api(lab, servers):
    source = servers(cursor_source(rows(1)))
    connect(lab, source, url=f"{source.url}/logs?key={SECRET}")
    assert SECRET not in json.dumps(lab.client.get("/overseer/connections").json())


def test_an_unreadable_shape_asks_for_a_mapping_then_flows(lab, servers):
    records = [{"id": 1, "payload": {"said": "How many?", "replied": "#### 1"}}]
    source = servers(lambda *_: (200, json.dumps(records).encode(), {}))
    connection_id = connect(lab, source, cursor_param=None, start_cursor=None)
    wait_for(lambda: agent_state(lab, connection_id).get("status") == "needs_mapping")
    lab.client.post("/overseer/cycle")
    listed = lab.client.get("/overseer/connections").json()["connections"][0]
    assert "payload.said" in listed["fields_seen"]
    alerts = lab.client.get("/overseer/summary", params={"task_pack": lab.pack}).json()["alerts"]
    assert [a["kind"] for a in alerts] == ["connector_down"]
    assert json.loads(alerts[0]["detail"])["connection"] == "Support bot"
    reply = lab.client.post(
        f"/overseer/connections/{connection_id}/mapping",
        json={"mapping": {"prompt": "payload.said", "output": "payload.replied"}},
    )
    assert reply.status_code == 200
    wait_for(lambda: len(stored_traces()) == 1)


def test_the_supervisor_revives_a_dead_agent(lab, servers):
    all_rows = rows(1)
    source = servers(cursor_source(all_rows))
    connection_id = connect(lab, source)
    wait_for(lambda: len(stored_traces()) == 1)
    directory = lab.data / "overseer-agents" / connection_id
    (directory / "stop").touch()
    wait_for(lambda: agent_state(lab, connection_id).get("status") == "stopped")
    state = agent_state(lab, connection_id)
    state["heartbeat_at"] = "2000-01-01T00:00:00+00:00"
    (directory / "state.json").write_text(json.dumps(state))
    all_rows.extend(rows(1, start=2))
    connections.supervise()
    assert connections._get(connection_id)["restarts"] == 1
    wait_for(lambda: len(stored_traces()) == 2)
    wait_for(lambda: agent_state(lab, connection_id).get("cursor") == 2)
    connections.supervise()
    assert connections._get(connection_id)["restarts"] == 0


def test_an_agent_that_will_not_recover_is_given_up_on_and_alerted(lab, servers, monkeypatch):
    source = servers(cursor_source(rows(1)))
    connection_id = connect(lab, source)
    provider = vms.PROVIDERS["local"]
    monkeypatch.setattr(provider, "read", lambda _: {"heartbeat_at": "2000-01-01T00:00:00+00:00"})
    attempts = []
    monkeypatch.setattr(provider, "ensure_running", attempts.append)
    for _ in range(connections.MAX_RESTARTS + 1):
        connections.supervise()
    row = connections._get(connection_id)
    assert row["status"] == "failed"
    assert len(attempts) == connections.MAX_RESTARTS
    connections.supervise()
    assert len(attempts) == connections.MAX_RESTARTS  # No more spending once it has given up.
    alerts = db.rows("select * from alerts where kind = 'connector_down'")
    assert "did not recover" in alerts[0]["detail"]


def test_requests_that_cannot_work_are_refused_up_front(lab, servers):
    source = servers(cursor_source([]))
    base = {"name": "x", "task_pack": lab.pack, "data_terms": TERMS}
    post = lambda **f: lab.client.post("/overseer/connections", json={**base, **f})  # noqa: E731
    assert post(kind="http").status_code == 422
    assert post(kind="http", url="http://example.com/logs").status_code == 422
    assert post(kind="http", url="ftp://example.com").status_code == 422
    assert post(kind="http", url="https://user:pw@example.com").status_code == 422
    assert post(kind="http", url="https://x.test/?since={cursor}").status_code == 422
    assert post(kind="http", url=f"{source.url}/l", auth_type="bearer").status_code == 422
    assert post(kind="http", url=f"{source.url}/l", mapping={"nonsense": "a"}).status_code == 422
    assert post(kind="http", url=f"{source.url}/l", task_pack="missing").status_code == 404
    assert post(kind="http", url=f"{source.url}/l", provider="superserve").status_code == 422
    assert post(kind="langfuse", auth_type="basic", secret=SECRET).status_code == 422
    assert (
        lab.client.post(
            "/overseer/connections",
            json={**base, "kind": "http", "url": f"{source.url}/l", "data_terms": "short"},
        ).status_code
        == 422
    )
    assert connections.listing()["connections"] == []


def test_without_a_vm_account_the_local_provider_is_used_and_says_why(lab):
    described = lab.client.get("/overseer/connections").json()
    assert described["default"] == "local"
    assert "SUPERSERVE_API_KEY" in described["providers"]["superserve"]["reason"]


def test_langfuse_preset_uses_the_v2_cursor_api_and_starts_from_now(lab):
    config = connections.source_config(
        {"kind": "langfuse", "username": "pk-lf-1", "secret": SECRET, "auth_type": "basic"},
        vms.PROVIDERS["superserve"],
    )
    assert config["url"].startswith("https://cloud.langfuse.com/api/public/v2/observations?type=GENERATION")
    assert "fields=core,basic,io,model,metrics,usage" in config["url"]
    assert config["auth"] == {"type": "basic", "username": "pk-lf-1"}
    assert config["start_cursor"] > "2020"
    assert SECRET not in json.dumps(config)
    observation = {
        "id": "g1",
        "input": json.dumps([{"role": "user", "content": "Hi"}]),
        "output": "Hello",
        "startTime": "2030-01-01T00:00:00Z",
        "model": "m1",
        "latency": 1.5,
        "usageDetails": {"input": 3, "output": 4},
        "environment": "production",
    }
    mapping = agent.discover([observation], config["mapping"])
    trace = agent.to_trace(observation, mapping, config["mapping"])
    assert trace["prompt"] == "Hi"
    assert trace["latency_ms"] == 1500
    assert trace["tokens"] == {"prompt": 3, "completion": 4}
    assert trace["deployment"] == "production"


# The VM, against a double of the SuperServe SDK.


class NotFoundError(Exception):
    pass


class FakeFiles:
    def __init__(self, vm):
        self.vm = vm

    def write(self, path, content, timeout=None):
        self.vm.written[path] = content

    def read(self, path, max_bytes=None, timeout=None):
        if path not in self.vm.written:
            raise NotFoundError(path)
        return self.vm.written[path]


class FakeCommands:
    def __init__(self, vm):
        self.vm = vm

    def run(self, command, timeout_seconds=None):
        self.vm.commands_run.append(command)
        return SimpleNamespace(exit_code=self.vm.exit_code(command))


class FakeVM:
    def __init__(self, vm_id, options):
        self.id, self.options = vm_id, options
        self.written, self.commands_run = {}, []
        self.exit_code = lambda _: 0
        self.files, self.commands = FakeFiles(self), FakeCommands(self)


class FakeCloud:
    def __init__(self):
        self.vms, self.killed, self.secrets, self.deleted = {}, [], {}, []
        self.created = 0
        self.fail_create = None
        self.exit_code = lambda _: 0

    # The Sandbox surface.
    def create(self, **options):
        if self.fail_create:
            raise self.fail_create
        self.created += 1
        vm = FakeVM(f"vm-{self.created}", options)
        vm.exit_code = self.exit_code
        self.vms[vm.id] = vm
        return vm

    def connect(self, vm_id):
        if vm_id not in self.vms:
            raise NotFoundError(vm_id)
        return self.vms[vm_id]

    def kill_by_id(self, vm_id):
        self.vms.pop(vm_id, None)
        self.killed.append(vm_id)


class FakeSecrets:
    def __init__(self, cloud):
        self.cloud = cloud

    def create(self, **fields):
        self.cloud.secrets[fields["name"]] = fields

    def delete_by_name(self, name):
        if name not in self.cloud.secrets:
            raise NotFoundError(name)
        del self.cloud.secrets[name]
        self.cloud.deleted.append(name)


@pytest.fixture
def cloud(lab, monkeypatch):
    double = FakeCloud()
    monkeypatch.setitem(
        vms.PROVIDERS,
        "superserve",
        vms.SuperServeVM(double, FakeSecrets(double), lambda **f: SimpleNamespace(**f)),
    )
    monkeypatch.setenv("SUPABASE_URL", "https://proj.supabase.co")
    monkeypatch.setattr(pull, "configured", lambda: True)
    monkeypatch.setattr(pull, "pull", lambda: {"pulled": 0})
    double.revoked = []
    monkeypatch.setattr(
        pull,
        "create_token",
        lambda source, pack, cap: {"id": f"tok-{source}", "token": TOKEN, "daily_cap": cap},
    )
    monkeypatch.setattr(pull, "revoke_token", lambda token_id: double.revoked.append(token_id))
    return double


def vm_connect(lab, **fields):
    body = {
        "name": "Support bot",
        "task_pack": lab.pack,
        "kind": "http",
        "url": "https://logs.customer.example/traces",
        "auth_type": "bearer",
        "secret": SECRET,
        "data_terms": TERMS,
        **fields,
    }
    return lab.client.post("/overseer/connections", json=body)


def test_the_vm_is_locked_down_and_never_holds_a_real_credential(lab, cloud):
    reply = vm_connect(lab)
    assert reply.status_code == 201
    assert reply.json()["provider"] == "superserve"
    assert lab.client.get("/overseer/connections").json()["default"] == "superserve"
    row = connections._get(reply.json()["id"])
    vm = cloud.vms[row["vm_id"]]
    options = vm.options
    assert options["network"].deny_out == ["0.0.0.0/0"]
    assert set(options["network"].allow_out) == {
        "1.1.1.1/32",
        "8.8.8.8/32",
        "logs.customer.example",
        "proj.supabase.co",
    }
    assert options["preview_access"] == "private"
    assert options["timeout_seconds"] == 3600  # The dead-man switch: no lab, no billing.
    assert options["auto_delete_seconds"] == 7 * 24 * 3600
    assert options["secrets"] == {
        "FORGE_SOURCE_SECRET": f"forge-{row['id']}-source",
        "FORGE_INGEST_TOKEN": f"forge-{row['id']}-ingest",
    }
    # Everything that entered the VM, and everything said to the VM API: no real credential.
    everything = json.dumps(
        {
            "options": {k: v for k, v in options.items() if k != "network"},
            "written": {k: v.decode() for k, v in vm.written.items() if k.endswith(".json")},
            "commands": vm.commands_run,
        }
    )
    assert SECRET not in everything
    assert TOKEN not in everything
    assert b"import" in vm.written["/tmp/forge-agent/agent.py"]
    assert any("nohup python3 -I agent.py" in c for c in vm.commands_run)
    # The real values sit with the provider, each usable on one host only.
    source = cloud.secrets[f"forge-{row['id']}-source"]
    ingest = cloud.secrets[f"forge-{row['id']}-ingest"]
    assert (source["value"], source["hosts"]) == (SECRET, ["logs.customer.example"])
    assert source["auth"] == {"type": "bearer"}
    assert (ingest["value"], ingest["hosts"]) == (TOKEN, ["proj.supabase.co"])
    # And they are not in the lab's own records.
    assert SECRET not in json.dumps(db.rows("select * from connections"))
    assert TOKEN not in json.dumps(db.rows("select * from connections"))


def test_basic_and_header_credentials_are_described_to_the_provider(lab, cloud):
    connection_id = vm_connect(lab, auth_type="api-key", header="X-Key", prefix="Token ").json()["id"]
    assert cloud.secrets[f"forge-{connection_id}-source"]["auth"] == {
        "type": "api-key",
        "header": "X-Key",
        "prefix": "Token ",
    }
    other = vm_connect(lab, kind="langfuse", url=None, auth_type="basic", username="pk-lf").json()["id"]
    assert cloud.secrets[f"forge-{other}-source"]["auth"] == {"type": "basic", "username": "pk-lf"}
    assert cloud.secrets[f"forge-{other}-source"]["hosts"] == ["cloud.langfuse.com"]


def test_a_source_with_no_credential_binds_only_the_ingest_token(lab, cloud):
    connection_id = vm_connect(lab, auth_type="none", secret=None).json()["id"]
    row = connections._get(connection_id)
    assert list(cloud.vms[row["vm_id"]].options["secrets"]) == ["FORGE_INGEST_TOKEN"]


def test_a_failed_boot_leaves_nothing_behind(lab, cloud):
    cloud.exit_code = lambda _: 1  # A template with no Python.
    connection_id = vm_connect(lab).json()["id"]
    row = connections._get(connection_id)
    assert row["status"] == "failed"
    assert "Python 3" in row["detail"]
    assert cloud.vms == {}
    assert len(cloud.killed) == 1
    assert cloud.secrets == {}
    assert cloud.revoked == [f"tok-{connection_id}"]
    assert row["secret_names"] is None
    reply = lab.client.post(f"/overseer/connections/{connection_id}/restart")
    assert reply.status_code == 409


def test_a_provider_error_message_is_never_shown_or_stored(lab, cloud):
    cloud.fail_create = RuntimeError(f"bad request echoing {SECRET}")
    connection_id = vm_connect(lab).json()["id"]
    row = connections._get(connection_id)
    assert row["status"] == "failed"
    assert row["detail"] == "The provider failed (RuntimeError)"


def test_a_lost_vm_is_replaced_from_the_last_reported_cursor(lab, cloud):
    connection_id = vm_connect(lab, cursor_param="since", start_cursor="0").json()["id"]
    first = connections._get(connection_id)["vm_id"]
    cloud.vms[first].written["/tmp/forge-agent/state.json"] = json.dumps(
        {"status": "syncing", "cursor": 41, "heartbeat_at": store.now(), "accepted": 41}
    ).encode()
    connections.supervise()
    row = connections._get(connection_id)
    assert (row["status"], row["cursor"]) == ("syncing", "41")
    assert json.loads(row["agent"])["accepted"] == 41

    del cloud.vms[first]  # Auto-deleted after a long pause, or lost.
    connections.supervise()
    row = connections._get(connection_id)
    assert row["vm_id"] != first
    assert row["status"] == "starting"
    replacement = cloud.vms[row["vm_id"]]
    assert json.loads(replacement.written["/tmp/forge-agent/config.json"])["start_cursor"] == "41"
    assert replacement.options["secrets"]["FORGE_SOURCE_SECRET"].endswith("-source")


def test_a_stalled_agent_is_restarted_inside_its_vm(lab, cloud):
    connection_id = vm_connect(lab).json()["id"]
    vm = cloud.vms[connections._get(connection_id)["vm_id"]]
    vm.written["/tmp/forge-agent/state.json"] = json.dumps(
        {"status": "syncing", "heartbeat_at": "2000-01-01T00:00:00+00:00"}
    ).encode()
    launches = sum("nohup" in c for c in vm.commands_run)
    connections.supervise()
    assert sum("nohup" in c for c in vm.commands_run) == launches + 1
    assert connections._get(connection_id)["vm_id"] == vm.id


def test_giving_up_stops_the_vm_but_keeps_credentials_for_a_retry(lab, cloud):
    connection_id = vm_connect(lab).json()["id"]
    first = connections._get(connection_id)["vm_id"]
    connections._update(connection_id, restarts=connections.MAX_RESTARTS)
    connections.supervise()
    row = connections._get(connection_id)
    assert row["status"] == "failed"
    assert first in cloud.killed
    assert len(cloud.secrets) == 2
    reply = lab.client.post(f"/overseer/connections/{connection_id}/restart")
    assert reply.status_code == 200
    assert reply.json()["has_vm"] is True


def test_an_unreachable_control_plane_costs_no_attempt_and_no_new_vm(lab, cloud, monkeypatch):
    connection_id = vm_connect(lab).json()["id"]
    created = len(cloud.vms)

    def down(_):
        raise ConnectionError("control plane down")

    monkeypatch.setattr(cloud, "connect", down)
    connections.supervise()
    row = connections._get(connection_id)
    assert row["restarts"] == 0
    assert "ConnectionError" in row["detail"]
    assert len(cloud.vms) == created


def test_disconnecting_removes_the_vm_the_credentials_and_the_token(lab, cloud):
    connection_id = vm_connect(lab).json()["id"]
    assert lab.client.post(f"/overseer/connections/{connection_id}/disconnect").status_code == 200
    assert cloud.vms == {}
    assert cloud.secrets == {}
    assert cloud.revoked == [f"tok-{connection_id}"]
    assert connections.listing()["connections"] == []


def test_a_failed_disconnect_is_not_forgotten(lab, cloud, monkeypatch):
    connection_id = vm_connect(lab).json()["id"]

    def down(_):
        raise ConnectionError("down")

    monkeypatch.setattr(cloud, "kill_by_id", down)
    assert lab.client.post(f"/overseer/connections/{connection_id}/disconnect").status_code == 502
    row = connections._get(connection_id)
    assert row["status"] == "disconnect_failed"
    assert row["vm_id"]
    monkeypatch.setattr(cloud, "kill_by_id", FakeCloud.kill_by_id.__get__(cloud))
    assert lab.client.post(f"/overseer/connections/{connection_id}/disconnect").status_code == 200
    assert cloud.vms == {}


def test_the_number_of_paid_vms_is_capped(lab, cloud, monkeypatch):
    monkeypatch.setenv("FORGE_OVERSEER_MAX_VMS", "1")
    assert vm_connect(lab).status_code == 201
    refused = vm_connect(lab)
    assert refused.status_code == 422
    assert "FORGE_OVERSEER_MAX_VMS" in refused.json()["detail"]


def test_a_vm_refuses_sources_it_could_never_reach(lab, cloud):
    assert vm_connect(lab, url="http://logs.customer.example").status_code == 422
    refused = vm_connect(lab, url="https://127.0.0.1/logs")
    assert refused.status_code == 422
    assert "private network" in refused.json()["detail"]
