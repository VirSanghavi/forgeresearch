"""Freestyle VMs for the research agents, against a scripted Freestyle and Responses API."""

import json

import pytest

from services.core import agents, workbench


class FakeFreestyle:
    def __init__(self, fail_delete=False):
        self.calls = []
        self.fail_delete = fail_delete
        self.n = 0

    def create(self, slug, ttl_seconds, idle_seconds, auto_delete_seconds, metadata):
        self.n += 1
        self.calls.append(("create", slug, ttl_seconds, idle_seconds, auto_delete_seconds, metadata))
        return {"id": f"vm-{self.n}", "slug": slug, "resources": {"cpu": 4}}

    def exec(self, vm_id, command, timeout_seconds, stdin=None):
        self.calls.append(("exec", vm_id, command, stdin))
        if "python3" in command:
            return {"statusCode": 0, "stdout": f"ran {stdin.decode()}\n", "stderr": ""}
        return {"statusCode": 0, "stdout": "", "stderr": ""}

    def pause(self, vm_id):
        self.calls.append(("pause", vm_id))

    def delete(self, vm_id):
        self.calls.append(("delete", vm_id))
        if self.fail_delete:
            raise RuntimeError("Freestyle DELETE: 503")

    def of(self, kind):
        return [c for c in self.calls if c[0] == kind]


@pytest.fixture
def bench(monkeypatch):
    for name in ("FORGE_VM_RUNS_PER_AGENT", "FORGE_VM_IDLE_SECONDS", "FORGE_VM_AUTO_DELETE_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    events = []
    b = workbench.Workbenches(
        "forge-ab12CD34", 600, client=FakeFreestyle(), on_event=lambda *e: events.append(e)
    )
    b.events = events
    token = workbench.current.set(b)
    yield b
    workbench.current.reset(token)


def test_slugs_are_valid_freestyle_slugs():
    assert workbench.slug("forge-AB_12", "data") == "forge-forge-ab-12-data"
    assert len(workbench.slug("x" * 80, "governance")) == 63


def test_each_role_gets_one_vm_reused_across_runs_and_subagents(bench):
    fs = bench.client
    out = bench.run_python("data", "data", "print(1)", {"failures": [1]})
    bench.run_python("data", "data", "print(2)", {"failures": [1]})
    bench.run_python("data", "truncation analyst", "print(3)", {"failures": [1]})
    bench.run_python("evaluator", "evaluator", "print(4)", {"x": 2})
    assert [c[1] for c in fs.of("create")] == ["forge-forge-ab12cd34-data", "forge-forge-ab12cd34-evaluator"]
    create = fs.of("create")[0]
    # Deleted by Freestyle itself half an hour after the campaign's deadline, and when left paused.
    assert create[2] == 600 + 1800 and create[3] == 60 and create[4] == 1800
    assert out["stdout"] == "ran print(1)\n" and out["exit_code"] == 0 and out["vm"] == "vm-1"
    writes = [c for c in fs.of("exec") if "evidence.json" in c[2]]
    # The evidence is written once per folder; a subagent works in its own folder.
    assert [c[2].split(" ")[2] for c in writes] == [
        workbench.WORK,
        f"{workbench.WORK}/truncation-analyst",
        workbench.WORK,
    ]
    assert json.loads(writes[0][3]) == {"failures": [1]}
    runs = [c for c in fs.of("exec") if "python3" in c[2]]
    assert f"cd {workbench.WORK}/truncation-analyst" in runs[2][2] and "timeout 60 python3" in runs[2][2]


def test_close_deletes_every_vm_and_reports_runs(bench):
    bench.run_python("data", "data", "print(1)", {})
    bench.run_python("trainer", "trainer", "print(1)", {})
    summary = bench.close()
    assert sorted(c[1] for c in bench.client.of("delete")) == ["vm-1", "vm-2"]
    assert all(v["deleted"] for v in summary) and sum(v["runs"] for v in summary) == 2
    assert bench.events[-1][1] == "Deleted 2 of 2 Freestyle VMs after 2 agent runs."
    assert bench.close() == []  # Nothing is deleted twice.


def test_a_failed_delete_is_reported_not_raised(monkeypatch):
    b = workbench.Workbenches("c", 60, client=FakeFreestyle(fail_delete=True))
    b.run_python("data", "data", "print(1)", {})
    [vm] = b.close()
    assert not vm["deleted"] and "503" in vm["error"]


# ----- The agent loop --------------------------------------------------------------------------


class Reply:
    status_code = 200

    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


def answer(rid, value):
    return {
        "id": rid,
        "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(value)}]}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def run_code(rid, *codes):
    return {
        "id": rid,
        "output": [
            {
                "type": "function_call",
                "name": "run_python",
                "call_id": f"{rid}-{i}",
                "arguments": json.dumps({"code": c}),
            }
            for i, c in enumerate(codes)
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


@pytest.fixture
def api(monkeypatch):
    for name in (
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "FORGE_AGENT_MODEL",
        "FORGE_AGENT_MAX_CALLS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    requests, replies = [], []

    def post(url, headers, json, timeout):
        requests.append(json)
        return Reply(replies.pop(0)(json))

    monkeypatch.setattr(agents.httpx, "post", post)
    return requests, replies


SCHEMA = agents._text({"summary": {"type": "string"}})


def test_an_agent_runs_code_in_its_vm_then_answers_from_what_it_printed(api, bench):
    requests, replies = api
    replies += [
        lambda b: run_code("r1", "print(len(failures))"),
        lambda b: answer("r2", {"summary": "9 of 11 hit the token limit"}),
    ]
    result, session = agents.consult("data", agents.DATA, {"failures": []}, SCHEMA)
    assert result == {"summary": "9 of 11 hit the token limit"}
    first, second = requests
    assert {t["name"] for t in first["tools"]} == {"spawn_subagents", "run_python"}
    assert "own Linux VM" in first["instructions"]
    sent = json.loads(second["input"][0]["output"])
    assert sent["stdout"] == "ran print(len(failures))\n" and second["previous_response_id"] == "r1"
    assert session.transcript()["vm_runs"] == 1 and session.held == 0
    assert bench.client.of("pause") == [("pause", "vm-1")]  # Paused as soon as the decision ends.


def test_runs_are_capped_and_the_answer_call_is_always_kept(api, bench, monkeypatch):
    monkeypatch.setenv("FORGE_AGENT_MAX_CALLS", "3")
    bench.limits["runs_per_agent"] = 5
    requests, replies = api
    replies += [
        lambda b: run_code("r1", "print(1)"),
        lambda b: run_code("r2", "print(2)"),
        lambda b: answer("r3", {"summary": "done"}),
    ]
    result, session = agents.consult("data", agents.DATA, {}, SCHEMA)
    assert result == {"summary": "done"} and session.calls == 3 and session.held == 0
    # The third and last call is the held answer: no tools, so it cannot spend more.
    assert requests[2]["tool_choice"] == "none" and requests[1].get("tool_choice") is None


def test_extra_runs_in_one_round_are_refused(api, bench):
    bench.limits["runs_per_agent"] = 1
    requests, replies = api
    replies += [lambda b: run_code("r1", "print(1)", "print(2)"), lambda b: answer("r2", {"summary": "ok"})]
    agents.consult("data", agents.DATA, {}, SCHEMA)
    outputs = [json.loads(o["output"]) for o in requests[1]["input"]]
    assert outputs[0]["exit_code"] == 0 and outputs[1] == {
        "error": "No runs left. Answer with what you have."
    }
    assert requests[1]["tool_choice"] == "none"


def test_a_vm_failure_goes_back_to_the_agent(api, bench):
    def broken(*a, **k):
        raise RuntimeError("Freestyle POST /v5/vms: 429 too many VMs")

    bench.client.create = broken
    requests, replies = api
    replies += [lambda b: run_code("r1", "print(1)"), lambda b: answer("r2", {"summary": "estimated"})]
    result, _ = agents.consult("data", agents.DATA, {}, SCHEMA)
    assert result == {"summary": "estimated"}
    assert "429" in json.loads(requests[1]["input"][0]["output"])["error"]


def test_without_freestyle_no_vm_tool_is_offered(api):
    requests, replies = api
    replies += [lambda b: answer("r1", {"summary": "small"})]
    agents.consult("data", agents.DATA, {}, SCHEMA)
    assert "tools" not in requests[0] or all(t["name"] != "run_python" for t in requests[0]["tools"])
    assert "own Linux VM" not in requests[0]["instructions"]
