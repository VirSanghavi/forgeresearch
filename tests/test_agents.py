"""Model-backed agents and their subagents, against a scripted Responses API."""

import json

import pytest

from services.core import agents, director


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


def spawn(rid, names):
    return {
        "id": rid,
        "output": [
            {
                "type": "function_call",
                "name": "spawn_subagents",
                "call_id": f"call-{rid}",
                "arguments": json.dumps({"agents": [{"name": n, "brief": f"Look at {n}"} for n in names]}),
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


@pytest.fixture
def api(monkeypatch):
    for name in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "FORGE_AGENT_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    requests = []
    script = {}

    def post(url, headers, json, timeout):
        requests.append(json)
        task = (json.get("input") or "") if isinstance(json.get("input"), str) else ""
        for marker, reply in script.items():
            if marker(json, task):
                return Reply(reply(json, task))
        raise AssertionError(f"unscripted request {json}")

    monkeypatch.setattr(agents.httpx, "post", post)
    return requests, script


def is_parent(body, task):
    return '"task"' not in task and "previous_response_id" not in body


def test_agent_spins_up_subagents_and_uses_their_findings(api):
    requests, script = api
    script[lambda b, t: is_parent(b, t)] = lambda b, t: spawn("r-parent", ["arithmetic", "formatting"])
    script[lambda b, t: "Look at" in t] = lambda b, t: answer(
        "r-sub", {"finding": "q-3 drops a carry" if "arithmetic" in t else "q-7 never writes ####"}
    )
    script[lambda b, t: b.get("previous_response_id") == "r-parent"] = lambda b, t: answer(
        "r-final", {"summary": "two patterns"}
    )

    result, session = agents.consult(
        "data", agents.DATA, {"failures": []}, agents._text({"summary": {"type": "string"}})
    )

    assert result == {"summary": "two patterns"}
    # The parent may delegate; its follow-up carries both findings and must answer, not delegate again.
    first, *subs, last = requests
    assert first["tools"][0]["name"] == "spawn_subagents"
    findings = json.loads(last["input"][0]["output"])["results"]
    assert {f["name"] for f in findings} == {"arithmetic", "formatting"}
    assert {f["finding"] for f in findings} == {"q-3 drops a carry", "q-7 never writes ####"}
    assert last["tool_choice"] == "none"
    assert session.calls == 4
    assert session.usage == {"input_tokens": 40, "output_tokens": 20}
    helpers = session.subagents()
    assert sorted(h["name"] for h in helpers) == ["arithmetic", "formatting"]
    assert all(h["parent"] == "Data researcher" and h["depth"] == 1 and h["finding"] for h in helpers)
    transcript = session.transcript()
    assert transcript["agents"][0]["children"] and transcript["provider"] == "OpenAI"


def test_subagents_at_the_depth_limit_cannot_delegate(api, monkeypatch):
    monkeypatch.setenv("FORGE_AGENT_DEPTH", "1")
    requests, script = api
    script[lambda b, t: is_parent(b, t)] = lambda b, t: spawn("r-parent", ["helper"])
    script[lambda b, t: "Look at" in t] = lambda b, t: answer("r-sub", {"finding": "done"})
    script[lambda b, t: "previous_response_id" in b] = lambda b, t: answer("r-final", {"summary": "ok"})
    agents.consult("data", agents.DATA, {}, agents._text({"summary": {"type": "string"}}))
    subagent_request = next(r for r in requests if "Look at" in str(r.get("input")))
    assert "tools" not in subagent_request


def test_requests_beyond_the_subagent_cap_are_dropped(api, monkeypatch):
    monkeypatch.setenv("FORGE_AGENT_MAX_SUBAGENTS", "2")
    requests, script = api
    script[lambda b, t: is_parent(b, t)] = lambda b, t: spawn("r-parent", ["a", "b", "c", "d"])
    script[lambda b, t: "Look at" in t] = lambda b, t: answer("r-sub", {"finding": "x"})
    script[lambda b, t: "previous_response_id" in b] = lambda b, t: answer("r-final", {"summary": "ok"})
    _, session = agents.consult("data", agents.DATA, {}, agents._text({"summary": {"type": "string"}}))
    assert len(session.subagents()) == 2


def test_the_call_budget_always_leaves_room_for_the_answer(api, monkeypatch):
    monkeypatch.setenv("FORGE_AGENT_MAX_CALLS", "4")
    requests, script = api
    script[lambda b, t: is_parent(b, t)] = lambda b, t: spawn("r-parent", ["a", "b", "c"])
    # Subagents would delegate again if they could; with this budget they are not offered the tool.
    script[lambda b, t: "Look at" in t and "tools" in b] = lambda b, t: spawn("r-sub", ["x"])
    script[lambda b, t: "Look at" in t] = lambda b, t: answer("r-sub", {"finding": "x"})
    script[lambda b, t: "previous_response_id" in b] = lambda b, t: answer("r-final", {"summary": "ok"})
    result, session = agents.consult("data", agents.DATA, {}, agents._text({"summary": {"type": "string"}}))
    # One call for the parent, two subagents (the third does not fit), and the parent's answer.
    assert result == {"summary": "ok"}
    assert session.calls == 4 and session.held == 0
    assert len(session.subagents()) == 2
    assert all("tools" not in r for r in requests if "Look at" in str(r.get("input")))


def test_nested_subagents_share_one_budget_and_every_level_answers(api, monkeypatch):
    monkeypatch.setenv("FORGE_AGENT_MAX_CALLS", "8")
    requests, script = api
    script[lambda b, t: is_parent(b, t)] = lambda b, t: spawn("r-parent", ["a", "b"])
    script[lambda b, t: "Look at a" in t and "tools" in b] = lambda b, t: spawn("r-a", ["a1", "a2", "a3"])
    script[lambda b, t: "previous_response_id" in b and b["previous_response_id"] == "r-a"] = lambda b, t: (
        answer("r-a-final", {"finding": "a combined"})
    )
    script[lambda b, t: "previous_response_id" in b] = lambda b, t: answer("r-final", {"summary": "ok"})
    script[lambda b, t: "Look at" in t] = lambda b, t: answer("r-leaf", {"finding": "leaf"})
    result, session = agents.consult("data", agents.DATA, {}, agents._text({"summary": {"type": "string"}}))
    assert result == {"summary": "ok"}
    assert session.calls <= 8 and session.held == 0
    names = [h["name"] for h in session.subagents()]
    assert {"a", "b"} <= set(names)
    assert any(h["depth"] == 2 for h in session.subagents())


def test_a_failed_subagent_is_reported_to_its_parent(api):
    requests, script = api
    script[lambda b, t: is_parent(b, t)] = lambda b, t: spawn("r-parent", ["broken"])

    def boom(b, t):
        raise RuntimeError("upstream 500")

    script[lambda b, t: "Look at" in t] = boom
    script[lambda b, t: "previous_response_id" in b] = lambda b, t: answer("r-final", {"summary": "ok"})
    result, session = agents.consult("data", agents.DATA, {}, agents._text({"summary": {"type": "string"}}))
    assert result == {"summary": "ok"}
    sent = json.loads(requests[-1]["input"][0]["output"])["results"][0]
    assert sent["name"] == "broken" and "upstream 500" in sent["error"]


def test_director_delegates_and_reports_its_subagents(api):
    requests, script = api
    proposal = {
        "strategy": "broad_sft",
        "learning_rate": 0.0002,
        "lora_rank": 8,
        "hypothesis": "Most failures are format slips; train worked solutions.",
        "focus": "formatting",
    }
    script[lambda b, t: is_parent(b, t)] = lambda b, t: spawn("r-dir", ["format check"])
    script[lambda b, t: "Look at" in t] = lambda b, t: answer("r-sub", {"finding": "5 of 8 lack ####"})
    script[lambda b, t: "previous_response_id" in b] = lambda b, t: answer("r-dir-final", proposal)
    config = {"objective": "Improve math", "strategies": ["broad_sft", "targeted_sft"]}
    development = {"score": 0.5, "n": 2, "predictions": []}
    result = director.propose(config, development, [], {"objective": "Improve math"})
    assert result["strategy"] == "broad_sft"
    assert result["director"] == f"OpenAI · {agents.model('director')}"
    assert result["response_id"] == "r-dir-final"
    assert [s["name"] for s in result["subagents"]] == ["format check"]
    assert result["agent_session"]["calls"] == 3
    assert "spawn subagents" in requests[0]["instructions"]


def test_without_keys_agents_are_off_and_the_director_stays_local(monkeypatch):
    for name in ("AZURE_OPENAI_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert agents.status() == {
        "connected": False,
        "provider": None,
        "models": {},
        "subagents": None,
        "vms": None,
    }
    config = {"objective": "Improve math", "strategies": ["broad_sft"]}
    development = {"score": 0.5, "n": 2, "predictions": []}
    assert director.propose(config, development, [])["director"] == "Evidence-driven local director"


def test_each_role_can_use_its_own_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("FORGE_AGENT_MODEL", "team-model")
    monkeypatch.setenv("FORGE_AGENT_MODEL_GOVERNANCE", "council-model")
    models = agents.status()["models"]
    assert models["governance"] == "council-model"
    assert models["director"] == models["data"] == "team-model"


def test_evidence_helpers_never_include_correct_answers_as_failures():
    before = {
        "predictions": [
            {"id": "a", "question": "q", "output": "1", "expected": "1", "correct": True},
            {"id": "b", "question": "q", "output": "2", "expected": "3", "correct": False},
        ]
    }
    after = {
        "predictions": [
            {"id": "a", "question": "q", "output": "9", "expected": "1", "correct": False},
            {"id": "b", "question": "q", "output": "3", "expected": "3", "correct": True},
        ]
    }
    assert [f["id"] for f in agents.failures(before)] == ["b"]
    diff = agents.changes(before, after)
    assert [r["id"] for r in diff["fixed"]] == ["b"]
    assert [r["id"] for r in diff["broke"]] == ["a"]
    assert diff["still_wrong"] == 0


def test_director_is_asked_again_instead_of_repeating_identical_training(api):
    requests, script = api
    same = {
        "strategy": "targeted_sft",
        "learning_rate": 0.0001,
        "lora_rank": 8,
        "hypothesis": "Train the production failures again.",
        "focus": "production_failures",
    }
    changed = same | {"learning_rate": 0.0002, "hypothesis": "A higher learning rate this time."}
    script[lambda b, t: "rejected_proposal" in t] = lambda b, t: answer("r-2", changed)
    script[lambda b, t: True] = lambda b, t: answer("r-1", same)
    config = {"objective": "Improve math", "strategies": ["targeted_sft", "broad_sft"]}
    previous = [{"strategy": "targeted_sft", "learning_rate": 0.0001, "lora_rank": 8, "score": 0.8}]
    development = {"score": 0.8, "n": 2, "predictions": []}
    result = director.propose(config, development, previous, {"objective": "Improve math"})
    assert result["learning_rate"] == 0.0002
    assert "would reproduce the same weights" in requests[-1]["input"]
    assert result["agent_session"]["calls"] == 2


def test_director_refuses_a_second_identical_proposal(api):
    requests, script = api
    same = {
        "strategy": "targeted_sft",
        "learning_rate": 0.0001,
        "lora_rank": 8,
        "hypothesis": "Train the production failures again.",
        "focus": "production_failures",
    }
    script[lambda b, t: True] = lambda b, t: answer("r-1", same)
    config = {"objective": "Improve math", "strategies": ["targeted_sft"]}
    previous = [{"strategy": "targeted_sft", "learning_rate": 0.0001, "lora_rank": 8, "score": 0.8}]
    with pytest.raises(ValueError, match="refusing to rerun identical training"):
        director.propose(config, {"score": 0.8, "n": 2, "predictions": []}, previous, {"objective": "x"})


@pytest.mark.parametrize("budget", ["7", "8", "9", "10", "11", "12"])
def test_siblings_that_all_delegate_never_starve_one_another(api, monkeypatch, budget):
    monkeypatch.setenv("FORGE_AGENT_MAX_CALLS", budget)
    requests, script = api
    script[lambda b, t: is_parent(b, t)] = lambda b, t: spawn("r-parent", ["a", "b", "c"])
    import time

    # "a" decides to delegate last, after its siblings have started their own subagents.
    def late(n):
        def reply(b, t):
            time.sleep({"a": 0.3, "b": 0.0, "c": 0.1}[n])
            return spawn(f"r-{n}", [f"{n}1", f"{n}2"])

        return reply

    for name in "abc":
        script[lambda b, t, n=name: f"Look at {n}" in t and "tools" in b] = late(name)
    script[lambda b, t: b.get("previous_response_id") == "r-parent"] = lambda b, t: answer(
        "r-final", {"summary": "ok"}
    )
    script[lambda b, t: "previous_response_id" in b] = lambda b, t: answer("r-mid", {"finding": "combined"})
    script[lambda b, t: "Look at" in t] = lambda b, t: answer("r-leaf", {"finding": "leaf"})
    for _ in range(3):
        result, session = agents.consult(
            "data", agents.DATA, {}, agents._text({"summary": {"type": "string"}})
        )
        assert result == {"summary": "ok"}
        assert session.held == 0 and session.calls <= int(budget)
        assert not [h for h in session.subagents() if "error" in h], session.subagents()
