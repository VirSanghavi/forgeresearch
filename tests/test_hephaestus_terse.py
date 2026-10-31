"""What Terse and full command of the research team add to Hephaestus, against a real native runtime.

A scripted model stands in for Azure here. These tests exercise the native runtime integration
without making claims about live campaign results.
"""

import importlib.util
import json
import re
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import conftest
import httpx
import pytest
from fastapi.testclient import TestClient
from test_hephaestus import CHAT, FakeLab, Messages, said, script_model, starts, text_reply, tool_call

from services.core import agents, director, store, team
from services.core.hephaestus import brain, desk, harness, ledger, operations, tools
from services.core.hephaestus.service import Service

LOCAL = "http://127.0.0.1:8000"
CODE = re.compile(r'confirm ([0-9a-f]{4})"')


class Lab(FakeLab):
    """The fake lab, plus the routes Hephaestus's new tools read."""

    def __call__(self, request):
        if request.url.path == "/agents/usage":
            return httpx.Response(
                200, json={"model_calls": 3, "input_tokens": 10, "output_tokens": 5, "by_model": {}}
            )
        return super().__call__(request)


def _start(tmp_path, monkeypatch, runtime=None):
    monkeypatch.setattr(store, "DATA", tmp_path)
    monkeypatch.setattr(tools, "free_memory_percent", lambda: 80)
    messages, lab = Messages(tmp_path / "chat.db"), Lab()
    client = httpx.Client(transport=httpx.MockTransport(lab), base_url=LOCAL)
    service = Service(CHAT, lab=tools.Lab(client), messages=messages)
    service.tick()
    return service, messages, lab


@pytest.fixture(autouse=True)
def no_real_model(monkeypatch):
    for name in (
        "AZURE_OPENAI_API_KEY",
        "OPENAI_API_KEY",
        "FORGE_HEPHAESTUS_AZURE_KEY",
        "FORGE_AGENT_MODEL_HEPHAESTUS",
        "FORGE_BACKEND",
    ):
        monkeypatch.delenv(name, raising=False)
    agents._deployments.clear()


@pytest.fixture
def world(terse, tmp_path, monkeypatch):
    return _start(tmp_path, monkeypatch)


@pytest.fixture
def real_lab(terse, tmp_path, monkeypatch):
    from services import api

    monkeypatch.setattr(store, "DATA", tmp_path)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    monkeypatch.setattr(store, "ARTIFACTS", artifacts)
    posted = []
    monkeypatch.setattr(api, "send", lambda *a, **o: posted.append((a, o)) or {"message_id": "relay-1"})
    desk._service.clear()
    return TestClient(api.app, base_url=LOCAL), api, posted


def _campaign(cid, **fields):
    store.save(
        {
            "id": cid,
            "name": "x",
            "config": {"strategies": ["broad_sft", "targeted_sft"]},
            "status": "training",
            "created_at": store.now(),
            **fields,
        }
    )
    return cid


# Limits and approvals: atomic, whichever process asks.


def test_concurrent_uses_can_never_pass_a_limit(terse):
    book = ledger.Ledger()
    with ThreadPoolExecutor(16) as pool:
        taken = list(pool.map(lambda _: book.take("campaigns", 5), range(16)))
    assert sum(taken) == 5
    assert book.used("campaigns") == 5


def test_an_approval_code_works_once_however_many_ask_at_once(terse):
    book = ledger.Ledger("chat")
    book.hold("hash", "stop_training", {"campaign_id": "forge-aaaaaaaa"}, "why", 60)
    with ThreadPoolExecutor(8) as pool:
        found = list(pool.map(lambda _: book.claim_approval("hash")["found"], range(8)))
    assert sum(found) == 1
    assert book.waiting_approvals() == []


def test_several_approvals_can_wait_at_once(world):
    service, _, lab = world
    lab.campaigns.append({"id": "forge-aaaaaaaa", "status": "running", "config": {}})
    first = brain.answer(service.context, "admin", "stop forge-aaaaaaaa")
    second = brain.answer(service.context, "admin", "computer delete")
    assert CODE.search(first)
    assert "Approval needed" in first
    assert "don't have a VM" in second or len(service.ledger.waiting_approvals()) == 2
    assert "Done: stop_training" in brain.answer(
        service.context, "admin", f"confirm {CODE.search(first).group(1)}"
    )


def test_the_owner_can_decline_and_the_code_is_spent(world):
    service, _, lab = world
    lab.campaigns.append({"id": "forge-aaaaaaaa", "status": "running", "config": {}})
    code = CODE.search(brain.answer(service.context, "admin", "stop forge-aaaaaaaa")).group(1)
    assert brain.answer(service.context, "viewer", f"decline {code}") == "Only the owner can decline that."
    assert "Nothing ran" in brain.answer(service.context, "admin", f"decline {code}")
    assert brain.answer(service.context, "admin", f"confirm {code}") == "Nothing is waiting for that code."
    assert not [r for r in lab.requests if r[1].endswith("/cancel")]


def test_only_a_hash_of_a_code_is_kept(world):
    service, _, lab = world
    lab.campaigns.append({"id": "forge-aaaaaaaa", "status": "running", "config": {}})
    code = CODE.search(brain.answer(service.context, "admin", "stop forge-aaaaaaaa")).group(1)
    kept = json.dumps([service.ledger.waiting_approvals(), service.ledger.read_audit(40)])
    assert code not in kept


def test_a_changed_or_missing_audit_row_breaks_the_chain():
    # Terse's storage isn't something a test can edit in place, so this runs the actor's own check over rows.
    spec = importlib.util.spec_from_file_location(
        "forge_actors", store.ROOT / "services/actors/src/actors.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows, previous = [], ""
    for seq, entry in enumerate(("first", "second", "third"), 1):
        link = module._chain(previous, entry)
        rows.append({"seq": seq, "entry": entry, "previous": previous, "hash": link})
        previous = link

    def verify(found):
        return json.loads(module.HephaestusActor.verify_audit(SimpleNamespace(_rows=lambda _: found)))

    assert verify(rows) == {"ok": True, "rows": 3, "broken_at": None}
    assert verify([rows[0], {**rows[1], "entry": "changed"}, rows[2]])["broken_at"] == 2
    assert verify([rows[0], rows[2]])["broken_at"] == 3


def test_the_doctor_reports_terse_and_a_model_that_isnt_deployed(world, monkeypatch):
    service, messages, _ = world
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake")
    monkeypatch.setenv("FORGE_AGENT_MODEL_HEPHAESTUS", "gpt-5-mini")
    monkeypatch.setattr(agents, "deployments", lambda *_: ["gpt-6-luna", "gpt-6.1-sol"])
    found = {c["name"]: c for c in harness.checks(service.context.lab, messages.path)}
    assert found["Terse"]["ok"] and "audit chain verified" in found["Terse"]["detail"]
    assert not found["Model"]["ok"] and "gpt-6.1-sol" in found["Model"]["detail"]


def test_with_terse_down_nothing_is_lost_or_done(tmp_path, monkeypatch):
    monkeypatch.setenv("DURABLE_ACTORS_CONTROL_PLANE_URL", "http://127.0.0.1:9")
    monkeypatch.setattr(ledger, "WRITE_RETRY_SECONDS", 0.2)
    monkeypatch.setattr(ledger, "READ_RETRY_SECONDS", 0.2)
    ledger._transports.clear()
    monkeypatch.setattr(store, "DATA", tmp_path)
    messages, lab = Messages(tmp_path / "chat.db"), Lab()
    client = httpx.Client(transport=httpx.MockTransport(lab), base_url=LOCAL)
    service = Service(CHAT, lab=tools.Lab(client), messages=messages)
    with pytest.raises(ledger.Unavailable):
        service.tick()
    assert messages.sent == []
    assert starts(lab) == []
    assert not harness._terse()["ok"]


# Crashes: nothing lost, nothing done twice.


class Died(BaseException):
    """The process dying, as far as the code running at that moment can tell."""


def test_a_real_crash_mid_turn_is_picked_up_without_repeating_what_happened(tmp_path, monkeypatch):
    runtime = conftest.Runtime(tmp_path).start()
    conftest.use_runtime(monkeypatch, runtime)
    data = tmp_path / "data"
    data.mkdir()
    try:
        service, messages, lab = _start(data, monkeypatch)
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake")
        script = [[tool_call("start_training", task_pack="gsm8k")]]

        def dies_after_acting(payload):
            if script:
                return {"id": "r1", "output": script.pop()}
            raise Died

        service.post = dies_after_acting
        messages.add("@hephaestus train the maths task", mine=True)
        with pytest.raises(Died):
            service.read_new()
        assert len(starts(lab)) == 1
        runtime.kill()
        runtime.start()

        # Back up. The model tries the same thing again, and the journal answers instead of the lab.
        post, sent = script_model(
            [tool_call("start_training", task_pack="gsm8k")], text_reply("It's running.")
        )
        reborn = Service(CHAT, lab=service.context.lab, messages=messages, post=post)
        # Terse fences a killed runtime's data for a while, and a write waits that out.
        reborn.ledger.advance(0)
        assert reborn.ledger.open_turns(reborn.recover_before)
        reborn.tick()
        assert len(starts(lab)) == 1
        assert [m["text"] for m in reborn.ledger.undelivered()] == []
        replies = said(messages)
        assert len(replies) == 1
        assert "picked it back up" in replies[0] and "It's running." in replies[0]
        note = " ".join(m["content"] for m in sent[0]["input"] if m["role"] == "developer")
        assert "Finished, don't repeat" in note and "start_training" in note
        # Nothing read twice, and nothing to pick up next time.
        reborn.tick()
        assert len(said(messages)) == 1
        assert reborn.ledger.open_turns(10**12) == []
    finally:
        runtime.stop()


def test_a_reply_written_but_not_sent_is_sent_after_a_restart(world):
    service, messages, _ = world
    turn = service.ledger.begin_turn("m1", "message", "admin", "hello")
    service.ledger.finish_turn(turn["id"], "The reply.", announce=True)
    assert messages.sent == []
    Service(CHAT, lab=service.context.lab, messages=messages).tick()
    assert said(messages) == ["The reply."]
    Service(CHAT, lab=service.context.lab, messages=messages).tick()
    assert said(messages) == ["The reply."]


def test_a_failed_send_is_retried_then_given_up_on(world):
    service, messages, _ = world
    service.ledger.post("hello")

    def refuses(*_):
        raise OSError("Messages is busy")

    real, messages.send = messages.send, refuses
    for _ in range(5):
        service.deliver()
    messages.send = real
    service.deliver()
    assert messages.sent == []


def test_a_turn_that_keeps_crashing_it_gets_an_apology_not_another_try(world, monkeypatch):
    service, messages, _ = world
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake")
    turn = service.ledger.begin_turn("m1", "message", "admin", "do the thing that crashes")
    for _ in range(brain.MAX_RESUMES):
        service.ledger.resume_turn(turn["id"])
    post, sent = script_model()
    reborn = Service(CHAT, lab=service.context.lab, messages=messages, post=post)
    reborn.tick()
    assert sent == []
    assert "interrupted" in said(messages)[0] and "do the thing" in said(messages)[0]
    assert reborn.ledger.open_turns(10**12) == []


def test_a_used_approval_is_never_run_again_after_a_crash(world):
    service, messages, lab = world
    lab.campaigns.append({"id": "forge-aaaaaaaa", "status": "running", "config": {}})
    code = CODE.search(brain.answer(service.context, "admin", "stop forge-aaaaaaaa")).group(1)
    service.ledger.claim_approval(tools._code_hash(code))
    service.ledger.begin_turn("m9", "message", "admin", f"confirm {code}")
    Service(CHAT, lab=service.context.lab, messages=messages).tick()
    assert "interrupted" in said(messages)[-1]
    assert not [r for r in lab.requests if r[1].endswith("/cancel")]


def test_an_action_of_unknown_outcome_isnt_sent_again_when_that_isnt_safe(world):
    service, _, lab = world
    turn = service.ledger.begin_turn("m1", "message", "admin", "export")
    arguments = {"method": "POST", "path": "/overseer/cycle"}
    key = tools._action_key(turn["id"], "lab_request", arguments)
    service.ledger.begin_action(key, turn["id"], "lab_request", arguments)
    service.context.turn = turn["id"]
    with pytest.raises(tools.ToolError, match="can't tell"):
        tools.run(service.context, "admin", "lab_request", arguments)
    assert not [r for r in lab.requests if r[1] == "/overseer/cycle"]


def test_one_action_twice_in_a_turn_happens_once(world, monkeypatch):
    service, _, lab = world
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake")
    tools.run(service.context, "operator", "start_training", {"task_pack": "gsm8k"})
    cid = lab.campaigns[0]["id"]
    stop = tool_call("stop_training", campaign_id=cid)
    post, _ = script_model([stop], [stop], text_reply("Stopped."))
    assert brain.answer(service.context, "admin", "stop it, really", post) == "Stopped."
    assert len([r for r in lab.requests if r[1].endswith("/cancel")]) == 1


def test_blank_optional_fields_from_a_model_ask_for_nothing(world):
    # Seen live on gpt-6.1-sol, which fills every optional field it's offered.
    service, _, lab = world
    blanks = {"strategies": [], "model_id": "", "max_seconds": 0, "dev_samples": 0, "final_samples": 0}
    started = tools.run(service.context, "admin", "start_training", {"task_pack": "gsm8k", **blanks})
    assert "started" in started
    config = starts(lab)[0][2]
    assert "model_id" not in config and "strategies" not in config
    assert config["max_seconds"] == 1800


def test_asking_for_less_than_the_everyday_limits_needs_no_approval(world):
    # Seen live: a mission asking for a short run was held as if it had asked for a long one.
    service, _, lab = world
    small = {"task_pack": "gsm8k", "max_seconds": 600, "dev_samples": 4, "final_samples": 4}
    assert "started" in tools.run(service.context, "admin", "start_training", small)
    assert starts(lab)[0][2]["max_seconds"] == 600
    lab.campaigns[0]["status"] = "completed"
    held = tools.run(service.context, "admin", "start_training", {"task_pack": "gsm8k", "max_seconds": 3600})
    assert "needs_confirmation" in held


def test_a_refused_start_gives_back_its_place_in_the_days_limit(world):
    service, _, lab = world
    lab.campaigns.append({"id": "forge-aaaaaaaa", "status": "running", "config": {}})
    with pytest.raises(tools.ToolError, match="owns the local GPU"):
        tools.run(service.context, "operator", "start_training", {"task_pack": "gsm8k"})
    assert service.ledger.used("campaigns") == 0


def test_the_same_key_starts_one_campaign_through_the_real_route(real_lab, monkeypatch):
    client, api, _ = real_lab

    class Authority:
        def initialize(self, _):
            return json.dumps({"status": "initialized"})

        def snapshot(self):
            return "{}"

    monkeypatch.setattr(api, "actor", lambda _: Authority())
    monkeypatch.setattr(api, "governance", lambda _: Authority())
    started = []
    monkeypatch.setattr(api, "spawn", started.append)
    headers = {"Idempotency-Key": "abc123"}
    first = client.post("/campaigns", json={}, headers=headers)
    second = client.post("/campaigns", json={}, headers=headers)
    assert first.status_code == second.status_code == 201
    assert first.json()["id"] == second.json()["id"]
    assert len(started) == 1


# Command of the research team, through the lab's real routes and a real TeamActor.


def test_only_an_admin_commands_the_team():
    control = {
        "team_status",
        "configure_agent",
        "ask_team",
        "hold_campaign",
        "release_campaign",
        "assign_experiment",
        "finish_research",
        "direct_team",
        "start_mission",
    }
    assert control <= set(tools.allowed("admin"))
    assert not control & set(tools.allowed("operator"))
    assert not control & set(tools.allowed("viewer"))


def test_an_order_reaches_the_whole_team_once_and_shows_who_read_it(real_lab):
    client, _, posted = real_lab
    cid = _campaign("forge-a0000001")
    order = {"id": "order-1", "roles": list(agents.ROLES), "text": "Cite question ids in everything."}
    assert client.post(f"/campaigns/{cid}/team/orders", json=order).status_code == 201
    assert client.post(f"/campaigns/{cid}/team/orders", json=order).status_code == 201
    assert len(posted) == 1
    assert posted[0][1] == {"route": "room"}
    assert team.read_orders(cid, "data", "exp-001") == ["Cite question ids in everything."]
    assert team.read_orders(cid, "data", "exp-001") == ["Cite question ids in everything."]
    roster = client.get(f"/campaigns/{cid}/team").json()
    assert roster["orders"][0]["read"] == [{"role": "data", "where": "exp-001"}]


def test_an_order_to_one_agent_stays_with_that_agent(real_lab):
    client, _, _ = real_lab
    cid = _campaign("forge-a0000002")
    client.post(
        f"/campaigns/{cid}/team/orders", json={"id": "o", "roles": ["trainer"], "text": "Watch the loss."}
    )
    assert team.read_orders(cid, "trainer", "e") == ["Watch the loss."]
    assert team.read_orders(cid, "director", "e") == []


def test_settings_are_checked_against_real_deployments_and_ceilings(real_lab, monkeypatch):
    client, _, _ = real_lab
    monkeypatch.setattr(agents, "deployments", lambda *_: ["gpt-6-luna", "gpt-6.1-sol"])
    cid = _campaign("forge-a0000003")
    route = f"/campaigns/{cid}/team/agents/director"
    refused = client.post(route, json={"model": "gpt-5-mini"})
    assert refused.status_code == 422 and "gpt-6.1-sol" in refused.json()["detail"]
    assert client.post(route, json={"max_calls": 99999}).status_code == 422
    assert client.post(route, json={"max_subagents": -1}).status_code == 422
    assert client.post(f"/campaigns/{cid}/team/agents/nobody", json={"enabled": False}).status_code == 404
    applied = client.post(route, json={"model": "gpt-6.1-sol", "max_subagents": 8}).json()
    assert applied["settings"] == {"model": "gpt-6.1-sol", "max_subagents": 8}
    assert team.settings_for(cid, "director")["model"] == "gpt-6.1-sol"
    assert client.get(f"/campaigns/{cid}/team").json()["agents"]["director"]["limits"]["max_subagents"] == 8
    # Null goes back to the lab's default.
    assert client.post(route, json={"model": None}).json()["settings"] == {"max_subagents": 8}


def test_a_session_runs_on_the_settings_it_was_given(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake")
    monkeypatch.setenv("FORGE_AGENT_MODEL", "gpt-6-luna")
    seen = []

    def post(payload):
        seen.append(payload["model"])
        return {
            "id": "r",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": '{"finding": "x"}'}]}],
        }

    monkeypatch.setattr(agents, "_post", post)
    settings = {"model": "gpt-6.1-sol", "max_calls": 30, "depth": 1}
    _, session = agents.consult("data", "Look.", {"a": 1}, agents.FINDING, settings)
    assert seen == ["gpt-6.1-sol"]
    assert session.limits["max_calls"] == 30 and session.limits["depth"] == 1
    assert session.limits["max_subagents"] == 4
    assert session.transcript()["model"] == "gpt-6.1-sol"
    agents.consult("data", "Look.", {"a": 1}, agents.FINDING)
    assert seen[-1] == "gpt-6-luna"


def test_standing_the_director_down_falls_back_to_the_local_director(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake")

    def never(*_, **__):
        raise AssertionError("The model was called")

    monkeypatch.setattr(agents, "consult", never)
    config = {"task_pack": "gsm8k", "strategies": ["broad_sft"], "objective": "Improve the maths task"}
    development = {"score": 0.5, "predictions": [], "n": 10, "correct": 5}
    found = director.propose(config, development, [], context={}, settings={"enabled": False})
    assert found["director"] == "Evidence-driven local director"


def test_an_assignment_is_checked_like_a_proposal_and_handed_out_once(real_lab):
    client, _, _ = real_lab
    earlier = {"strategy": "broad_sft", "learning_rate": 0.0002, "lora_rank": 8}
    cid = _campaign("forge-a0000004", experiments=[{"id": "exp-001", "proposal": earlier}])
    route = f"/campaigns/{cid}/team/assign"
    good = {
        "id": "a1",
        "strategy": "targeted_sft",
        "learning_rate": 0.0001,
        "lora_rank": 16,
        "hypothesis": "A higher rank should help the multi-step failures.",
        "focus": "multi_step",
    }
    assert client.post(route, json=good | {"strategy": "grpo"}).status_code == 422
    assert client.post(route, json=good | {"learning_rate": 5.0}).status_code == 422
    assert client.post(route, json=good | {"hypothesis": "short"}).status_code == 422
    repeat = good | {"strategy": "broad_sft", "learning_rate": 0.0002, "lora_rank": 8}
    assert client.post(route, json=repeat).status_code == 409
    assert client.post(route, json=good).status_code == 201
    control = team.control(cid)
    taken = json.loads(control.take_assignment("exp-002"))
    assert taken["proposal"]["strategy"] == "targeted_sft"
    # A retried call for the same experiment gets the same answer; the next experiment gets nothing.
    assert json.loads(control.take_assignment("exp-002"))["id"] == "a1"
    assert control.take_assignment("exp-003") == ""
    assert client.get(f"/campaigns/{cid}/team").json()["assigned"][0]["ran_as"] == "exp-002"


def test_a_hold_waits_until_released_and_a_stop_still_works(real_lab):
    client, _, _ = real_lab
    cid = _campaign("forge-a0000005")
    assert team.wait_while_held(cid, lambda: False) is False
    client.post(f"/campaigns/{cid}/team/hold", json={"reason": "Let me look at the first result."})
    naps, announced = [], []

    def nap(_):
        naps.append(1)
        if len(naps) == 3:
            client.post(f"/campaigns/{cid}/team/release")

    assert team.wait_while_held(cid, lambda: False, announced.append, nap) is True
    assert len(naps) == 3
    assert announced[0]["reason"] == "Let me look at the first result."
    client.post(f"/campaigns/{cid}/team/hold", json={"reason": "Again."})
    assert team.wait_while_held(cid, lambda: True, sleep=lambda _: pytest.fail("It should stop")) is True


def test_finish_is_recorded_for_the_worker_and_commands_stop_when_the_campaign_ends(real_lab):
    client, _, _ = real_lab
    cid = _campaign("forge-a0000006")
    assert client.post(f"/campaigns/{cid}/team/finish", json={"reason": "Good enough."}).status_code == 200
    assert team.snapshot(cid)["finish"]["reason"] == "Good enough."
    ended = _campaign("forge-a0000007", status="completed")
    for route, body in (("hold", {"reason": "x"}), ("finish", {"reason": "x"}), ("release", None)):
        assert client.post(f"/campaigns/{ended}/team/{route}", json=body).status_code == 409
    assert client.get(f"/campaigns/{ended}/team").status_code == 200


def test_a_command_still_lands_when_relay_is_down(real_lab, monkeypatch):
    client, api, _ = real_lab

    def down(*_, **__):
        raise RuntimeError("Agent Relay delivery failed")

    monkeypatch.setattr(api, "send", down)
    cid = _campaign("forge-a0000008")
    assert client.post(f"/campaigns/{cid}/team/hold", json={"reason": "Stop there."}).status_code == 200
    assert team.snapshot(cid)["held"]["reason"] == "Stop there."


def test_a_stood_down_agent_says_so_instead_of_replying(real_lab, monkeypatch):
    client, _, _ = real_lab
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake")
    monkeypatch.setattr(agents, "deployments", lambda *_: None)
    cid = _campaign("forge-a0000009")
    client.post(f"/campaigns/{cid}/team/agents/data", json={"enabled": False})
    reply = client.post(f"/campaigns/{cid}/agents/data/messages", json={"text": "Why?"}).json()
    assert reply["reply"] is None and "stood down" in reply["note"]


def test_an_agents_own_words_only_reach_a_model_that_already_sees_them(real_lab, monkeypatch):
    client, _, _ = real_lab
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake")
    seen = []

    class Session:
        def transcript(self):
            return {"calls": 1, "usage": {}}

        def subagents(self):
            return []

    def consult(role, instructions, evidence, schema, settings=None):
        seen.append(evidence)
        return {"reply": "Ok."}, Session()

    monkeypatch.setattr(agents, "consult", consult)
    cid = _campaign("forge-a000000a", diagnosis={"summary": "QUOTES an example"})
    client.post(f"/campaigns/{cid}/agents/data/messages", json={"text": "Why?"})
    assert "QUOTES" not in json.dumps(seen[-1])
    client.post(f"/campaigns/{cid}/agents/data/messages", json={"text": "Why?", "own_work": True})
    assert "QUOTES" in json.dumps(seen[-1])


@pytest.mark.parametrize(
    ("surface", "own", "sees"),
    [("agent-chat", False, True), ("agent-chat", True, False), ("imessage", False, False)],
)
def test_hephaestus_only_asks_for_their_words_where_they_may_go(terse, monkeypatch, surface, own, sees):
    for name in ("ENDPOINT", "KEY", "MODEL"):
        if own:
            monkeypatch.setenv(f"FORGE_HEPHAESTUS_AZURE_{name}", "x")
        else:
            monkeypatch.delenv(f"FORGE_HEPHAESTUS_AZURE_{name}", raising=False)
    bodies = []

    def lab(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(201, json={"delivered": True, "reply": "Ok.", "note": None})

    client = httpx.Client(transport=httpx.MockTransport(lab), base_url=LOCAL)
    context = SimpleNamespace(lab=tools.Lab(client), surface=surface, ledger=ledger.Ledger("c"))
    tools.run(
        context, "admin", "message_agent", {"campaign_id": "forge-a000000b", "agent": "data", "text": "Why?"}
    )
    assert bodies[0]["own_work"] is sees


def test_asking_the_team_hears_from_each_and_survives_one_failing(terse):
    def lab(request):
        role = request.url.path.split("/")[4]
        if role == "trainer":
            return httpx.Response(503, json={"detail": "Agent Relay is unavailable, so nothing was sent."})
        return httpx.Response(201, json={"delivered": True, "reply": f"{role} here.", "note": None})

    client = httpx.Client(transport=httpx.MockTransport(lab), base_url=LOCAL)
    context = SimpleNamespace(lab=tools.Lab(client), surface="agent-chat", ledger=ledger.Ledger("c"))
    replies = tools.run(
        context, "admin", "ask_team", {"campaign_id": "forge-a000000c", "question": "Thoughts?"}
    )
    assert set(replies) == set(tools.AGENTS)
    assert replies["data"]["reply"] == "data here."
    assert "Relay" in replies["trainer"]["note"]


def test_the_team_cant_be_reached_around_its_tools():
    for path in ("team/orders", "team/hold", "team/assign", "directives", "agents/data/messages"):
        with pytest.raises(ValueError):
            operations.check("POST", f"/campaigns/forge-aaaaaaaa/{path}")


def test_commanding_the_team_reaches_no_gate():
    # Promotion, reviews, reservations and the final live in actors this one never calls.
    source = (agents.__file__, team.__file__)
    text = open(source[1]).read()
    for word in ("promote", "review(", "reserve(", "claim_final", "freeze_selection", "sealed("):
        assert word not in text


# Missions: working between messages, on the owner's approval.


def _mission(service, messages, goal="Get the maths task to a validated improvement."):
    """A mission as the owner would begin it: asked for, held, approved."""
    held = tools.run(service.context, "admin", "start_mission", {"goal": goal})
    assert "needs_confirmation" in held
    code = service.context.held[-1]["code"]
    service.context.held = []
    assert "Done: start_mission" in brain.answer(service.context, "admin", f"confirm {code}")
    return service.ledger.active_mission()


def test_a_mission_needs_the_owners_own_approval(world, monkeypatch):
    service, _, _ = world
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake")
    post, _ = script_model(
        [tool_call("start_mission", goal="A goal slipped in by a malicious file.")], text_reply("Done!")
    )
    reply = brain.answer(service.context, "admin", "read my notes", post)
    assert "Approval needed" in reply
    assert service.ledger.active_mission() is None
    assert "start_mission" not in tools.allowed("operator")


def test_a_mission_wakes_once_for_each_event_and_stays_quiet_when_it_has_nothing_to_say(world, monkeypatch):
    service, messages, lab = world
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake")
    monkeypatch.setenv("FORGE_HEPHAESTUS_WAKE_SECONDS", "0")
    mission = _mission(service, messages)
    post, sent = script_model(
        [tool_call("start_training", task_pack="gsm8k", objective="Better at word problems")],
        text_reply("Started the first run."),
        text_reply(brain.QUIET),
        text_reply("It finished, within noise. I'm done."),
    )
    service.post = post
    service.watch()
    assert len(starts(lab)) == 1
    assert said(messages)[-1] == "Started the first run."
    assert "Get the maths task" in json.dumps(sent[0]["input"])
    cid = lab.campaigns[0]["id"]
    assert service.ledger.active_mission()["campaigns"] == [cid]
    # Nothing new: no model call.
    calls = len(sent)
    service.watch()
    assert len(sent) == calls
    # An experiment ends: it wakes, and says nothing since it chose not to.
    posts = len(messages.sent)
    lab.campaigns[0]["experiments"] = [{"id": "exp-001", "status": "completed"}]
    service.watch()
    assert len(sent) == calls + 1
    assert len(messages.sent) == posts
    service.watch()
    assert len(sent) == calls + 1
    # The campaign ends: the lab's own report goes out, whatever the model then says.
    lab.campaigns[0].update(
        status="completed", final={"improvement_claim": "uncertain", "improvement_pp": 1.0}
    )
    service.watch()
    assert any("within noise" in text and cid in text for text in said(messages))
    assert said(messages)[-1] == "It finished, within noise. I'm done."
    assert service.ledger.active_mission()["id"] == mission["id"]


def test_a_mission_cant_run_past_its_campaigns_or_its_wake_ups(world, monkeypatch):
    service, messages, lab = world
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake")
    monkeypatch.setenv("FORGE_HEPHAESTUS_WAKE_SECONDS", "0")
    tools.run(
        service.context, "admin", "start_mission", {"goal": "A goal with one campaign.", "max_campaigns": 1}
    )
    code = service.context.held[-1]["code"]
    service.context.held = []
    brain.answer(service.context, "admin", f"confirm {code}")
    tools.run(service.context, "operator", "start_training", {"task_pack": "gsm8k"})
    lab.campaigns[0]["status"] = "completed"
    with pytest.raises(tools.ToolError, match="used its 1 campaigns"):
        tools.run(service.context, "operator", "start_training", {"task_pack": "gsm8k"})
    mission = service.ledger.active_mission()
    book = service.ledger
    for event in range(mission["max_wakes"]):
        assert book.claim_wake(mission["id"], f"e{event}", 0)["wake"]
    assert book.claim_wake(mission["id"], "one-more", 0) == {"wake": False, "why": "limit"}
    assert book.active_mission() is None
    assert book.missions()[0]["status"] == "paused"


def test_only_one_mission_at_a_time_and_the_owner_can_end_it(world, monkeypatch):
    service, messages, _ = world
    _mission(service, messages)
    with pytest.raises(tools.ToolError, match="already active"):
        tools.run(service.context, "admin", "start_mission", {"goal": "A second mission."}, confirmed=True)
    assert "Mission stopped" in brain.command(service.context, "admin", "mission stop")
    assert service.ledger.active_mission() is None
    assert "finished" in brain.command(service.context, "admin", "missions")


def test_a_check_in_fires_once_even_across_a_restart(world, monkeypatch):
    service, messages, _ = world
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake")
    service.ledger.schedule("Look at the overnight run.", 0)
    service.ledger.schedule("Not yet.", 3600)
    post, sent = script_model(text_reply("Nothing changed overnight."))
    service.post = post
    service.watch()
    assert said(messages) == ["Nothing changed overnight."]
    assert "overnight run" in json.dumps(sent[0]["input"])
    Service(CHAT, lab=service.context.lab, messages=messages, post=post).watch()
    assert len(sent) == 1
    assert [c["note"] for c in service.ledger.scheduled()] == ["Not yet."]


def test_waking_with_no_model_says_so_instead_of_pretending(world):
    service, messages, _ = world
    service.ledger.schedule("Look at it.", 0)
    service.watch()
    assert "no model is connected" in said(messages)[0]


# Azure.


def test_credits_are_measured_from_the_providers_counts_and_a_token_limit_holds(world, monkeypatch):
    service, _, _ = world
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake")
    monkeypatch.setenv("FORGE_AGENT_MODEL_HEPHAESTUS", "gpt-6.1-sol")
    monkeypatch.setenv("FORGE_HEPHAESTUS_DAILY_TOKENS", "100")
    calls = []

    def post(payload):
        calls.append(payload)
        usage = {
            "input_tokens": 90,
            "output_tokens": 20,
            "input_tokens_details": {"cached_tokens": 40},
            "output_tokens_details": {"reasoning_tokens": 5},
        }
        return {"id": "r", "output": text_reply("Hello."), "usage": usage}

    assert brain.answer(service.context, "operator", "how are things?", post) == "Hello."
    assert "limit of tokens" in brain.answer(service.context, "operator", "how are things?", post)
    assert len(calls) == 1
    assert calls[0]["model"] == "gpt-6.1-sol"
    found = tools.run(service.context, "operator", "credits", {})
    today = found["hephaestus_today"]
    assert today["tokens"] == 110 and today["cached_input_tokens"] == 40 and today["tokens_left"] == 0
    assert today["by_model"] == {"gpt-6.1-sol": 110}
    assert found["research_agents_today"]["model_calls"] == 3
    assert "110 of 100 tokens" in brain.command(service.context, "operator", "credits")


def _reply(status, **headers):
    return httpx.Response(
        status, json={"id": "r", "output": []}, headers=headers, request=httpx.Request("POST", "http://x")
    )


def test_a_busy_provider_is_waited_out_as_it_asks(monkeypatch):
    replies = [_reply(429, **{"retry-after-ms": "1500"}), _reply(503), _reply(200)]
    monkeypatch.setattr(agents.httpx, "post", lambda *_, **__: replies.pop(0))
    naps = []
    assert agents.responses("http://x", {}, {}, sleep=naps.append)["id"] == "r"
    assert naps == [1.5, 2.0]


def test_a_dropped_connection_is_retried_and_a_lasting_one_is_raised(monkeypatch):
    def down(*_, **__):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(agents.httpx, "post", down)
    naps = []
    with pytest.raises(httpx.ConnectError):
        agents.responses("http://x", {}, {}, sleep=naps.append)
    assert len(naps) == agents.MODEL_ATTEMPTS - 1


@pytest.mark.parametrize("status", [400, 401, 404])
def test_a_refused_request_isnt_repeated(monkeypatch, status):
    calls = []
    monkeypatch.setattr(agents.httpx, "post", lambda *_, **__: calls.append(1) or _reply(status))
    with pytest.raises(httpx.HTTPStatusError):
        agents.responses("http://x", {}, {}, sleep=lambda _: None)
    assert len(calls) == 1


def test_deployments_come_from_the_resource_not_the_catalog(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://lab.openai.azure.com/openai/v1")
    seen = []

    def get(url, **options):
        seen.append(url)
        return httpx.Response(
            200,
            json={"data": [{"id": "gpt-6.1-sol"}, {"id": "gpt-6-luna"}]},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(agents.httpx, "get", get)
    assert agents.deployments() == ["gpt-6-luna", "gpt-6.1-sol"]
    assert agents.deployments() == ["gpt-6-luna", "gpt-6.1-sol"]
    assert seen == ["https://lab.openai.azure.com/openai/deployments"]


def test_a_wrong_deployment_name_is_explained_without_the_providers_words(world, monkeypatch):
    service, _, _ = world
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake")
    monkeypatch.setenv("FORGE_AGENT_MODEL_HEPHAESTUS", "gpt-5-mini")
    monkeypatch.setattr(agents, "deployments", lambda *_: ["gpt-6-luna", "gpt-6.1-sol"])

    def missing(_):
        response = httpx.Response(
            404, text="SECRET request details", request=httpx.Request("POST", "http://x")
        )
        raise httpx.HTTPStatusError("SECRET", request=response.request, response=response)

    reply = brain.answer(service.context, "admin", "how are things?", missing)
    assert "gpt-5-mini" in reply and "gpt-6.1-sol" in reply
    assert "SECRET" not in reply


def test_its_last_step_has_to_answer(world, monkeypatch):
    service, _, _ = world
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake")
    script = [[tool_call("lab_status")] for _ in range(brain.MAX_ROUNDS - 1)] + [
        text_reply("Here's what I found.")
    ]
    post, sent = script_model(*script)
    assert brain.answer(service.context, "admin", "keep looking", post) == "Here's what I found."
    assert sent[-1]["tool_choice"] == "none"


def test_a_tool_that_breaks_is_reported_to_the_model_not_blamed_on_it(world, monkeypatch):
    service, _, _ = world
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake")

    def broken(_context):
        raise KeyError("oops")

    original = tools.TOOLS["lab_status"]
    monkeypatch.setitem(tools.TOOLS, "lab_status", (broken, *original[1:]))
    post, sent = script_model([tool_call("lab_status")], text_reply("That check failed."))
    assert brain.answer(service.context, "viewer", "status please", post) == "That check failed."
    assert json.loads(sent[1]["input"][0]["output"]) == {"failed": "KeyError"}


def test_the_briefing_carries_the_lab_and_notes_only_reach_an_admin(world, monkeypatch):
    service, _, lab = world
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake")
    lab.campaigns.append({"id": "forge-aaaaaaaa", "status": "training", "name": "x", "config": {}})
    tools.run(service.context, "admin", "remember_note", {"text": "The owner prefers short runs."})
    post, sent = script_model(text_reply("A."), text_reply("B."))
    brain.answer(service.context, "admin", "hello there", post)
    brain.answer(service.context, "operator", "hello there", post)
    admin = sent[0]["input"][0]
    assert admin["role"] == "developer"
    assert "forge-aaaaaaaa" in admin["content"] and "prefers short runs" in admin["content"]
    assert "forge-aaaaaaaa" in sent[1]["input"][0]["content"]
    assert "prefers short runs" not in json.dumps(sent[1]["input"])


def test_the_private_conversation_is_durable_and_says_what_its_doing(real_lab, monkeypatch):
    client, _, _ = real_lab
    lab = TestClient(client.app, base_url=LOCAL)
    bot = Service("agent-chat", lab=tools.Lab(lab), messages=desk.Transport(), surface="agent-chat")
    monkeypatch.setattr(desk, "service", lambda: bot)
    assert client.get("/hephaestus/messages").json()["messages"] == []
    assert client.post("/hephaestus/messages", json={"text": "tasks"}).status_code == 202
    history = client.get("/hephaestus/messages").json()
    assert [m["sender"] for m in history["messages"]] == ["you", "hephaestus"]
    assert history["thinking"] is False and history["waiting"] == []
    # A new process reads the same conversation.
    desk._service.clear()
    assert len(client.get("/hephaestus/messages").json()["messages"]) == 2


def test_the_private_conversation_says_when_terse_is_away(real_lab, monkeypatch):
    client, _, _ = real_lab

    def away():
        raise ledger.Unavailable("gone")

    monkeypatch.setattr(desk, "history", away)
    reply = client.get("/hephaestus/messages")
    assert reply.status_code == 503 and "Terse" in reply.json()["detail"]
