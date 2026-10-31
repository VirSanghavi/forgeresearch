"""SIGKILL the real Terse runtime in every ordering against a mutation and prove exactly-once state.

The runtime runs under the same RuntimeSupervisor that scripts/lab.py uses. A TCP fault proxy
between client and runtime places the kill precisely: before a call, while its request is in
flight, or after the runtime committed but before the client read the response. Each kill
costs about 30 s: after SIGKILL Terse fences the new process until the dead owner's lease lapses.
"""

import json
import os
import re
import socket
import threading
import time
import uuid
from pathlib import Path
import httpx
import pytest
from fastapi.testclient import TestClient
from services.actors.generated import actors
from services.core import chaos
from services.core.actor_client import RETRYABLE, ReconciledTransport, StableClient

ROOT = Path(__file__).resolve().parents[1]
PORT = 7122
URL = f"http://127.0.0.1:{PORT}"


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    folder = tmp_path_factory.mktemp("crash")
    log = (folder / "runtime.log").open("a")
    supervisor = chaos.RuntimeSupervisor(
        [
            str(ROOT / ".venv/bin/durable-actors"),
            *("dev", "--project", str(ROOT / "services/actors"), "--port", str(PORT)),
            *("--data-dir", str(folder / "state"), "--no-watch"),
        ],
        URL,
        folder / "state",
        log,
        env=os.environ | {"DURABLE_ACTORS_CONTROL_PLANE_URL": URL},
        folder=folder,
        max_restarts=50,
    )
    supervisor.start()
    stop = threading.Event()
    errors = []

    def watch():
        while not stop.is_set():
            try:
                supervisor.poll()
            except Exception as error:  # surfaced by the test that caused it
                errors.append(error)
                return
            time.sleep(0.05)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    yield {"folder": folder, "supervisor": supervisor, "errors": errors}
    stop.set()
    watcher.join(timeout=5)
    supervisor.stop()
    log.close()


def transport(lab, url=URL):
    return ReconciledTransport(
        StableClient(url, http=httpx.Client(timeout=10)), retry_seconds=150, journal_folder=lab["folder"]
    )


def identity():
    return "crash-" + uuid.uuid4().hex[:10]


def restarts(lab):
    return [r for r in chaos.read_lines(lab["folder"] / "runtime.jsonl") if r["type"] == "restarted"]


def wait_restarted(lab, count, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        assert not lab["errors"], lab["errors"]
        if len(restarts(lab)) >= count:
            return restarts(lab)[count - 1]
        time.sleep(0.05)
    raise TimeoutError("supervisor did not restart the runtime")


def retries(lab, actor_id):
    return [r for r in chaos.read_lines(lab["folder"] / "retries.jsonl") if r["actor_id"] == actor_id]


class FaultProxy:
    """Forwards bytes to the runtime; one armed trap kills the runtime at a chosen moment."""

    def __init__(self, lab):
        self.lab, self.trap, self.fired, self.generation = lab, None, threading.Event(), 0
        self.server = socket.create_server(("127.0.0.1", 0))
        self.port = self.server.getsockname()[1]
        threading.Thread(target=self.accept, daemon=True).start()

    def arm(self, marker, mode):
        self.fired.clear()
        self.generation += 1
        self.trap = (marker.encode(), mode)

    def kill(self):
        self.lab["killed"] = chaos.kill_runtime(self.lab["folder"])
        self.fired.set()

    def accept(self):
        while True:
            try:
                client, _ = self.server.accept()
            except OSError:
                return
            threading.Thread(target=self.serve, args=(client,), daemon=True).start()

    def serve(self, client):
        try:
            upstream = socket.create_connection(("127.0.0.1", PORT), timeout=30)
        except OSError:
            client.close()
            return
        swallow = threading.Event()

        def down():
            try:
                while data := upstream.recv(65536):
                    if swallow.is_set():
                        # The runtime answered, so the commit is durable. Lose the answer.
                        self.kill()
                        break
                    client.sendall(data)
            except OSError:
                pass
            for s in (client, upstream):
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                s.close()

        threading.Thread(target=down, daemon=True).start()
        seen, generation = b"", self.generation
        try:
            while data := client.recv(65536):
                if generation != self.generation:  # only bytes sent after arming can match
                    seen, generation = b"", self.generation
                seen += data
                trap = self.trap
                if trap and trap[0] in seen:
                    self.trap = None
                    upstream.sendall(data)
                    if trap[1] == "during":
                        self.kill()  # the request reached the runtime; its fate is unknown
                        client.close()
                        return
                    swallow.set()
                    continue
                upstream.sendall(data)
        except OSError:
            pass


@pytest.fixture(scope="module")
def proxy(lab):
    p = FaultProxy(lab)
    yield p
    p.server.close()


def campaign(t, budget=100):
    c = actors.ResearchCampaignActor.get(identity(), t)
    c.initialize(json.dumps({"max_seconds": budget}))
    return c


def assert_budget_consistent(state):
    settled = sum(
        r.get("actual_seconds", 0) for r in state["reservations"].values() if r["status"] == "settled"
    )
    assert state["consumed_seconds"] == pytest.approx(settled)
    reserved = sum(r["seconds"] for r in state["reservations"].values() if r["status"] == "reserved")
    assert reserved + state["consumed_seconds"] <= state["config"]["max_seconds"]


def test_every_failure_the_transport_retries_is_one_that_cannot_double_apply():
    # unavailable: never executed. outcome_unknown: retried under the same request ID, which
    # Terse answers from its commit record (see test_kill_after_commit_before_response).
    assert RETRYABLE == {"outcome_unknown", "unavailable"}


def test_kill_before_mutation(lab):
    t = transport(lab)
    c = campaign(t)
    before = len(restarts(lab))
    kill = chaos.kill_runtime(lab["folder"])
    # The runtime is dead: the call cannot connect, then is fenced, then runs exactly once.
    state = json.loads(c.reserve("exp-1", 40))
    record = wait_restarted(lab, before + 1)
    assert record["old_pid"] == kill["pid"] and record["signal"] == "SIGKILL"
    assert record["pid"] != kill["pid"]
    assert list(state["reservations"]) == ["exp-1"]
    [retry] = retries(lab, c._actor_id)
    assert retry["outcome"] == "completed" and "unavailable" in retry["errors"]
    assert retry["attempts"] > 1


def test_kill_during_mutation(lab, proxy):
    t = transport(lab, f"http://127.0.0.1:{proxy.port}")
    c = campaign(t)
    c.reserve("exp-1", 30)
    c.settle("exp-1", 12.5)
    proxy.arm("exp-during", "during")
    before = len(restarts(lab))
    state = json.loads(c.reserve("exp-during", 50))
    assert proxy.fired.is_set()
    wait_restarted(lab, before + 1)
    # Whether or not the runtime committed before it died, the reservation exists exactly once.
    assert state["reservations"]["exp-during"]["seconds"] == 50
    assert state["consumed_seconds"] == 12.5
    assert_budget_consistent(state)
    with pytest.raises(Exception, match="Compute budget exhausted"):
        c.reserve("exp-over", 40)  # 50 reserved + 12.5 used + 40 > 100: no phantom headroom
    [retry] = retries(lab, c._actor_id)
    assert retry["ambiguous"] and retry["outcome"] == "completed"


def test_kill_after_commit_before_response(lab, proxy):
    t = transport(lab, f"http://127.0.0.1:{proxy.port}")
    c = campaign(t)
    c.freeze_selection("baseline")
    proxy.arm('"claim_final"', "after")
    before = len(restarts(lab))
    # claim_final is NOT idempotent in application code: a second execution returns False.
    # The runtime committed it and the answer was lost; the retry reuses the request ID, and
    # Terse returns the recorded completion instead of running it again.
    assert c.claim_final() is True
    assert proxy.fired.is_set()
    wait_restarted(lab, before + 1)
    [retry] = retries(lab, c._actor_id)
    assert retry["ambiguous"] and retry["attempts"] > 1
    assert json.loads(c.snapshot())["final_claimed"] is True
    # Control: a genuinely new call (new request ID) is executed and refused.
    assert c.claim_final() is False


def test_kill_after_settle_commit_does_not_double_count(lab, proxy):
    t = transport(lab, f"http://127.0.0.1:{proxy.port}")
    c = campaign(t)
    c.reserve("exp-1", 60)
    proxy.arm("exp-1", "after")
    before = len(restarts(lab))
    state = json.loads(c.settle("exp-1", 41.25))
    assert proxy.fired.is_set()
    wait_restarted(lab, before + 1)
    assert state["consumed_seconds"] == 41.25
    assert_budget_consistent(json.loads(c.snapshot()))


def test_two_kills_in_a_row(lab):
    t = transport(lab)
    c = campaign(t)
    c.reserve("exp-1", 20)
    expected = c.snapshot()
    before = len(restarts(lab))
    first = chaos.kill_runtime(lab["folder"])
    wait_restarted(lab, before + 1)
    second = chaos.kill_runtime(lab["folder"])  # kill the replacement while it is still fenced
    assert second["pid"] != first["pid"]
    state = json.loads(c.reserve("exp-2", 30))
    records = restarts(lab)[before:]
    assert [r["old_pid"] for r in records[:2]] == [first["pid"], second["pid"]]
    assert all(r["signal"] == "SIGKILL" for r in records[:2])
    assert {k: v for k, v in state["reservations"].items() if k == "exp-1"} == json.loads(expected)[
        "reservations"
    ]
    assert sorted(state["reservations"]) == ["exp-1", "exp-2"]
    assert_budget_consistent(state)


def test_campaign_sequence_survives_kill_with_identical_state(lab):
    """reserve -> kill -> prove state -> retry -> complete, through the API's incident code."""
    t = transport(lab)
    cid = "forge-" + uuid.uuid4().hex[:8]
    c = actors.ResearchCampaignActor.get(cid, t)
    g = actors.GovernanceActor.get(cid, t)
    r = actors.ArtifactRegistryActor.get(cid, t)
    e = actors.ExperimentActor.get(f"{cid}-exp-001", t)
    c.initialize(json.dumps({"max_seconds": 300}))
    g.initialize(json.dumps({"reward_version": "exact-answer-v1", "max_regression_drop": 0.25}))
    c.transition("baseline", "Measuring the original model")
    c.reserve(f"{cid}-baseline", 60)
    c.settle(f"{cid}-baseline", 21.5)
    r.register(json.dumps({"id": "a1", "kind": "dataset", "sha256": "c" * 64}))
    g.review(
        json.dumps(
            {"id": "pre", "phase": "preflight", "reward_version": "exact-answer-v1", "provenance_valid": True}
        )
    )
    e.initialize(json.dumps({"id": "exp-001", "strategy": "broad_sft"}))
    e.update("approved", "{}")
    c.reserve(f"{cid}-exp-001", 120)  # a lease held across the crash
    c.transition("training", "Training exp-001")

    incidents = chaos.Incidents(lab["folder"], URL)
    incident = incidents.begin(cid, ["exp-001"])
    assert incident["kill"]["signal"] == "SIGKILL"
    end = time.time() + 120
    while incidents.all()[0]["phase"] not in {"restored", "failed"} and time.time() < end:
        time.sleep(0.2)
    done = incidents.all()[0]
    assert done["phase"] == "restored", done.get("error")
    proof = done["proof"]
    assert proof["identical"] == proof["total"] == 4
    assert proof["hash_before"] == proof["hash_after"]
    assert not proof["budget"]["double_counted"] and proof["budget"]["consumed_after"] == 21.5
    assert proof["leases"] == {"before": 2, "after": 2, "held": 1, "unchanged": True}
    assert proof["governance"]["retained"] and proof["governance"]["reviews_after"] == 1
    assert proof["actors"][0]["owner_before"], "Terse ownership record not found"
    for row in proof["actors"]:
        # Terse's own ownership record: same committed state version, a new owner epoch.
        if row["owner_before"]:
            assert row["owner_after"]["state_version"] == row["owner_before"]["state_version"]
            assert row["owner_after"]["epoch"] > row["owner_before"]["epoch"]
    assert done["downtime_seconds"] >= done["process_seconds"] > 0

    # The campaign continues where it was and completes.
    e.update("training", "{}")
    e.metric(json.dumps({"step": 1, "loss": 1.2}))
    e.update("evaluating", "{}")
    e.update("completed", json.dumps({"score": 0.5}))
    state = json.loads(c.settle(f"{cid}-exp-001", 80.0))
    c.freeze_selection("baseline")
    assert c.claim_final() is True
    final = json.loads(c.transition("completed", "Research complete"))
    assert final["consumed_seconds"] == 101.5 == state["consumed_seconds"]
    assert_budget_consistent(final)
    assert [d["status"] for d in final["decisions"]] == ["baseline", "training", "completed"]


def test_chaos_endpoint_requires_flag_and_loopback(monkeypatch):
    from services.api import app

    host = {"host": "127.0.0.1:8000"}
    with TestClient(app, client=("127.0.0.1", 50000)) as local:
        monkeypatch.delenv("FORGE_CHAOS", raising=False)
        assert local.post("/chaos/actors/kill", headers=host).status_code == 403
        assert local.get("/chaos/actors", headers=host).json()["enabled"] is False
    monkeypatch.setenv("FORGE_CHAOS", "1")
    with TestClient(app, client=("10.0.0.5", 50000)) as remote:
        response = remote.post("/chaos/actors/kill", headers=host)
        assert response.status_code == 403 and re.search("this machine", response.json()["detail"])


def test_supervisor_gives_up_on_a_crash_loop(tmp_path):
    log = (tmp_path / "log").open("a")
    s = chaos.RuntimeSupervisor(
        ["false"], "http://127.0.0.1:9", tmp_path, log, folder=tmp_path, max_restarts=2
    )
    s._spawn()
    s.process.wait()
    s.restarts = [{"exited_at": time.time()}, {"exited_at": time.time()}]
    with pytest.raises(RuntimeError, match="not restarting"):
        s.poll()
    [exit] = chaos.read_lines(tmp_path / "runtime.jsonl")
    assert exit["type"] == "exited" and exit["returncode"] == 1
    log.close()
