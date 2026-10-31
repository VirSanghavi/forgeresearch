"""Behavioral checks against the real Terse native runtime, not actor mocks."""

import json
import os
import signal
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import httpx
import pytest
from durable_actors.client import Client, ActorInvocationError
from services.actors.generated import actors

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    folder = tmp_path_factory.mktemp("terse")
    port = 7121
    url = f"http://127.0.0.1:{port}"
    log = (folder / "runtime.log").open("w")
    commands = [
        str(ROOT / ".venv/bin/durable-actors"),
        "dev",
        "--project",
        str(ROOT / "services/actors"),
        "--port",
        str(port),
        "--data-dir",
        str(folder / "state"),
    ]
    env = os.environ | {"DURABLE_ACTORS_CONTROL_PLANE_URL": url}
    process = subprocess.Popen(commands, env=env, stdout=log, stderr=log, start_new_session=True)
    client = Client(url, http=httpx.Client(timeout=15))
    for _ in range(120):
        try:
            actors.ResearchCampaignActor.get("health", client).snapshot()
            break
        except Exception:
            if process.poll() is not None:
                raise RuntimeError((folder / "runtime.log").read_text()) from None
            time.sleep(0.25)
    else:
        raise TimeoutError("Terse test runtime did not start")
    yield client, process, commands, env, log
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=15)
    client.close()
    log.close()


def identity():
    return "test-" + uuid.uuid4().hex[:12]


def campaign(runtime):
    c = actors.ResearchCampaignActor.get(identity(), runtime[0])
    c.initialize(json.dumps({"max_seconds": 100}))
    return c


def test_concurrent_reservations_cannot_overspend(runtime):
    c = campaign(runtime)

    def reserve(i):
        try:
            c.reserve(f"experiment-{i}", 60)
            return True
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=8) as p:
        outcomes = list(p.map(reserve, range(8)))
    assert sum(outcomes) == 1
    s = json.loads(c.snapshot())
    assert sum(r["seconds"] for r in s["reservations"].values()) == 60


def test_budget_duplicate_missing_and_conflicting_orderings(runtime):
    c = campaign(runtime)
    # A without B: no spend occurs merely because a reservation exists.
    c.reserve("a", 40)
    c.reserve("a", 40)
    assert json.loads(c.snapshot())["consumed_seconds"] == 0
    # B without A: settlement cannot create a phantom reservation.
    with pytest.raises(ActorInvocationError):
        c.settle("missing", 20)
    # A then B and repeated B are idempotent; conflicting B fails.
    c.settle("a", 20)
    c.settle("a", 20)
    assert json.loads(c.snapshot())["consumed_seconds"] == 20
    with pytest.raises(ActorInvocationError):
        c.settle("a", 21)


def test_campaign_config_and_final_selection_are_immutable(runtime):
    c = campaign(runtime)
    with pytest.raises(ActorInvocationError):
        c.initialize(json.dumps({"max_seconds": 999}))
    assert c.claim_final() is False
    c.freeze_selection("baseline")
    with pytest.raises(ActorInvocationError):
        c.freeze_selection("different")
    assert c.claim_final() is True
    assert c.claim_final() is False
    with pytest.raises(ActorInvocationError):
        c.reserve("late", 1)


def test_environment_isolated_idempotent_hidden_and_versioned(runtime):
    c = runtime[0]
    eid = identity()
    w = actors.RLWorldActor.get(eid, c)
    initial = json.loads(w.reset(eid, "What is 7 times 6?", "42", 42))
    assert "_answer" not in initial and "hidden_answer" not in initial
    action = {"episode_id": eid, "action_id": "answer", "expected_version": 0, "answer": "42"}
    result = json.loads(w.step(json.dumps(action)))
    assert result["reward"] == 1 and result["terminated"]
    assert json.loads(w.step(json.dumps(action))) == result
    assert len(result["actions"]) == 1
    with pytest.raises(ActorInvocationError):
        w.step(json.dumps(action | {"action_id": "stale"}))
    with pytest.raises(ActorInvocationError):
        w.step(json.dumps(action | {"answer": "43"}))
    other = actors.RLWorldActor.get(identity(), c)
    assert json.loads(other.reset("other", "Independent task", "2", 0))["version"] == 0
    assert json.loads(w.observe())["version"] == 1


def evidence(key, weight_id, eval_id):
    return {
        "id": key,
        "phase": "checkpoint",
        "reward_version": "exact-answer-v1",
        "sealed_data_access": False,
        "provenance_valid": True,
        "weights_changed": True,
        "independent_reload": True,
        "artifact_verified": True,
        "sha256": "a" * 64,
        "regression_drop": 0,
        "checkpoint_artifact_id": weight_id,
        "evaluation_artifact_id": eval_id,
        "evaluation_sha256": "b" * 64,
    }


@pytest.mark.parametrize(
    "ordering", ["artifact_then_review", "review_then_artifact", "concurrent", "artifact_only", "review_only"]
)
def test_promotion_requires_both_evidence_and_approval_in_every_order(runtime, ordering):
    cid = identity()
    g = actors.GovernanceActor.get(cid, runtime[0])
    r = actors.ArtifactRegistryActor.get(cid, runtime[0])
    g.initialize(json.dumps({"reward_version": "exact-answer-v1", "max_regression_drop": 0.25}))
    e = evidence("checkpoint", "weights", "eval")

    def artifacts():
        r.register(json.dumps({"id": "weights", "kind": "checkpoint", "sha256": "a" * 64}))
        r.register(json.dumps({"id": "eval", "kind": "evaluation", "sha256": "b" * 64}))

    def review():
        g.review(json.dumps(e))

    if ordering == "artifact_then_review":
        artifacts()
        review()
    elif ordering == "review_then_artifact":
        review()
        with pytest.raises(ActorInvocationError):
            g.promote("checkpoint", "a" * 64)
        artifacts()
    elif ordering == "concurrent":
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(lambda f: f(), [artifacts, review]))
    elif ordering == "artifact_only":
        artifacts()
    else:
        review()
    if ordering in {"artifact_only", "review_only"}:
        with pytest.raises(ActorInvocationError):
            g.promote("checkpoint", "a" * 64)
    else:
        assert g.promote("checkpoint", "a" * 64) == "checkpoint"


def test_rejected_checkpoint_cannot_be_reapproved_or_promoted(runtime):
    g = actors.GovernanceActor.get(identity(), runtime[0])
    g.initialize(json.dumps({"reward_version": "exact-answer-v1", "max_regression_drop": 0.25}))
    e = evidence("bad", "w", "e") | {"reward_version": "always-one", "sealed_data_access": True}
    assert json.loads(g.review(json.dumps(e)))["decision"] == "rejected"
    with pytest.raises(ActorInvocationError):
        g.promote("bad", "a" * 64)
    with pytest.raises(ActorInvocationError):
        g.review(json.dumps(e | {"reward_version": "exact-answer-v1", "sealed_data_access": False}))


def test_v1_policies_review_as_before_and_v2_policies_pin_the_task_pack(runtime):
    v1 = actors.GovernanceActor.get(identity(), runtime[0])
    v1.initialize(json.dumps({"reward_version": "exact-answer-v1", "max_regression_drop": 0.25}))
    old = json.loads(v1.review(json.dumps(evidence("old", "w", "e"))))
    assert old["decision"] == "approved" and old["policy_version"] == "forge-policy-v1"

    v2 = actors.GovernanceActor.get(identity(), runtime[0])
    policy = {
        "reward_version": "exact-answer-v1",
        "max_regression_drop": 0.25,
        "task_pack_sha256": "c" * 64,
        "policy_version": "forge-policy-v2",
    }
    v2.initialize(json.dumps(policy))
    same = json.loads(v2.review(json.dumps(evidence("same", "w", "e") | {"task_pack_sha256": "c" * 64})))
    assert same["decision"] == "approved" and same["policy_version"] == "forge-policy-v2"
    swapped = json.loads(v2.review(json.dumps(evidence("swap", "w", "e") | {"task_pack_sha256": "d" * 64})))
    assert swapped["reason_codes"] == ["TASK_PACK_MODIFIED"]


def test_synthetic_training_data_needs_verified_provenance_terms_and_separation(runtime):
    g = actors.GovernanceActor.get(identity(), runtime[0])
    g.initialize(json.dumps({"reward_version": "exact-answer-v1", "max_regression_drop": 0.25}))
    good = {
        "provenance_verified": True,
        "teacher_terms_present": True,
        "checker_matches_pack": True,
        "separation_verified": True,
        "verification": "reference",
    }
    ok = json.loads(g.review(json.dumps(evidence("syn-ok", "w", "e") | {"synthetic": good})))
    assert ok["decision"] == "approved"
    bad = {key: False for key in good} | {"verification": "format_only"}
    rejected = json.loads(g.review(json.dumps(evidence("syn-bad", "w", "e") | {"synthetic": bad})))
    assert rejected["reason_codes"] == [
        "SYNTHETIC_PROVENANCE_MISSING",
        "TEACHER_TERMS_MISSING",
        "SYNTHETIC_CHECKER_MISMATCH",
        "SYNTHETIC_SEPARATION_FAILED",
        "SYNTHETIC_UNVERIFIED",
    ]


def test_zz_committed_state_survives_native_sigkill(runtime):
    client, process, commands, env, log = runtime
    cid = identity()
    c = actors.ResearchCampaignActor.get(cid, client)
    c.initialize(json.dumps({"max_seconds": 100}))
    c.reserve("survive", 30)
    expected = c.snapshot()
    os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=10)
    replacement = subprocess.Popen(commands, env=env, stdout=log, stderr=log, start_new_session=True)
    try:
        with Client("http://127.0.0.1:7121", http=httpx.Client(timeout=5)) as fresh:
            for _ in range(120):
                try:
                    restored = actors.ResearchCampaignActor.get(cid, fresh).snapshot()
                    break
                except ActorInvocationError:
                    time.sleep(0.25)
            else:
                raise TimeoutError("Actor did not recover after native restart")
            assert restored == expected
    finally:
        if replacement.poll() is None:
            os.killpg(replacement.pid, signal.SIGTERM)
            replacement.wait(timeout=15)
