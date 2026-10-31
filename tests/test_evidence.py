import hashlib
import json

from fastapi.testclient import TestClient

from services import api
from services.core import evidence, store


def snapshot(cid="forge-e1", **changes):
    final = {
        "baseline": {"score": 0.0, "n": 4, "correct": 0},
        "checkpoint": {"score": 0.5, "n": 4, "correct": 2},
        "checkpoint_id": f"{cid}-exp-001",
        "improvement_pp": 50.0,
        "paired_bootstrap_95_pp": [25.0, 75.0],
        "improved_examples": 2,
        "regressed_examples": 0,
        "improvement_claim": "validated",
        "selection_basis": "development only",
        "test_queries_per_model": 1,
        "production": {
            "baseline": {"correct": 0, "n": 3},
            "checkpoint": {"correct": 1, "n": 3},
            "improvement_pp": 33.3,
            "paired_bootstrap_95_pp": [0.0, 66.7],
            "improvement_claim": "uncertain",
        },
    }
    pack = "c9c0fc51b6841b6967a99aea2609dbd7facbcd97d734a835b17704ce3692807b"
    s = {
        "id": cid,
        "name": "Production failures",
        "status": "completed",
        "config": {"model_id": "Qwen/Qwen2.5-0.5B-Instruct"},
        "hardware": {"backend": "modal", "name": "NVIDIA H100 80GB HBM3", "type": "cuda"},
        "elapsed_seconds": 931.9,
        "compute_seconds": 809.2,
        "experiments": [{"id": "exp-001"}],
        "sealed_ids": ["a", "b", "c", "d"],
        "actor": {"final_claimed": True, "selection_frozen": True, "version": 11, "decisions": [{}, {}, {}]},
        "final": final,
        "reviews": [
            {
                "decision": "approved",
                "reason_codes": [],
                "evidence": {"sealed_data_access": False, "task_pack_sha256": pack},
            },
            {
                "decision": "rejected",
                "reason_codes": ["REGRESSION_THRESHOLD_EXCEEDED"],
                "evidence": {"sealed_data_access": False, "task_pack_sha256": pack},
            },
        ],
        "agent_calls": [
            {
                "role": "director",
                "model": "gpt-6-luna",
                "calls": 5,
                "usage": {"input_tokens": 16100, "output_tokens": 1638},
                "subagents": ["development-evidence", "experiment-design"],
            },
            {
                "role": "director",
                "model": "gpt-6-luna",
                "calls": 1,
                "usage": {"input_tokens": 100, "output_tokens": 10},
                "subagents": ["development-evidence"],
            },
            {"role": "data", "model": "gpt-6-luna", "calls": 1, "usage": {}, "subagents": []},
        ],
        "artifacts": [],
    }
    s.update(changes)
    return s


def with_artifact(tmp_path, monkeypatch, s, content=b"weights", registered=None):
    monkeypatch.setattr(store, "ARTIFACTS", tmp_path)
    folder = tmp_path / s["id"] / "exp-001"
    folder.mkdir(parents=True)
    (folder / "adapter.safetensors").write_bytes(content)
    s["artifacts"] = [
        {
            "id": "a1",
            "kind": "checkpoint",
            "path": "exp-001/adapter.safetensors",
            "sha256": registered or hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        }
    ]
    return s


EVENTS = [
    {"kind": "message_pending", "message": {"id": "m1"}},
    {"kind": "message", "message_id": "m1", "relay_id": "r1", "receipt": {"delivered": True}},
    {"kind": "message_pending", "message": {"id": "m2"}},
    {
        "kind": "message",
        "relay_id": "r2",
        "receipt": {"delivered": True, "recipient_readback_verified": True},
    },
    {"kind": "stage", "text": "Loading the saved adapter into a fresh model for evaluation"},
]


def states(summary):
    return {c["id"]: c["state"] for c in summary["checks"]}


def test_complete_campaign_passes_every_recorded_check(tmp_path, monkeypatch):
    s = with_artifact(tmp_path, monkeypatch, snapshot())
    summary = evidence.summarize(s, EVENTS)
    assert set(states(summary).values()) == {"pass"}
    assert summary["sealed"]["claim"] == "validated"
    assert summary["sealed"]["interval_pp"] == [25.0, 75.0]
    assert summary["sealed"]["kept_original"] is False
    # A production claim is reported as recorded, even when it differs from the sealed claim.
    assert summary["production"]["claim"] == "uncertain"
    governance = next(c for c in summary["checks"] if c["id"] == "governance")
    assert "1 rejected" in governance["detail"] and "REGRESSION_THRESHOLD_EXCEEDED" in governance["detail"]


def test_tampered_artifact_fails_and_missing_file_is_partial(tmp_path, monkeypatch):
    tampered = with_artifact(tmp_path, monkeypatch, snapshot("forge-t"), registered="0" * 64)
    assert states(evidence.summarize(tampered, EVENTS))["sha256"] == "fail"

    gone = with_artifact(tmp_path, monkeypatch, snapshot("forge-g"))
    (tmp_path / "forge-g" / "exp-001" / "adapter.safetensors").unlink()
    summary = evidence.summarize(gone, EVENTS)
    assert states(summary)["sha256"] == "partial"
    assert summary["artifacts"] == {"count": 1, "verified": 0, "mismatched": 0, "missing": 1, "bytes": 7}


def test_artifact_path_cannot_escape_the_campaign_folder(tmp_path, monkeypatch):
    s = with_artifact(tmp_path, monkeypatch, snapshot())
    (tmp_path / "secret").write_bytes(b"x")
    s["artifacts"][0]["path"] = "../secret"
    s["artifacts"][0]["sha256"] = hashlib.sha256(b"x").hexdigest()
    assert evidence.artifacts(s)["missing"] == 1


def test_missing_facts_read_as_not_recorded_never_as_passed(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "ARTIFACTS", tmp_path)
    s = snapshot(actor={}, reviews=[], agent_calls=None)
    s["final"] = {k: v for k, v in s["final"].items() if k != "selection_basis"}
    result = states(evidence.summarize(s, []))
    assert result["sealed_once"] == "not_recorded"
    assert result["selection_frozen"] == "not_recorded"
    assert result["governance"] == "not_recorded"
    assert result["no_sealed_access"] == "not_recorded"
    assert result["pack_pinned"] == "not_recorded"
    assert evidence.summarize(s, [])["agents"] is None


def test_unfinished_campaign_has_no_sealed_result(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "ARTIFACTS", tmp_path)
    summary = evidence.summarize(snapshot(status="cancelled", final={}), [])
    assert summary["sealed"] is None and summary["production"] is None
    assert states(summary)["sealed_once"] == "not_applicable"
    assert states(summary)["reload"] == "not_applicable"


def test_kept_original_and_disclosed_probe(tmp_path, monkeypatch):
    s = with_artifact(tmp_path, monkeypatch, snapshot())
    s["final"].update(checkpoint_id="baseline", improvement_claim="none", improvement_pp=0.0)
    s["reviews"].append(
        {"decision": "rejected", "evidence": {"sealed_data_access": True, "controlled_probe": True}}
    )
    summary = evidence.summarize(s, EVENTS)
    assert summary["sealed"]["kept_original"] is True
    sealed = next(c for c in summary["checks"] if c["id"] == "no_sealed_access")
    # The probe is a disclosed adversarial fixture: it is reported, not counted as a breach.
    assert sealed["state"] == "pass" and "1 disclosed probe" in sealed["detail"]
    frozen = next(c for c in summary["checks"] if c["id"] == "selection_frozen")
    assert "the original model" in frozen["detail"]


def test_real_sealed_access_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "ARTIFACTS", tmp_path)
    s = snapshot()
    s["reviews"][0]["evidence"]["sealed_data_access"] = True
    assert states(evidence.summarize(s, EVENTS))["no_sealed_access"] == "fail"


def test_relay_counts_only_confirmed_receipts_in_both_event_shapes():
    unconfirmed = EVENTS + [{"kind": "message_pending", "message": {"id": "m3"}}]
    assert evidence.relay(unconfirmed) == {"sent": 3, "delivered": 2, "readback_verified": 1}


def test_agent_calls_sum_by_role():
    agents = evidence.agents(snapshot())
    assert agents["calls"] == 7
    assert agents["input_tokens"] == 16200 and agents["output_tokens"] == 1648
    assert agents["subagent_runs"] == 3
    assert agents["subagent_kinds"] == ["development-evidence", "experiment-design"]
    assert [r["role"] for r in agents["by_role"]] == ["director", "data"]


def test_evidence_route_reads_stored_campaigns(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA", tmp_path)
    monkeypatch.setattr(store, "ARTIFACTS", tmp_path / "artifacts")
    (tmp_path / "artifacts").mkdir()
    s = snapshot()
    with store.db() as c:
        c.execute("INSERT INTO campaigns VALUES(?,?)", (s["id"], json.dumps(s)))
    body = TestClient(api.app, base_url="http://127.0.0.1:8000").get("/evidence").json()
    assert [c["id"] for c in body] == ["forge-e1"]
    assert body[0]["sealed"]["claim"] == "validated"


def test_vm_runs_and_workbenches_are_summed_only_when_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "ARTIFACTS", tmp_path)
    s = snapshot()
    assert evidence.summarize(s, [])["vms"] is None
    assert evidence.agents(s)["vm_runs"] == 0
    s["agent_calls"][0]["vm_runs"] = 3
    s["workbenches"] = [
        {"id": "vm-1", "role": "director", "runs": 3, "seconds": 4.5, "deleted": True},
        {"id": "vm-2", "role": "data", "runs": 1, "seconds": 1.0, "deleted": False, "error": "Timeout"},
    ]
    summary = evidence.summarize(s, [])
    assert summary["agents"]["vm_runs"] == 3
    assert summary["agents"]["by_role"][0]["vm_runs"] == 3
    assert summary["vms"] == {"count": 2, "runs": 4, "seconds": 5.5, "deleted": 1}
