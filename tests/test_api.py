from concurrent.futures import ThreadPoolExecutor
import json
import time
import hashlib
import httpx
from fastapi.testclient import TestClient
from services import api
from services.core import store

LOCAL = "http://127.0.0.1:8000"


def test_single_gpu_slot_serializes_concurrent_launches(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA", tmp_path)
    output = tmp_path / "artifacts"
    output.mkdir()
    monkeypatch.setattr(store, "ARTIFACTS", output)

    class CampaignAuthority:
        def initialize(self, _config):
            return json.dumps({"status": "initialized"})

    class GovernanceAuthority:
        def initialize(self, _policy):
            return "{}"

    starts = []

    def spawn(cid):
        time.sleep(0.03)
        starts.append(cid)

    monkeypatch.setattr(api, "actor", lambda _: CampaignAuthority())
    monkeypatch.setattr(api, "governance", lambda _: GovernanceAuthority())
    monkeypatch.setattr(api, "spawn", spawn)
    with TestClient(api.app, base_url=LOCAL) as c:

        def create(_):
            return c.post("/campaigns", json={}).status_code

        with ThreadPoolExecutor(6) as pool:
            results = list(pool.map(create, range(6)))
        assert results.count(201) == 1
        assert results.count(409) == 5
        assert len(starts) == 1
        campaign_id = starts[0]
        assert c.post(f"/campaigns/{campaign_id}/cancel").status_code == 200
        assert store.get(campaign_id)["cancel_requested"] is True


def test_unknown_config_rejected_and_cross_origin_mutations_blocked():
    with TestClient(api.app, base_url=LOCAL) as c:
        assert c.post("/campaigns", json={"arbitrary_command": "no"}).status_code == 422
        assert (
            c.post("/campaigns", json={}, headers={"origin": "https://untrusted.example"}).status_code == 403
        )
        assert c.get("/campaigns/no-such-campaign").status_code == 404


def test_download_serves_exact_bytes_and_blocks_tampering_and_path_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA", tmp_path)
    root = tmp_path / "artifacts"
    folder = root / "download"
    folder.mkdir(parents=True)
    monkeypatch.setattr(store, "ARTIFACTS", root)
    content = b"immutable checkpoint mechanics"
    target = folder / "checkpoint.bin"
    target.write_bytes(content)
    store.save(
        {
            "id": "download",
            "status": "completed",
            "artifacts": [
                {"id": "good", "path": "checkpoint.bin", "sha256": hashlib.sha256(content).hexdigest()},
                {"id": "escape", "path": "../../outside", "sha256": "0" * 64},
            ],
        }
    )
    with TestClient(api.app, base_url=LOCAL) as client:
        response = client.get("/campaigns/download/artifacts/good")
        assert response.status_code == 200 and response.content == content
        target.write_bytes(b"changed")
        assert client.get("/campaigns/download/artifacts/good").status_code == 409
        assert client.get("/campaigns/download/artifacts/escape").status_code == 404


def test_dns_rebound_hosts_are_refused_for_reads_and_writes():
    # A rebound page reaches 127.0.0.1 with its own Host and a matching Origin.
    rebound = {"host": "evil.example:8000", "origin": "http://evil.example:8000"}
    with TestClient(api.app, base_url=LOCAL) as c:
        assert c.get("/campaigns", headers=rebound).status_code == 403
        assert c.get("/campaigns", headers={"host": "evil.example:8000"}).status_code == 403
        assert c.post("/campaigns", json={}, headers=rebound).status_code == 403
        assert c.get("/campaigns").status_code == 200
        assert c.get("/campaigns", headers={"origin": "http://127.0.0.1:3000"}).status_code == 200


def test_production_scan_round_trip_and_campaign_reference(tmp_path, monkeypatch):
    from services.core import production

    monkeypatch.setattr(store, "DATA", tmp_path)
    monkeypatch.setattr(production, "sealed_questions", lambda: ["What is the sealed question?"])
    traces = "\n".join(
        [
            '{"id": "a", "prompt": "How many pens?", "output": "12 * 7 = 74\\n#### 74"}',
            '{"id": "b", "prompt": "What is the sealed question?", "output": "no answer"}',
            "{broken",
        ]
    )
    with TestClient(api.app, base_url=LOCAL) as c:
        created = c.post("/production/scans", json={"name": "Week 43", "traces": traces})
        assert created.status_code == 201
        body = created.json()
        assert (body["traces"], body["flagged"], body["malformed"]) == (2, 1, 1)
        assert body["excluded_sealed_overlap"] == ["b"] and "findings" not in body
        listed = c.get("/production/scans").json()
        assert [s["id"] for s in listed] == [body["id"]]
        full = c.get(f"/production/scans/{body['id']}").json()
        assert full["findings"][0]["signals"][0]["evidence"]["computed"] == "84"
        assert c.get("/production/scans/scan-00000000").status_code == 404
        assert c.get("/production/scans/..%2F..%2Fetc").status_code == 404
        missing = c.post("/campaigns", json={"production_scan": "scan-00000000"})
        assert missing.status_code == 404
        assert c.post("/campaigns", json={"production_scan": "../x"}).status_code == 422
        assert c.post("/production/scans", json={"name": "x", "traces": "", "extra": 1}).status_code == 422


def test_health_reports_the_engine_the_relay_bridge_uses(monkeypatch):
    bridges = {
        "local": {"local": True, "transport": "SDK 14.0.0 · Engine 12.0.0", "endpoint": "127.0.0.1:7528"},
        "hosted": {
            "local": False,
            "transport": "SDK 14.0.0 · Relaycast API v1",
            "endpoint": "cast.agentrelay.com",
        },
    }
    for kind, bridge in bridges.items():

        def get(url, timeout, bridge=bridge):
            return httpx.Response(200, json={"ok": True, **bridge} if url.endswith("/health") else {})

        monkeypatch.setattr(api.httpx, "get", get)
        with TestClient(api.app, base_url=LOCAL) as c:
            relay = c.get("/health").json()["services"]["relay"]
        assert relay["connected"] is True
        assert relay["mode"] == ("Local runtime" if kind == "local" else "Hosted Relaycast")
        assert relay["transport"] == bridge["transport"]
        assert relay["endpoint"] == bridge["endpoint"]


def test_the_4b_model_is_refused_where_it_cannot_run(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA", tmp_path)
    four = {"model_id": "Qwen/Qwen3.5-4B", "strategies": ["targeted_sft", "broad_sft"]}
    with TestClient(api.app, base_url=LOCAL) as c:
        monkeypatch.setenv("FORGE_BACKEND", "local")
        refused = c.post("/campaigns", json=four)
        assert refused.status_code == 422 and "needs a cloud GPU" in refused.json()["detail"]
        monkeypatch.setenv("FORGE_BACKEND", "modal")
        for extra in ({"strategies": ["grpo"]}, {"final_backend": "lm_eval_azure"}):
            refused = c.post("/campaigns", json=four | extra)
            assert refused.status_code == 422, extra
