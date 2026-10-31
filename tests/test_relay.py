import os
import uuid
import httpx
import pytest
from pydantic import ValidationError
from services.core import engine, store
from services.core.contracts import Message

BRIDGE = os.getenv("FORGE_RELAY_BRIDGE", "http://127.0.0.1:7529")
live = pytest.mark.skipif(
    os.getenv("FORGE_LIVE_RELAY_TEST") != "1",
    reason="Requires a running Relay bridge (FORGE_RELAY_BRIDGE, local or hosted engine)",
)


def message(**overrides):
    m = {
        "id": "test-" + uuid.uuid4().hex,
        "campaign_id": "transport-test",
        "sender": "director",
        "recipient": "trainer",
        "category": "TRAINING_ASSIGNED",
        "text": "Verify exact recipient bytes before authorizing work.",
    }
    m.update(overrides)
    return m


def test_message_routes_are_closed():
    assert Message(**message()).route == "dm"
    assert Message(**message(route="room", reply_to="241700929207930880")).reply_to
    with pytest.raises(ValidationError):
        Message(**message(route="broadcast"))
    with pytest.raises(ValidationError):
        Message(**message(reply_to="../escape"))


@pytest.fixture
def journal(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA", tmp_path)
    monkeypatch.setattr(store, "ARTIFACTS", tmp_path / "artifacts")
    (tmp_path / "artifacts").mkdir()
    return tmp_path


def test_send_records_route_thread_and_receipt(journal, monkeypatch):
    monkeypatch.setenv("FORGE_RELAY_BRIDGE_TOKEN", "per-launch-secret")
    posted = []

    def post(url, json, timeout, headers=None):
        posted.append((url, json, headers))
        return httpx.Response(
            200,
            json={
                "delivered": True,
                "message_id": "r2",
                "route": "thread",
                "read_by": [{"name": "director"}],
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(engine.httpx, "post", post)
    receipt = engine.send(
        "c1", "governance", "trainer", "CHECKPOINT_VERDICT", "Approved.", "e1", "room", "r1"
    )
    assert receipt["message_id"] == "r2"
    assert posted[0][1]["route"] == "room" and posted[0][1]["reply_to"] == "r1"
    assert posted[0][2] == {"Authorization": "Bearer per-launch-secret"}
    events = store.events("c1")
    assert [e["kind"] for e in events] == ["message_pending", "message"]
    delivered = events[1]
    assert delivered["route"] == "thread"
    assert delivered["thread_parent_id"] == "r1"
    assert delivered["message_id"] == events[0]["message"]["id"]


def test_unconfirmed_delivery_pauses_the_campaign(journal, monkeypatch):
    monkeypatch.setattr(
        engine.httpx,
        "post",
        lambda url, json, timeout, headers=None: httpx.Response(
            200, json={"delivered": False}, request=httpx.Request("POST", url)
        ),
    )
    with pytest.raises(RuntimeError, match="paused before downstream work"):
        engine.send("c2", "director", "trainer", "TRAINING_ASSIGNED", "Go.")
    assert [e["kind"] for e in store.events("c2")] == ["message_pending"]


def test_reaction_is_recorded_and_never_blocks(journal, monkeypatch):
    def post(url, json, timeout, headers=None):
        assert url.endswith("/react") and json["emoji"] == "white_check_mark"
        return httpx.Response(
            200,
            json={"reactions": [{"emoji": "white_check_mark", "count": 1, "agents": ["governance"]}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(engine.httpx, "post", post)
    assert engine.react("c3", "governance", "r1", "white_check_mark")
    event = store.events("c3")[-1]
    assert event["kind"] == "reaction" and event["relay_id"] == "r1"

    def down(url, json, timeout, headers=None):
        raise httpx.ConnectError("bridge down")

    monkeypatch.setattr(engine.httpx, "post", down)
    assert engine.react("c3", "governance", "r1", "x") is None
    assert store.events("c3")[-1]["kind"] == "relay_warning"


@live
def test_real_relay_dm_room_thread_and_reaction():
    dm = message()
    first = httpx.post(BRIDGE + "/message", json=dm, headers=engine.bridge_headers(), timeout=30).json()
    second = httpx.post(BRIDGE + "/message", json=dm, headers=engine.bridge_headers(), timeout=30).json()
    assert first["message_id"] == second["message_id"]
    assert first["route"] == "dm" and first["recipient_readback_verified"] is True
    assert [r["name"] for r in first["read_by"]] == ["trainer"]
    assert first["deliveries"][0]["source"] == "inbox"

    post = httpx.post(
        BRIDGE + "/message",
        json=message(sender="trainer", recipient="governance", category="CHECKPOINT_SAVED", route="room"),
        headers=engine.bridge_headers(),
        timeout=30,
    ).json()
    assert {r["name"] for r in post["read_by"]} == {"director", "data", "evaluator", "governance"}

    reply = httpx.post(
        BRIDGE + "/message",
        json=message(
            sender="governance",
            recipient="trainer",
            category="CHECKPOINT_VERDICT",
            route="room",
            reply_to=post["message_id"],
        ),
        headers=engine.bridge_headers(),
        timeout=30,
    ).json()
    assert reply["route"] == "thread" and reply["thread_parent_id"] == post["message_id"]

    reacted = httpx.post(
        BRIDGE + "/react",
        json={
            "campaign_id": "transport-test",
            "sender": "governance",
            "message_id": post["message_id"],
            "emoji": "white_check_mark",
        },
        headers=engine.bridge_headers(),
        timeout=30,
    ).json()
    assert any(r["emoji"] == "white_check_mark" and "governance" in r["agents"] for r in reacted["reactions"])
