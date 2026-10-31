import json
from types import SimpleNamespace
import pytest
from services.core import azure_final

WORKER = {"resource_group": "forge-eval-338fedca", "name": "forge-evaluator"}
READY = {"power": [{"code": "PowerState/running"}], "agent": [{"code": "ProvisioningState/succeeded"}]}


def fake_cli(monkeypatch, states):
    calls = []
    iterator = iter(states)

    def run(command, **_kwargs):
        calls.append(command[2])
        value = next(iterator) if command[2] == "get-instance-view" else None
        return SimpleNamespace(returncode=0, stdout=json.dumps(value) if value else "")

    monkeypatch.setattr(azure_final.subprocess, "run", run)
    monkeypatch.setattr(azure_final.time, "sleep", lambda _: None)
    return calls


def test_running_evaluator_is_reused_with_a_new_shutdown_bound(monkeypatch):
    calls = fake_cli(monkeypatch, [READY])
    azure_final.ensure_worker(WORKER, lambda: False)
    assert calls == ["auto-shutdown", "get-instance-view"]


def test_deallocated_worker_starts_then_waits_for_its_agent(monkeypatch):
    calls = fake_cli(
        monkeypatch,
        [
            {"power": [{"code": "PowerState/deallocated"}], "agent": None},
            {"power": [{"code": "PowerState/running"}], "agent": []},
            READY,
        ],
    )
    azure_final.ensure_worker(WORKER, lambda: False)
    assert calls.count("start") == 1
    assert calls[-1] == "get-instance-view"


def test_no_other_resource_can_be_started(monkeypatch):
    calls = fake_cli(monkeypatch, [])
    with pytest.raises(ValueError):
        azure_final.ensure_worker(WORKER | {"resource_group": "someone-elses-project"}, lambda: False)
    assert not calls


def test_cancellation_stops_the_scoped_worker(monkeypatch):
    calls = fake_cli(monkeypatch, [])
    with pytest.raises(InterruptedError):
        azure_final.ensure_worker(WORKER, lambda: True)
    assert calls[-1] == "deallocate"
