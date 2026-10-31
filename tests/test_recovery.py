"""Exercise lost workers and simultaneous read-model writers at their real boundaries."""

import multiprocessing
import subprocess
import sys
from pathlib import Path

from services.core import lifecycle, store


def increment(data, artifacts):
    store.DATA = Path(data)
    store.ARTIFACTS = Path(artifacts)
    for _ in range(12):
        store.patch("parallel", lambda s: s.update(count=s["count"] + 1))


def isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA", tmp_path)
    output = tmp_path / "artifacts"
    output.mkdir()
    monkeypatch.setattr(store, "ARTIFACTS", output)


def test_cross_process_updates_cannot_lose_reviews_or_progress(tmp_path, monkeypatch):
    isolate(tmp_path, monkeypatch)
    store.save({"id": "parallel", "count": 0, "reviews": ["approved"]})
    context = multiprocessing.get_context("spawn")
    writers = [
        context.Process(target=increment, args=(str(tmp_path), str(store.ARTIFACTS))) for _ in range(4)
    ]
    for writer in writers:
        writer.start()
    for writer in writers:
        writer.join(timeout=20)
        assert writer.exitcode == 0
    assert store.get("parallel")["count"] == 48
    assert store.get("parallel")["reviews"] == ["approved"]


def test_dead_worker_is_interrupted_once_with_evidence_preserved(tmp_path, monkeypatch):
    isolate(tmp_path, monkeypatch)
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait(timeout=5)
    store.save(
        {
            "id": "lost",
            "status": "training",
            "worker_pid": process.pid,
            "progress": {"done": 8},
            "recording": {"status": "recording"},
            "experiments": [{"id": "saved", "status": "completed"}, {"id": "active", "status": "training"}],
            "artifacts": [{"sha256": "preserved"}],
        }
    )
    lifecycle.reconcile()
    lifecycle.reconcile()
    recovered = store.get("lost")
    assert recovered["status"] == "interrupted"
    assert recovered["progress"] is None
    assert recovered["recording"]["status"] == "failed"
    assert [e["status"] for e in recovered["experiments"]] == ["completed", "interrupted"]
    assert recovered["artifacts"] == [{"sha256": "preserved"}]
    assert len(store.events("lost")) == 1


def test_a_long_command_line_is_still_recognized():
    # Linux ps truncates piped output at 80 columns unless asked not to; the marker sits past it.
    padding = "x" * 200
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)", padding, "services.core.runner", "forge-long"]
    )
    try:
        assert lifecycle.worker_alive(child.pid, "forge-long")
    finally:
        child.kill()
        child.wait()
    assert not lifecycle.worker_alive(child.pid, "forge-long")


def test_unrelated_live_pid_cannot_keep_a_dead_campaign_running():
    import os

    assert not lifecycle.worker_alive(os.getpid(), "not-this-process")


def test_completion_before_reconciliation_does_not_emit_a_false_interruption(tmp_path, monkeypatch):
    isolate(tmp_path, monkeypatch)
    store.save({"id": "finished", "status": "evaluating", "worker_pid": 1})

    def completion_wins(_pid, _campaign_id):
        store.patch("finished", lambda s: s.update(status="completed"))
        return False

    monkeypatch.setattr(lifecycle, "worker_alive", completion_wins)
    lifecycle.reconcile()
    assert store.get("finished")["status"] == "completed"
    assert store.events("finished") == []


def test_killed_recorder_is_detected_even_after_training_completed(tmp_path, monkeypatch):
    isolate(tmp_path, monkeypatch)
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait(timeout=5)
    store.save(
        {"id": "camera", "status": "completed", "recording": {"status": "recording", "pid": process.pid}}
    )
    lifecycle.reconcile()
    assert store.get("camera")["status"] == "completed"
    assert store.get("camera")["recording"]["status"] == "failed"
    assert store.events("camera")[0]["kind"] == "recording_error"
