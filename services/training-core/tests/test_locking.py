from __future__ import annotations

import os
import subprocess
import sys

import pytest

from forge_training.locking import HELD, acquire_model_lock

HOLDER = (
    "import time; from forge_training.locking import acquire_model_lock; "
    "handle = acquire_model_lock('holder'); print('held', flush=True); time.sleep(60)"
)


def test_model_jobs_exclude_each_other_and_release_on_exit(tmp_path, monkeypatch):
    lock = tmp_path / "model-job.lock"
    env = {**os.environ, "FORGE_MODEL_LOCK": str(lock)}
    env.pop(HELD, None)
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLDER], env=env, stdout=subprocess.PIPE, text=True
    )
    monkeypatch.setenv("FORGE_MODEL_LOCK", str(lock))
    monkeypatch.setenv("FORGE_MODEL_LOCK_TIMEOUT", "1")
    monkeypatch.delenv(HELD, raising=False)
    try:
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(TimeoutError, match="model-job lock"):
            acquire_model_lock("contender")
        # Phase subprocesses of a job that holds the lock do not wait on their parent.
        monkeypatch.setenv(HELD, "1")
        assert acquire_model_lock("child phase") is None
    finally:
        holder.kill()
        holder.wait()
    monkeypatch.delenv(HELD)
    # The operating system released the dead holder's lock; no stale lock remains.
    handle = acquire_model_lock("next job")
    assert handle is not None
    assert '"command": "next job"' in lock.read_text()
    handle.close()


def test_lock_can_be_disabled_explicitly(monkeypatch):
    monkeypatch.setenv("FORGE_MODEL_LOCK", "off")
    monkeypatch.delenv(HELD, raising=False)
    assert acquire_model_lock("unlocked") is None
