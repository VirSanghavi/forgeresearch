"""Host-wide exclusion for model-loading commands: one model job at a time per machine.

Two concurrent model jobs on a small machine thrash memory and swap. The lock is an
advisory ``flock`` that the operating system releases when the holding process exits,
so a crashed job never leaves a stale lock. Child phases of a held job inherit it.
"""

from __future__ import annotations

import fcntl
import json
import os
import sys
import time
from pathlib import Path

HELD = "FORGE_MODEL_LOCK_HELD"


def lock_path() -> Path | None:
    configured = os.environ.get("FORGE_MODEL_LOCK")
    if configured == "off":
        return None
    return Path(configured) if configured else Path.home() / ".cache/forge/model-job.lock"


def acquire_model_lock(command: str, wait: bool = True):
    """Blocks until no other model job holds the lock; returns the handle to keep open.

    With wait=False a held lock raises RuntimeError at once instead of queueing.
    """
    path = lock_path()
    if path is None or os.environ.get(HELD) == "1":
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    deadline = time.monotonic() + float(os.environ.get("FORGE_MODEL_LOCK_TIMEOUT", "7200"))
    waiting = False
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if not wait:
                handle.close()
                raise RuntimeError("Another model job is running on this machine") from None
            if time.monotonic() >= deadline:
                handle.close()
                raise TimeoutError(f"Timed out waiting for the model-job lock {path}") from None
            if not waiting:
                print(f"Waiting for the model-job lock {path}", file=sys.stderr, flush=True)
                waiting = True
            time.sleep(0.5)
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({"pid": os.getpid(), "command": command, "acquiredAt": time.time()}))
    handle.flush()
    # Phase subprocesses of this job must not wait on their own parent.
    os.environ[HELD] = "1"
    return handle
