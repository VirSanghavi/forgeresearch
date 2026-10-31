"""Standalone stdlib worker, also staged verbatim into a SuperServe sandbox."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


def execute(command, workspace, env, timeout, output_limit, cancelled=lambda: False, on_start=None):
    workspace = Path(workspace)
    control = workspace / ".forge"
    control.mkdir(exist_ok=True)
    started = time.monotonic()
    status = "failed"
    with (control / "stdout.log").open("wb") as stdout, (control / "stderr.log").open("wb") as stderr:
        process = subprocess.Popen(
            command, cwd=workspace, env=env, stdout=stdout, stderr=stderr, start_new_session=True
        )
        try:
            if on_start:
                on_start(process.pid)
            while True:
                size = os.fstat(stdout.fileno()).st_size + os.fstat(stderr.fileno()).st_size
                if cancelled():
                    status = "cancelled"
                    break
                if size > output_limit:
                    status = "output_limit"
                    break
                if time.monotonic() - started >= timeout:
                    status = "timed_out"
                    break
                if process.poll() is not None:
                    status = "succeeded" if process.returncode == 0 else "failed"
                    break
                time.sleep(0.025)
        finally:
            # Descendants are part of the job even if their parent already exited.
            process.poll()  # Reap an exited leader before killpg (macOS rejects zombie groups).
            if process.returncode is None and status in {"cancelled", "timed_out", "output_limit"}:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=2)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                if process.poll() is None:
                    raise
            process.wait()
            # Preserve bounded diagnostics even when the producer overshot between polls.
            kept_stdout = min(os.fstat(stdout.fileno()).st_size, output_limit)
            stdout.truncate(kept_stdout)
            stderr.truncate(min(os.fstat(stderr.fileno()).st_size, output_limit - kept_stdout))
    return {"status": status, "exit_code": process.returncode, "seconds": time.monotonic() - started}


def main():
    path = Path(sys.argv[1])
    job = json.loads(path.read_text())
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    result = execute(
        job["command"],
        path.parent.parent,
        os.environ.copy(),
        job["timeout_seconds"],
        job["max_output_bytes"],
        stop.is_set,
    )
    temporary = path.parent / "result.tmp"
    temporary.write_text(json.dumps(result))
    temporary.replace(path.parent / "result.json")


if __name__ == "__main__":
    main()
