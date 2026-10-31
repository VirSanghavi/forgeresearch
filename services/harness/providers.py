"""Execution adapters. Each remote attempt owns and cleans up its own sandbox."""

import json
import math
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from services.harness import worker
from services.harness.checks import evaluate_checks


def task_environment(task):
    missing = [name for name in task.env if name not in os.environ]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    return {name: os.environ[name] for name in task.env}


class LocalProvider:
    def run(self, task, workspace, timeout, cancelled, record):
        command = [sys.executable if arg == "{python}" else arg for arg in task.command]
        # Host credentials are not implicitly inherited. The command still has the host user's
        # filesystem privileges: use SuperServe for isolation from the host.
        env = {"PATH": os.environ.get("PATH", os.defpath), "PYTHONUNBUFFERED": "1"}
        env.update(task_environment(task))
        control = workspace / ".forge"
        job = {"command": command, "timeout_seconds": timeout, "max_output_bytes": task.max_output_bytes}
        (control / "job.json").write_text(json.dumps(job))
        # An independent supervisor keeps enforcing the deadline if the scheduler is killed.
        process = subprocess.Popen(
            [sys.executable, str(Path(worker.__file__).resolve()), str(control / "job.json")],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            record({"pid": process.pid})
            signalled = False
            while process.poll() is None:
                if cancelled() and not signalled:
                    process.send_signal(signal.SIGTERM)
                    signalled = True
                time.sleep(0.025)
            if process.returncode:
                if signalled:
                    return {"status": "cancelled", "exit_code": process.returncode}
                raise RuntimeError("Local supervisor failed before recording a result")
            return json.loads((control / "result.json").read_text())
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                process.wait(timeout=5)


class SuperServeProvider:
    def __init__(self, sdk=None):
        self.sdk = sdk

    def _sdk(self):
        if self.sdk is None:
            try:
                from superserve import Sandbox
            except ImportError as error:
                raise RuntimeError("Install the superserve extra: uv sync --extra superserve") from error
            self.sdk = Sandbox
        return self.sdk

    def cleanup(self, sandbox_id):
        self._sdk().kill_by_id(sandbox_id)

    def run(self, task, workspace, timeout, cancelled, record):
        env = task_environment(task)
        sdk = self._sdk()
        options = {
            "name": f"forge-{task.id}",
            "preview_access": "private",
            "timeout_seconds": math.ceil(timeout) + 60,
            "auto_delete_seconds": task.sandbox.auto_delete_seconds,
            "metadata": {"application": "forge-harness", "task": task.id},
        }
        if task.sandbox.template:
            options["from_template"] = task.sandbox.template
        if task.sandbox.snapshot:
            options["from_snapshot"] = task.sandbox.snapshot
        started = time.monotonic()
        sandbox = sdk.create(**options)
        stop_watcher = threading.Event()
        aborted = threading.Event()

        def watch():
            while not stop_watcher.wait(0.05):
                if cancelled() or time.monotonic() - started >= timeout:
                    aborted.set()
                    try:
                        sandbox.kill()
                    except Exception:
                        pass  # The final fresh delete request reports cleanup failure durably.
                    return

        watcher = threading.Thread(target=watch, daemon=True)
        remote = f"/tmp/forge-{uuid.uuid4().hex}"
        try:
            record({"sandbox_id": sandbox.id})
            watcher.start()
            setup = sandbox.commands.run(f"mkdir -p {remote}/.forge", timeout_seconds=30)
            if setup.exit_code:
                raise RuntimeError("SuperServe workspace setup failed")
            for item in task.inputs:
                target = f"{remote}/{item.path}"
                parent = shlex.quote(str(Path(target).parent))
                result = sandbox.commands.run(f"mkdir -p {parent}", timeout_seconds=30)
                if result.exit_code:
                    raise RuntimeError("SuperServe input directory setup failed")
                sandbox.files.write(target, (workspace / item.path).read_bytes(), timeout=30)
            sandbox.files.write(f"{remote}/.forge/worker.py", Path(worker.__file__).read_bytes(), timeout=30)
            remaining = max(0.1, timeout - (time.monotonic() - started))
            command = ["python3" if arg == "{python}" else arg for arg in task.command]
            job = {
                "command": command,
                "timeout_seconds": remaining,
                "max_output_bytes": task.max_output_bytes,
            }
            sandbox.files.write(f"{remote}/.forge/job.json", json.dumps(job), timeout=30)
            result = sandbox.commands.run(
                f"python3 {remote}/.forge/worker.py {remote}/.forge/job.json",
                cwd=remote,
                env=env,
                timeout_seconds=math.ceil(remaining) + 5,
            )
            if aborted.is_set():
                return {"status": "cancelled" if cancelled() else "timed_out", "exit_code": None}
            if result.exit_code:
                raise RuntimeError("SuperServe worker failed; use a template with Python 3 installed")
            outcome = json.loads(
                sandbox.files.read(f"{remote}/.forge/result.json", max_bytes=8192, timeout=30)
            )
            for name in ("stdout.log", "stderr.log"):
                content = sandbox.files.read(
                    f"{remote}/.forge/{name}", max_bytes=task.max_output_bytes, timeout=30
                )
                (workspace / ".forge" / name).write_bytes(content)
            if outcome["status"] == "succeeded":
                total = 0
                for name in task.outputs:
                    content = sandbox.files.read(
                        f"{remote}/{name}", max_bytes=task.max_artifact_bytes - total, timeout=30
                    )
                    total += len(content)
                    if total > task.max_artifact_bytes:
                        raise RuntimeError("Artifact size limit exceeded")
                    path = workspace / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(content)
                if task.sandbox.snapshot_on_success and all(
                    check["passed"] for check in evaluate_checks(task.checks, workspace)
                ):
                    snapshot = sandbox.snapshot(
                        kind="mem+fs", timeout=max(1, timeout - (time.monotonic() - started))
                    )
                    record({"snapshot_id": snapshot.id})
            return outcome
        except Exception:
            if aborted.is_set():
                return {"status": "cancelled" if cancelled() else "timed_out", "exit_code": None}
            raise
        finally:
            stop_watcher.set()
            if watcher.ident is not None:
                watcher.join(timeout=1)
            try:
                # A fresh API request also covers a failed request from the cancellation thread.
                sdk.kill_by_id(sandbox.id)
                record({"cleanup": "deleted"})
            except Exception as error:
                record({"cleanup": "required", "cleanup_error": type(error).__name__})
                raise RuntimeError(f"SuperServe cleanup failed for sandbox {sandbox.id}") from error
            finally:
                # The instance owns a pooled HTTP client; kill closes it even on API failure.
                try:
                    sandbox.kill()
                except Exception:
                    pass  # The fresh delete request above is authoritative for cleanup status.
