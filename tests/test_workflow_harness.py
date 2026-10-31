"""Real-process and provider-contract tests for the durable workflow harness."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from services.harness.__main__ import main
from services.harness.campaign import campaign_workflow
from services.harness.providers import SuperServeProvider
from services.harness.runtime import Harness, atomic_json, inspect_run, run_lock, verify_run
from services.harness.schema import Workflow


def task(id="one", code="print('hello')", **kwargs):
    return {"id": id, "command": ["{python}", "-c", code], **kwargs}


def plan(*tasks, **kwargs):
    return Workflow(name="test", tasks=list(tasks or [task()]), **kwargs)


def run(tmp_path, workflow, **kwargs):
    return Harness(workflow, tmp_path, tmp_path / "run", **kwargs).run()


@pytest.mark.parametrize(
    "tasks",
    [
        [task(), task()],
        [task(needs=["missing"])],
        [task(needs=["two"]), task("two", needs=["one"])],
        [task(outputs=["../escape"])],
        [task(outputs=["/absolute"])],
        [task(outputs=[".forge/job.json"])],
        [task(attempts=2)],
        [task(inputs=[{"path": "x", "source": "missing", "task": "two"}]), task("two")],
    ],
)
def test_invalid_graphs_and_contracts(tasks):
    with pytest.raises(ValidationError):
        plan(*tasks)


def test_real_dag_handoff_resume_and_tamper(tmp_path):
    workflow = plan(
        task(code="from pathlib import Path; Path('answer').write_text('42')", outputs=["answer"]),
        task(
            "two",
            "from pathlib import Path; assert Path('input').read_text() == '42'",
            needs=["one"],
            inputs=[{"path": "input", "source": "answer", "task": "one"}],
        ),
    )
    state = run(tmp_path, workflow)
    assert state["status"] == "succeeded"
    assert verify_run(tmp_path / "run")["artifacts"] == 1
    resumed = Harness(workflow, tmp_path, tmp_path / "run").run(resume=True)
    assert all(len(t["attempts"]) == 1 for t in resumed["tasks"].values())
    output = tmp_path / "run" / state["tasks"]["one"]["workspace"] / "answer"
    output.write_text("43")
    with pytest.raises(ValueError, match="integrity mismatch"):
        Harness(workflow, tmp_path, tmp_path / "run").run(resume=True)


def test_source_inputs_frozen_and_no_secret_inheritance(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_PRIVATE_SECRET", "must-not-inherit")
    (tmp_path / "source").write_text("original")
    workflow = plan(
        task(
            code="import os; from pathlib import Path; assert 'HARNESS_PRIVATE_SECRET' not in os.environ; "
            "assert Path('copied').read_text() == 'original'",
            inputs=[{"path": "copied", "source": "source"}],
        )
    )
    assert run(tmp_path, workflow)["status"] == "succeeded"
    (tmp_path / "source").write_text("changed")
    with pytest.raises(ValueError, match="inputs changed"):
        Harness(workflow, tmp_path, tmp_path / "run").run(resume=True)


def test_missing_env_fails_before_any_work(tmp_path, monkeypatch):
    monkeypatch.delenv("HARNESS_REQUIRED", raising=False)
    with pytest.raises(ValueError, match="Missing required"):
        run(tmp_path, plan(task(env=["HARNESS_REQUIRED"])))
    assert not (tmp_path / "run/state.json").exists()


def test_failure_blocks_dependents_but_not_independent_tasks(tmp_path):
    state = run(tmp_path, plan(task(code="raise SystemExit(9)"), task("two", needs=["one"]), task("other")))
    assert state["tasks"]["one"]["attempts"][0]["exit_code"] == 9
    assert state["tasks"]["two"]["status"] == "blocked"
    assert state["tasks"]["other"]["status"] == "succeeded"


@pytest.mark.parametrize(
    "code,kwargs,status",
    [
        ("import time; time.sleep(10)", {"timeout_seconds": 1}, "timed_out"),
        ("print('x' * 10000)", {"max_output_bytes": 1024}, "output_limit"),
        ("pass", {"outputs": ["missing"]}, "failed"),
        (
            "from pathlib import Path; Path('out').write_text('x' * 10)",
            {"outputs": ["out"], "max_artifact_bytes": 5},
            "failed",
        ),
        ("from pathlib import Path; Path('out').symlink_to('/etc/hosts')", {"outputs": ["out"]}, "failed"),
    ],
)
def test_limits_and_output_validation(tmp_path, code, kwargs, status):
    state = run(tmp_path, plan(task(code=code, **kwargs)))
    assert state["tasks"]["one"]["status"] == status


def test_explicit_safe_retries(tmp_path):
    workflow = plan(
        task(
            code="from pathlib import Path; raise SystemExit(0 if Path.cwd().name == 'attempt-2' else 1)",
            retry_safe=True,
            attempts=2,
            retry_delay_seconds=0,
        )
    )
    state = run(tmp_path, workflow)
    assert state["status"] == "succeeded"
    assert len(state["tasks"]["one"]["attempts"]) == 2


def test_cancellation_terminates_running_work(tmp_path):
    harness = Harness(plan(task(code="import time; time.sleep(30)")), tmp_path, tmp_path / "run")
    timer = threading.Timer(0.2, harness.stop.set)
    timer.start()
    started = time.monotonic()
    try:
        assert harness.run()["status"] == "cancelled"
    finally:
        timer.cancel()
    assert time.monotonic() - started < 5


class TrackingProvider:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0

    def run(self, task, workspace, timeout, cancelled, record):
        with self.lock:
            self.active += 1
            self.peak = max(self.active, self.peak)
        time.sleep(0.15)
        with self.lock:
            self.active -= 1
        return {"status": "succeeded", "exit_code": 0}


@pytest.mark.parametrize("resources,peak", [([], 2), (["gpu"], 1)])
def test_parallelism_and_exclusive_resources(tmp_path, resources, peak):
    provider = TrackingProvider()
    state = run(
        tmp_path,
        plan(task(resources=resources), task("two", resources=resources)),
        providers={"local": provider},
    )
    assert state["status"] == "succeeded"
    assert provider.peak == peak


def test_run_lock_fences_duplicate_scheduler(tmp_path):
    with run_lock(tmp_path / "run"), pytest.raises(RuntimeError, match="owns this run"):
        run(tmp_path, plan())


def test_uncertain_work_is_not_automatically_replayed(tmp_path):
    workflow = plan(task(retry_safe=True, attempts=2))
    run(tmp_path, workflow)
    state = inspect_run(tmp_path / "run")
    state["tasks"]["one"]["status"] = "running"
    atomic_json(tmp_path / "run/state.json", state)
    resumed = Harness(workflow, tmp_path, tmp_path / "run").run(resume=True)
    assert resumed["tasks"]["one"]["status"] == "uncertain"
    assert len(resumed["tasks"]["one"]["attempts"]) == 1
    with pytest.raises(ValueError, match="may still run"):
        Harness(workflow, tmp_path, tmp_path / "run").run(resume=True, retry_interrupted=True)
    state = inspect_run(tmp_path / "run")
    state["tasks"]["one"]["attempts"][0]["deadline"] = 0
    atomic_json(tmp_path / "run/state.json", state)
    retried = Harness(workflow, tmp_path, tmp_path / "run").run(resume=True, retry_interrupted=True)
    assert retried["status"] == "succeeded"
    assert len(retried["tasks"]["one"]["attempts"]) == 2


def test_event_tampering_is_detected(tmp_path):
    run(tmp_path, plan())
    state = inspect_run(tmp_path / "run")
    state["events"][0]["name"] = "tampered"
    atomic_json(tmp_path / "run/state.json", state)
    with pytest.raises(ValueError, match="journal integrity"):
        verify_run(tmp_path / "run")


class FakeFiles:
    def __init__(self):
        self.data = {}

    def write(self, path, content, *, timeout):
        self.data[path] = content.encode() if isinstance(content, str) else content

    def read(self, path, *, max_bytes, timeout):
        data = self.data[path]
        if len(data) > max_bytes:
            raise ValueError("File too large")
        return data


class FakeSandbox:
    id = "sandbox-contract-test"

    def __init__(self):
        self.files = FakeFiles()
        self.commands = self
        self.killed = False
        self.snapshot_args = None

    def run(self, command, *, timeout_seconds, cwd=None, env=None):
        if command.startswith("python3"):
            job = json.loads(self.files.data[f"{cwd}/.forge/job.json"])
            assert job["command"][0] == "python3"
            assert "SUPERSERVE_API_KEY" not in env
            self.files.data.update(
                {
                    f"{cwd}/.forge/result.json": b'{"status":"succeeded","exit_code":0}',
                    f"{cwd}/.forge/stdout.log": b"done",
                    f"{cwd}/.forge/stderr.log": b"",
                    f"{cwd}/result.txt": b"verified artifact",
                }
            )
        return SimpleNamespace(exit_code=0)

    def snapshot(self, *, kind, timeout):
        self.snapshot_args = {"kind": kind, "timeout": timeout}
        return SimpleNamespace(id="snapshot-contract-test")

    def kill(self):
        self.killed = True


class FakeSDK:
    def __init__(self, fail_cleanup=False):
        self.sandbox = FakeSandbox()
        self.fail_cleanup = fail_cleanup
        self.options = None

    def create(self, **kwargs):
        self.options = kwargs
        return self.sandbox

    def kill_by_id(self, sandbox_id):
        assert sandbox_id == self.sandbox.id
        if self.fail_cleanup:
            raise RuntimeError("secret-provider-error-do-not-persist")
        self.sandbox.killed = True


def test_superserve_contract_transfer_snapshot_and_cleanup(tmp_path):
    sdk = FakeSDK()
    provider = SuperServeProvider(sdk)
    workflow = plan(
        task(
            backend="superserve",
            outputs=["result.txt"],
            sandbox={"template": "forge/python", "snapshot_on_success": True},
        )
    )
    state = run(tmp_path, workflow, providers={"superserve": provider})
    attempt = state["tasks"]["one"]["attempts"][0]
    assert state["status"] == "succeeded"
    assert sdk.options["preview_access"] == "private"
    assert sdk.options["from_template"] == "forge/python"
    assert sdk.sandbox.snapshot_args["kind"] == "mem+fs"
    assert attempt["snapshot_id"] == "snapshot-contract-test"
    assert attempt["cleanup"] == "deleted"
    assert sdk.sandbox.killed
    assert verify_run(tmp_path / "run")["artifacts"] == 1


def test_superserve_cleanup_failure_is_durable_and_retried_on_resume(tmp_path):
    sdk = FakeSDK(fail_cleanup=True)
    provider = SuperServeProvider(sdk)
    workflow = plan(task(backend="superserve"))
    state = run(tmp_path, workflow, providers={"superserve": provider})
    assert state["status"] == "failed"
    assert state["tasks"]["one"]["attempts"][0]["cleanup"] == "required"
    assert "secret-provider-error" not in (tmp_path / "run/state.json").read_text()
    sdk.fail_cleanup = False
    Harness(workflow, tmp_path, tmp_path / "run", providers={"superserve": provider}).run(resume=True)
    assert sdk.sandbox.killed


def test_cli_local_example_and_status(tmp_path, capsys):
    example = Path(__file__).resolve().parents[1] / "examples/harness/local.json"
    assert main(["validate", str(example)]) == 0
    assert main(["run", str(example), "--root", str(tmp_path), "--output", str(tmp_path / "run")]) == 0
    assert main(["status", str(tmp_path / "run")]) == 0
    assert main(["verify", str(tmp_path / "run")]) == 0
    assert main(["resume", str(tmp_path / "run")]) == 0
    assert '"succeeded"' in capsys.readouterr().out


@pytest.mark.parametrize(
    "value,operator,expected,passed",
    [
        (0.8, "ge", 0.7, True),
        (0.5, "ge", 0.7, False),
        (True, "eq", True, True),
        (1, "eq", True, False),
        ("high", "gt", 1, False),
        (float("nan"), "eq", 0, False),
    ],
)
def test_quality_gates_control_dependency_release(tmp_path, value, operator, expected, passed):
    content = json.dumps({"score": value})
    workflow = plan(
        task(
            code=f"from pathlib import Path; Path('metrics.json').write_text({content!r})",
            outputs=["metrics.json"],
            checks=[
                {
                    "artifact": "metrics.json",
                    "pointer": "/score",
                    "operator": operator,
                    "value": expected,
                }
            ],
        ),
        task("consumer", needs=["one"]),
    )
    state = run(tmp_path, workflow)
    assert (state["status"] == "succeeded") == passed
    assert (state["tasks"]["consumer"]["status"] == "succeeded") == passed
    assert state["tasks"]["one"]["attempts"][0]["checks"][0]["passed"] == passed


def test_independent_watchdog_survives_scheduler_crash(tmp_path):
    workflow = plan(task(code="import time; time.sleep(60)", timeout_seconds=1))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(workflow.model_dump_json())
    folder = tmp_path / "run"
    process = subprocess.Popen(
        [sys.executable, "-m", "services.harness", "run", str(manifest), "--output", str(folder)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if (folder / "state.json").exists():
                attempts = inspect_run(folder)["tasks"]["one"]["attempts"]
                if attempts and attempts[-1].get("pid"):
                    break
            time.sleep(0.025)
        else:
            pytest.fail("Supervisor did not start")
        os.kill(process.pid, signal.SIGKILL)
        process.wait(timeout=2)
        result = folder / "tasks/one/attempt-1/.forge/result.json"
        deadline = time.monotonic() + 5
        while not result.exists() and time.monotonic() < deadline:
            time.sleep(0.025)
        assert json.loads(result.read_text())["status"] == "timed_out"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_campaign_adapter_preserves_existing_runner_without_secret_values(tmp_path):
    workflow = campaign_workflow(
        "forge-test",
        {"max_seconds": 300},
        tmp_path,
        {"FORGE_BACKEND": "modal", "PRIVATE_TOKEN": "never-persist"},
    )
    task = workflow.tasks[0]
    assert task.attempts == 1 and not task.retry_safe
    assert task.backend == "local"
    assert task.env == ["FORGE_BACKEND", "PRIVATE_TOKEN"]
    assert "never-persist" not in workflow.model_dump_json()
    assert task.command[-1] == "forge-test"
    assert task.inputs[0].source == "services/harness/campaign_entry.py"


def test_superserve_setup_failure_still_deletes_sandbox(tmp_path):
    sdk = FakeSDK()
    sdk.sandbox.run = lambda *a, **kw: SimpleNamespace(exit_code=1)
    sdk.sandbox.commands = sdk.sandbox
    state = run(tmp_path, plan(task(backend="superserve")), providers={"superserve": SuperServeProvider(sdk)})
    assert state["status"] == "failed"
    assert sdk.sandbox.killed
    assert state["tasks"]["one"]["attempts"][0]["cleanup"] == "deleted"


def test_superserve_cancel_deletes_active_sandbox(tmp_path):
    sdk = FakeSDK()
    original = sdk.sandbox.run

    def blocking(command, **kwargs):
        if command.startswith("python3"):
            deadline = time.monotonic() + 3
            while not sdk.sandbox.killed and time.monotonic() < deadline:
                time.sleep(0.025)
            raise RuntimeError("remote command disconnected")
        return original(command, **kwargs)

    sdk.sandbox.run = blocking
    harness = Harness(
        plan(task(backend="superserve")),
        tmp_path,
        tmp_path / "run",
        providers={"superserve": SuperServeProvider(sdk)},
    )
    timer = threading.Timer(0.2, harness.stop.set)
    timer.start()
    try:
        state = harness.run()
    finally:
        timer.cancel()
    assert state["status"] == "cancelled"
    assert sdk.sandbox.killed


def test_published_superserve_sdk_accepts_adapter_calls():
    """Runs when the optional extra is installed; no network or credentials required."""
    sdk = pytest.importorskip("superserve")
    from superserve.commands import Commands
    from superserve.files import Files
    from unittest.mock import create_autospec

    sandbox = create_autospec(sdk.Sandbox)
    sandbox.create(
        name="forge-test",
        preview_access="private",
        timeout_seconds=60,
        auto_delete_seconds=3600,
        metadata={"task": "test"},
        from_template="python",
    )
    sandbox.kill_by_id("sandbox-test")
    sandbox.snapshot(kind="mem+fs", timeout=60)
    commands = create_autospec(Commands, instance=True)
    commands.run("python3 worker.py", cwd="/tmp/forge", env={}, timeout_seconds=30)
    files = create_autospec(Files, instance=True)
    files.write("/tmp/forge/input", b"hello", timeout=30)
    files.read("/tmp/forge/output", max_bytes=1024, timeout=30)


def test_campaign_entry_runs_existing_module_with_launch_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_BACKEND", "modal")
    root = tmp_path / "source"
    (root / "services/harness").mkdir(parents=True)
    (root / "services/core").mkdir()
    entry = Path(__file__).resolve().parents[1] / "services/harness/campaign_entry.py"
    (root / "services/harness/campaign_entry.py").write_bytes(entry.read_bytes())
    (root / "services/core/runner.py").write_text(
        "import os, sys\nfrom pathlib import Path\n"
        "assert sys.argv[1] == 'forge-test'\nassert os.environ['FORGE_BACKEND'] == 'modal'\n"
        "Path('invoked').write_text('existing runner')\n"
    )
    workflow = campaign_workflow("forge-test", {"max_seconds": 30}, root, {"FORGE_BACKEND": "modal"})
    assert Harness(workflow, root, tmp_path / "run").run()["status"] == "succeeded"
    assert (root / "invoked").read_text() == "existing runner"


@pytest.mark.parametrize(
    "mode,module", [("classic", "services.core.runner"), ("durable", "services.harness.campaign")]
)
def test_command_center_selects_campaign_harness(tmp_path, monkeypatch, mode, module):
    import shutil
    from services.core import engine, store

    monkeypatch.setenv("FORGE_WORKER_HARNESS", mode)
    monkeypatch.setattr(store, "ARTIFACTS", tmp_path)
    (tmp_path / "forge-test").mkdir()
    monkeypatch.setattr(store, "patch", lambda *_: {"config": {"max_seconds": 30}})
    monkeypatch.setattr(shutil, "which", lambda _: None)
    calls = []

    def start(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(pid=123, wait=lambda **_: 0)

    monkeypatch.setattr(subprocess, "Popen", start)
    assert engine.spawn("forge-test") == 123
    assert calls[0] == [sys.executable, "-m", module, "forge-test"]


def test_recovery_releases_blocked_descendants(tmp_path):
    workflow = plan(task(retry_safe=True, attempts=2), task("two", needs=["one"]))
    run(tmp_path, workflow)
    state = inspect_run(tmp_path / "run")
    state["tasks"]["one"]["status"] = "uncertain"
    state["tasks"]["one"]["attempts"][0]["deadline"] = 0
    state["tasks"]["two"] = {"status": "blocked", "attempts": [], "artifacts": {}}
    # Give the recovered dependent a fresh attempt directory, as a genuinely blocked task has.
    import shutil

    shutil.rmtree(tmp_path / "run/tasks/two")
    atomic_json(tmp_path / "run/state.json", state)
    result = Harness(workflow, tmp_path, tmp_path / "run").run(resume=True, retry_interrupted=True)
    assert result["status"] == "succeeded"
    assert len(result["tasks"]["two"]["attempts"]) == 1


def test_campaign_reconciliation_recognizes_durable_worker(monkeypatch):
    from services.core import lifecycle

    monkeypatch.setattr(
        lifecycle,
        "process_alive",
        lambda pid, command: pid == 123 and command == "services.harness.campaign forge-test",
    )
    assert lifecycle.worker_alive(123, "forge-test")
    assert not lifecycle.worker_alive(123, "other-campaign")
