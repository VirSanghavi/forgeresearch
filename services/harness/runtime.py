"""Durable DAG scheduler with bounded attempts and verified artifact handoffs."""

import concurrent.futures
import contextlib
import fcntl
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

from services.harness.checks import evaluate_checks
from services.harness.providers import LocalProvider, SuperServeProvider, task_environment
from services.harness.schema import Workflow

TERMINAL = {"succeeded", "failed", "blocked", "cancelled", "timed_out", "output_limit", "uncertain"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def runtime_identity():
    return {
        "python": sys.version,
        "sources": {path.name: file_digest(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
    }


def safe_file(root, name):
    root = Path(root).resolve()
    path = root / name
    if not path.resolve().is_relative_to(root) or any(
        p.is_symlink() for p in [path, *path.parents] if p != root
    ):
        raise ValueError(f"Artifact escapes its workspace or uses a symlink: {name}")
    if not path.is_file():
        raise ValueError(f"Required file missing: {name}")
    return path


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


@contextlib.contextmanager
def run_lock(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another scheduler owns this run") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def inspect_run(folder):
    return json.loads((Path(folder) / "state.json").read_text())


def verify_run(folder):
    folder = Path(folder)
    state = inspect_run(folder)
    previous = None
    for index, event in enumerate(state["events"]):
        content = {key: value for key, value in event.items() if key != "sha256"}
        if event["sequence"] != index or event["previous"] != previous or digest(content) != event["sha256"]:
            raise ValueError("Event journal integrity mismatch")
        previous = event["sha256"]
    if state["plan_sha256"] != digest(state["plan"]):
        raise ValueError("Frozen workflow integrity mismatch")
    checked = 0
    declared = Workflow.model_validate(state["plan"])
    plans = {task.id: task for task in declared.tasks}
    if set(plans) != set(state["tasks"]):
        raise ValueError("Task state does not match the frozen workflow")
    for key, task in state["tasks"].items():
        if task["status"] != "succeeded":
            continue
        if set(task["artifacts"]) != set(plans[key].outputs):
            raise ValueError("Artifact coverage does not match the frozen workflow")
        completions = [e for e in state["events"] if e["kind"] == "attempt_finished" and e["task"] == key]
        if (
            not completions
            or completions[-1]["status"] != "succeeded"
            or completions[-1].get("artifacts_sha256") != digest(task["artifacts"])
        ):
            raise ValueError("Artifact manifest does not match the completion journal")
        for name, expected in task["artifacts"].items():
            path = safe_file(folder, f"{task['workspace']}/{name}")
            if file_digest(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
                raise ValueError(f"Artifact integrity mismatch: {name}")
            checked += 1
        spec = plans[key]
        if any(not check["passed"] for check in evaluate_checks(spec.checks, folder / task["workspace"])):
            raise ValueError("Completed task no longer satisfies its quality gates")
    return {"status": "verified", "artifacts": checked, "events": len(state["events"])}


class Harness:
    def __init__(self, workflow, root, folder, providers=None, on_event=None):
        self.plan = workflow if isinstance(workflow, Workflow) else Workflow.model_validate(workflow)
        self.root = Path(root).resolve()
        self.folder = Path(folder).resolve()
        self.providers = providers or {"local": LocalProvider(), "superserve": SuperServeProvider()}
        self.on_event = on_event
        self.mutex = threading.RLock()
        self.stop = threading.Event()
        self.state = None

    def event(self, kind, task=None, **fields):
        with self.mutex:
            events = self.state["events"]
            event = {
                "sequence": len(events),
                "previous": events[-1]["sha256"] if events else None,
                "kind": kind,
                "task": task,
                **fields,
            }
            event["sha256"] = digest(event)
            events.append(event)
            atomic_json(self.folder / "state.json", self.state)
            if self.on_event:
                self.on_event(event)

    def cancelled(self):
        return (
            self.stop.is_set() or (self.folder / "CANCEL").exists() or time.time() >= self.state["deadline"]
        )

    def _inputs(self):
        inputs = {}
        for task in self.plan.tasks:
            for item in task.inputs:
                if not item.task:
                    path = safe_file(self.root, item.source)
                    if path.stat().st_size > task.max_artifact_bytes:
                        raise ValueError(f"Input exceeds artifact limit: {item.source}")
                    inputs[item.source] = file_digest(path)
        return inputs

    def _prepare(self, resume, retry_interrupted):
        path = self.folder / "state.json"
        if path.exists():
            if not resume:
                raise ValueError("Run already exists; use resume or choose a new run directory")
            verify_run(self.folder)
            self.state = inspect_run(self.folder)
            if self.state["runtime"] != runtime_identity():
                raise ValueError("Harness implementation or Python runtime changed; start a new run")
            if self.state["plan"] != self.plan.model_dump() or self.state["inputs"] != self._inputs():
                raise ValueError("Workflow or source inputs changed; start a new run")
            if self.state["root"] != str(self.root):
                raise ValueError("Source root changed; start a new run")
            for task in self.plan.tasks:
                entry = self.state["tasks"][task.id]
                last = entry["attempts"][-1] if entry["attempts"] else {}
                if last.get("sandbox_id") and last.get("cleanup") != "deleted":
                    self.providers["superserve"].cleanup(last["sandbox_id"])
                    last["cleanup"] = "deleted"
                if entry["status"] in ("running", "uncertain"):
                    # Never kill a recorded PID on restart: it could now belong to someone else.
                    entry["status"] = "uncertain"
                    local_still_possible = (
                        task.backend == "local" and time.time() < last.get("deadline", 0) + 5
                    )
                    if retry_interrupted and local_still_possible:
                        raise ValueError(
                            "Interrupted local worker may still run; wait for its recorded deadline"
                        )
                    if retry_interrupted and task.retry_safe and len(entry["attempts"]) < task.attempts:
                        entry["status"] = "pending"
                    self.event("recovered", task.id, status=entry["status"])
            # A previously uncertain dependency may now be eligible to retry. Release only
            # descendants whose dependencies are all successful or eligible for execution.
            changed = True
            while changed:
                changed = False
                for task in self.plan.tasks:
                    entry = self.state["tasks"][task.id]
                    if entry["status"] == "blocked" and all(
                        self.state["tasks"][key]["status"] in {"pending", "succeeded"} for key in task.needs
                    ):
                        entry["status"] = "pending"
                        changed = True
                        self.event("dependency_recovered", task.id)
            return
        if resume:
            raise ValueError("Cannot resume a run without state.json")
        self.state = {
            "schema_version": 1,
            "plan": self.plan.model_dump(),
            "plan_sha256": digest(self.plan.model_dump()),
            "runtime": runtime_identity(),
            "root": str(self.root),
            "inputs": self._inputs(),
            "deadline": time.time() + self.plan.max_seconds,
            "status": "running",
            "events": [],
            "tasks": {
                task.id: {"status": "pending", "attempts": [], "artifacts": {}} for task in self.plan.tasks
            },
        }
        self.event("created", name=self.plan.name)

    def _stage(self, task, workspace):
        total = 0
        for item in task.inputs:
            if item.task:
                dependency = self.state["tasks"][item.task]
                source = safe_file(self.folder, f"{dependency['workspace']}/{item.source}")
                expected = dependency["artifacts"][item.source]["sha256"]
            else:
                source = safe_file(self.root, item.source)
                expected = self.state["inputs"][item.source]
            total += source.stat().st_size
            if total > task.max_artifact_bytes:
                raise ValueError("Input exceeds artifact size limit")
            content = source.read_bytes()
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f"Input checksum mismatch: {item.source}")
            destination = workspace / item.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)

    def _attempt(self, task, entry, attempt):
        workspace = self.folder / entry["workspace"]

        def record(fields):
            with self.mutex:
                attempt.update(fields)
                self.event("execution", task.id, **fields)

        try:
            workspace.mkdir(parents=True, exist_ok=False)
            (workspace / ".forge").mkdir()
            self._stage(task, workspace)
            timeout = min(attempt["deadline"], self.state["deadline"]) - time.time()
            if timeout <= 0 or self.cancelled():
                return {"status": "cancelled"}
            result = self.providers[task.backend].run(task, workspace, timeout, self.cancelled, record)
            if not isinstance(result, dict) or result.get("status") not in {
                "succeeded",
                "failed",
                "cancelled",
                "timed_out",
                "output_limit",
            }:
                raise ValueError("Provider returned an invalid terminal result")
            if result["status"] == "succeeded" and result.get("exit_code") != 0:
                raise ValueError("Provider reported success without a zero exit code")
            if result["status"] == "succeeded":
                artifacts = {}
                total = 0
                for name in task.outputs:
                    path = safe_file(workspace, name)
                    size = path.stat().st_size
                    total += size
                    if total > task.max_artifact_bytes:
                        raise ValueError("Artifacts exceed the configured size limit")
                    artifacts[name] = {"sha256": file_digest(path), "bytes": size}
                result["artifacts"] = artifacts
                result["checks"] = evaluate_checks(task.checks, workspace)
                if any(not check["passed"] for check in result["checks"]):
                    result["status"] = "failed"
                    result["error_type"] = "QualityGateFailed"
            return result
        except Exception as error:
            # Provider exception messages may contain credentials or signed URLs.
            return {"status": "failed", "error_type": type(error).__name__}

    def run(self, resume=False, retry_interrupted=False):
        with run_lock(self.folder):
            # Validate credentials exist before any task runs, without persisting values.
            if not resume:
                for task in self.plan.tasks:
                    task_environment(task)
            self._prepare(resume, retry_interrupted)
            for task in self.plan.tasks:
                if self.state["tasks"][task.id]["status"] == "pending":
                    task_environment(task)
            self.state["status"] = "running"
            self.event("scheduler_started")
            active = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.plan.max_parallel) as pool:
                try:
                    while True:
                        for future in list(active):
                            if not future.done():
                                continue
                            task = active.pop(future)
                            entry = self.state["tasks"][task.id]
                            result = future.result()
                            with self.mutex:
                                entry["attempts"][-1].update(result)
                                entry["status"] = result["status"]
                                entry["artifacts"] = result.get("artifacts", {})
                                if result["status"] in {"failed", "timed_out", "output_limit"}:
                                    if (
                                        task.retry_safe
                                        and len(entry["attempts"]) < task.attempts
                                        and not self.cancelled()
                                        and entry["attempts"][-1].get("cleanup") != "required"
                                    ):
                                        entry["status"] = "pending"
                                        entry["ready_at"] = time.time() + task.retry_delay_seconds * 2 ** (
                                            len(entry["attempts"]) - 1
                                        )
                                    elif self.plan.fail_fast:
                                        self.stop.set()
                                self.event(
                                    "attempt_finished",
                                    task.id,
                                    status=result["status"],
                                    next=entry["status"],
                                    artifacts_sha256=digest(entry["artifacts"]),
                                )
                        occupied = {resource for task in active.values() for resource in task.resources}
                        for task in self.plan.tasks:
                            entry = self.state["tasks"][task.id]
                            if entry["status"] != "pending":
                                continue
                            dependencies = [self.state["tasks"][key]["status"] for key in task.needs]
                            if self.cancelled() or any(s in TERMINAL - {"succeeded"} for s in dependencies):
                                entry["status"] = "cancelled" if self.cancelled() else "blocked"
                                self.event("task_finished", task.id, status=entry["status"])
                                continue
                            if (
                                len(active) >= self.plan.max_parallel
                                or set(task.resources) & occupied
                                or any(s != "succeeded" for s in dependencies)
                                or time.time() < entry.get("ready_at", 0)
                            ):
                                continue
                            with self.mutex:
                                attempt = {
                                    "number": len(entry["attempts"]) + 1,
                                    "backend": task.backend,
                                    "deadline": min(
                                        time.time() + task.timeout_seconds, self.state["deadline"]
                                    ),
                                }
                                entry["attempts"].append(attempt)
                                entry["workspace"] = f"tasks/{task.id}/attempt-{attempt['number']}"
                                entry["status"] = "running"
                                self.event("attempt_started", task.id, attempt=attempt["number"])
                            active[pool.submit(self._attempt, task, entry, attempt)] = task
                            occupied.update(task.resources)
                        if not active and all(e["status"] in TERMINAL for e in self.state["tasks"].values()):
                            break
                        time.sleep(0.025)
                except BaseException:
                    self.stop.set()
                    raise
            statuses = {entry["status"] for entry in self.state["tasks"].values()}
            self.state["status"] = "succeeded" if statuses == {"succeeded"} else "failed"
            if "cancelled" in statuses:
                self.state["status"] = "cancelled"
            self.event("finished", status=self.state["status"])
            return self.state
