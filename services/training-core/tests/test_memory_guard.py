from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest


@pytest.fixture
def guard():
    script = Path(__file__).parents[1] / "scripts/run_memory_bounded.py"
    spec = importlib.util.spec_from_file_location("memory_guard", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_guard_refuses_to_launch_when_memory_is_low(guard, monkeypatch, tmp_path):
    marker, report = tmp_path / "started", tmp_path / "report.json"
    monkeypatch.setattr(guard.psutil, "virtual_memory", lambda: SimpleNamespace(available=1))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guard",
            "--report",
            str(report),
            "--",
            sys.executable,
            "-c",
            f"open({str(marker)!r}, 'w').close()",
        ],
    )
    assert guard.main() == 1
    assert not marker.exists()
    assert json.loads(report.read_text())["status"] == "aborted"


def test_guard_preserves_command_exit_status(guard, monkeypatch, tmp_path):
    report = tmp_path / "report.json"
    monkeypatch.setattr(
        guard.psutil, "virtual_memory", lambda: SimpleNamespace(available=8 * 1024**3)
    )
    monkeypatch.setattr(guard.psutil, "swap_memory", lambda: SimpleNamespace(used=0))
    monkeypatch.setattr(
        sys,
        "argv",
        ["guard", "--report", str(report), "--", sys.executable, "-c", "raise SystemExit(7)"],
    )
    assert guard.main() == 7
    assert json.loads(report.read_text())["status"] == "failed"


def test_guard_terminates_the_worker_and_its_child(guard, monkeypatch, tmp_path):
    report, pidfile = tmp_path / "report.json", tmp_path / "pids.json"
    command = (
        "import json,os,subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        f"open({str(pidfile)!r},'w').write(json.dumps([os.getpid(),child.pid])); "
        "time.sleep(30)"
    )
    monkeypatch.setattr(
        guard.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=1 if pidfile.exists() else 8 * 1024**3),
    )
    monkeypatch.setattr(guard.psutil, "swap_memory", lambda: SimpleNamespace(used=0))
    monkeypatch.setattr(
        sys, "argv", ["guard", "--report", str(report), "--", sys.executable, "-c", command]
    )
    assert guard.main() == 1
    assert json.loads(report.read_text())["status"] == "aborted"
    for pid in json.loads(pidfile.read_text()):
        assert not psutil.pid_exists(pid)


def test_terminating_the_monitor_cleans_up_its_command(tmp_path):
    import subprocess
    import time

    script = Path(__file__).parents[1] / "scripts/run_memory_bounded.py"
    pidfile, report = tmp_path / "worker-pid", tmp_path / "report.json"
    worker = f"import os,time; open({str(pidfile)!r},'w').write(str(os.getpid())); time.sleep(30)"
    process = subprocess.Popen(
        [
            sys.executable,
            str(script),
            "--report",
            str(report),
            "--start-available-mb",
            "1",
            "--minimum-available-mb",
            "1",
            "--max-swap-growth-mb",
            "1048576",
            "--",
            sys.executable,
            "-c",
            worker,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pidfile.exists()
        process.terminate()
        _, stderr = process.communicate(timeout=10)
        assert process.returncode == 1, stderr
        assert not psutil.pid_exists(int(pidfile.read_text()))
        assert "termination signal" in json.loads(report.read_text())["reason"]
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
