"""A real Terse runtime for the tests that need one. It starts only when a test asks for it."""

import os
import signal
import socket
import subprocess
import time
import uuid
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]


class Runtime:
    """One native Terse process over one data folder. It can be killed and started again on that folder."""

    def __init__(self, folder):
        self.folder, self.process = Path(folder), None
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"

    def start(self):
        self.log = (self.folder / "runtime.log").open("a")
        self.process = subprocess.Popen(
            [
                str(ROOT / ".venv/bin/durable-actors"),
                *("dev", "--project", str(ROOT / "services/actors"), "--port", str(self.port)),
                *("--data-dir", str(self.folder / "state"), "--no-watch"),
            ],
            env=os.environ | {"DURABLE_ACTORS_CONTROL_PLANE_URL": self.url},
            stdout=self.log,
            stderr=self.log,
            start_new_session=True,
        )
        for _ in range(240):
            try:
                if httpx.get(self.url, timeout=1).status_code < 500:
                    return self
            except httpx.HTTPError:
                if self.process.poll() is not None:
                    raise RuntimeError((self.folder / "runtime.log").read_text()) from None
            time.sleep(0.25)
        raise TimeoutError("Terse test runtime did not start")

    def kill(self):
        os.killpg(self.process.pid, signal.SIGKILL)
        self.process.wait(timeout=15)
        self.log.close()

    def stop(self):
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            self.process.wait(timeout=15)
        self.log.close()


@pytest.fixture(scope="session")
def terse_runtime(tmp_path_factory):
    runtime = Runtime(tmp_path_factory.mktemp("terse")).start()
    yield runtime
    runtime.stop()


def use_runtime(monkeypatch, runtime):
    """Point every actor client at this runtime, with a mind of its own for this test."""
    from services.core import actor_client

    monkeypatch.setenv("DURABLE_ACTORS_CONTROL_PLANE_URL", runtime.url)
    monkeypatch.setenv("FORGE_HEPHAESTUS_ACTOR", "test-" + uuid.uuid4().hex[:12])
    monkeypatch.setattr(
        actor_client,
        "transport",
        actor_client.ReconciledTransport(actor_client.StableClient(runtime.url), retry_seconds=5),
    )


@pytest.fixture
def terse(terse_runtime, monkeypatch):
    use_runtime(monkeypatch, terse_runtime)
    return terse_runtime
