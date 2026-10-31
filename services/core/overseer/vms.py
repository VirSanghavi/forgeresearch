"""Where a connection's agent runs: its own SuperServe VM, or a process on this machine.

A VM holds no real credential. Each one reaches only DNS, the traffic source and the ingest host, and
sees stand-in tokens that SuperServe swaps for the real values on requests to those hosts. An auto-pause
window is the dead-man switch: while the lab supervises, it resumes the VM; if the lab goes away, the
VM pauses and billing stops.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

from services.core import store
from services.core.overseer import agent as agent_module

# The sandbox's resolvers: a deny-all rule also blocks DNS, and domain rules need names to resolve.
DNS = ["1.1.1.1/32", "8.8.8.8/32"]
REMOTE = "/tmp/forge-agent"
MAX_STATE = 1024 * 1024


class VMError(RuntimeError):
    """A failure whose message is ours, so it is safe to show."""


def host_of(url):
    return urlsplit(url).hostname or ""


def active_seconds():
    return int(os.environ.get("FORGE_OVERSEER_VM_ACTIVE_SECONDS") or 3600)


def retain_seconds():
    return int(os.environ.get("FORGE_OVERSEER_VM_RETAIN_SECONDS") or 7 * 24 * 3600)


class LocalVM:
    """A plain process on this machine, not a security boundary. For development and private networks."""

    name = "local"
    needs_https = False

    def ready(self):
        return True, None

    def _directory(self, connection):
        return store.DATA / "overseer-agents" / connection["id"]

    def store_secrets(self, connection, config, secrets):
        directory = self._directory(connection)
        directory.mkdir(parents=True, exist_ok=True)
        directory.chmod(0o700)
        path = directory / "secrets.json"
        path.touch(mode=0o600)
        path.chmod(0o600)
        path.write_text(json.dumps(secrets))
        return ["secrets.json"]

    def provision(self, connection, config):
        directory = self._directory(connection)
        shutil.copyfile(agent_module.__file__, directory / "agent.py")
        self.write_config(connection, config)
        self.ensure_running(connection)
        return f"local-{connection['id']}"

    def write_config(self, connection, config):
        directory = self._directory(connection)
        temporary = directory / "config.json.tmp"
        temporary.write_text(json.dumps(config))
        temporary.replace(directory / "config.json")

    def ensure_running(self, connection):
        directory = self._directory(connection)
        secrets = json.loads((directory / "secrets.json").read_text())
        (directory / "stop").unlink(missing_ok=True)
        with open(directory / "agent.log", "ab") as log:
            # Only the named credentials reach the agent, not this process's environment. Isolated
            # mode and the copied file mean it can't import anything from the repository.
            subprocess.Popen(
                [sys.executable, "-I", str(directory / "agent.py"), str(directory)],
                cwd=directory,
                env={"PATH": os.environ.get("PATH", ""), **secrets},
                stdout=log,
                stderr=log,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )

    def read(self, connection):
        """The agent's state, {} before its first write, or None when the VM is gone."""
        directory = self._directory(connection)
        if not (directory / "config.json").exists():
            return None
        try:
            return json.loads((directory / "state.json").read_text())
        except (OSError, ValueError):
            return {}

    def halt(self, connection):
        directory = self._directory(connection)
        if directory.exists():
            (directory / "stop").touch()
            deadline = time.monotonic() + 5
            state = directory / "state.json"
            while time.monotonic() < deadline:
                if not state.exists() or json.loads(state.read_text()).get("status") == "stopped":
                    break
                time.sleep(0.1)

    def remove(self, connection):
        self.halt(connection)
        shutil.rmtree(self._directory(connection), ignore_errors=True)


class SuperServeVM:
    name = "superserve"
    needs_https = True

    def __init__(self, sandbox=None, secret=None, network=None):
        self.sandbox, self.secret, self.network = sandbox, secret, network

    def _sdk(self):
        if self.sandbox is None:
            try:
                from superserve import NetworkConfig, Sandbox, Secret
            except ImportError as error:
                raise VMError("Install the superserve extra: uv sync --extra superserve") from error
            self.sandbox, self.secret, self.network = Sandbox, Secret, NetworkConfig
        return self.sandbox, self.secret, self.network

    def ready(self):
        if not os.getenv("SUPERSERVE_API_KEY") and self.sandbox is None:
            return False, "Set SUPERSERVE_API_KEY to give the agent its own VM"
        from services.core.overseer import pull

        if not pull.configured():
            return (
                False,
                "A VM ships traces through the Supabase buffer: set SUPABASE_URL and the service key",
            )
        return True, None

    @staticmethod
    def _source_auth(auth):
        if auth["type"] == "bearer":
            return {"type": "bearer"}
        if auth["type"] == "basic":
            return {"type": "basic", "username": auth.get("username", "")}
        return {"type": "api-key", "header": auth["header"], "prefix": auth.get("prefix", "")}

    def store_secrets(self, connection, config, secrets):
        _, secret, _ = self._sdk()
        created = []
        try:
            if secrets.get("FORGE_SOURCE_SECRET"):
                name = f"forge-{connection['id']}-source"
                secret.create(
                    name=name,
                    value=secrets["FORGE_SOURCE_SECRET"],
                    hosts=[host_of(config["url"])],
                    auth=self._source_auth(config["auth"]),
                )
                created.append(name)
            name = f"forge-{connection['id']}-ingest"
            secret.create(
                name=name,
                value=secrets["FORGE_INGEST_TOKEN"],
                hosts=[host_of(config["sink_url"])],
                auth={"type": "bearer"},
            )
            created.append(name)
        except Exception:
            for name in created:
                self._delete_secret(name)
            raise
        return created

    def _delete_secret(self, name):
        _, secret, _ = self._sdk()
        try:
            secret.delete_by_name(name)
        except Exception as error:
            if type(error).__name__ != "NotFoundError":
                raise

    def provision(self, connection, config):
        sandbox_sdk, _, network = self._sdk()
        bindings = {}
        for name in json.loads(connection["secret_names"] or "[]"):
            bindings["FORGE_SOURCE_SECRET" if name.endswith("-source") else "FORGE_INGEST_TOKEN"] = name
        vm = sandbox_sdk.create(
            name=f"forge-{connection['id']}",
            timeout_seconds=active_seconds(),
            auto_delete_seconds=retain_seconds(),
            metadata={"application": "forge-overseer", "connection": connection["id"]},
            secrets=bindings,
            network=network(
                allow_out=[*DNS, host_of(config["url"]), host_of(config["sink_url"])],
                deny_out=["0.0.0.0/0"],
            ),
            preview_access="private",
        )
        try:
            probe = vm.commands.run(
                f"mkdir -p {REMOTE} && python3 --version && curl --version", timeout_seconds=30
            )
            if probe.exit_code:
                raise VMError("The VM's template needs Python 3 and curl for the agent")
            vm.files.write(f"{REMOTE}/agent.py", Path(agent_module.__file__).read_bytes(), timeout=30)
            vm.files.write(f"{REMOTE}/config.json", json.dumps(config).encode(), timeout=30)
            self._launch(vm)
        except Exception:
            sandbox_sdk.kill_by_id(vm.id)
            raise
        return vm.id

    @staticmethod
    def _launch(vm):
        started = vm.commands.run(
            f"sh -c 'cd {REMOTE} && nohup python3 -I agent.py {REMOTE} >> agent.log 2>&1 < /dev/null &'",
            timeout_seconds=30,
        )
        if started.exit_code:
            raise VMError("The agent would not start on the VM")

    def _connect(self, connection):
        sandbox_sdk, _, _ = self._sdk()
        try:
            return sandbox_sdk.connect(connection["vm_id"])
        except Exception as error:
            if type(error).__name__ == "NotFoundError":
                return None
            raise

    def write_config(self, connection, config):
        vm = self._connect(connection)
        if vm:
            vm.files.write(f"{REMOTE}/config.json", json.dumps(config).encode(), timeout=30)

    def ensure_running(self, connection):
        vm = self._connect(connection)
        if vm is None:
            raise VMError("The VM is gone")
        self._launch(vm)

    def read(self, connection):
        # Connecting also resumes a VM its auto-pause window paused, which is what keeps it running
        # for as long as the lab is here to watch it.
        vm = self._connect(connection)
        if vm is None:
            return None
        try:
            return json.loads(vm.files.read(f"{REMOTE}/state.json", max_bytes=MAX_STATE, timeout=30))
        except Exception as error:
            if type(error).__name__ == "NotFoundError" or isinstance(error, ValueError):
                return {}
            raise

    def halt(self, connection):
        if connection.get("vm_id"):
            sandbox_sdk, _, _ = self._sdk()
            sandbox_sdk.kill_by_id(connection["vm_id"])

    def remove(self, connection):
        self.halt(connection)
        for name in json.loads(connection["secret_names"] or "[]"):
            self._delete_secret(name)


PROVIDERS = {"local": LocalVM(), "superserve": SuperServeVM()}


def default():
    ready, _ = PROVIDERS["superserve"].ready()
    return "superserve" if ready else "local"


def describe():
    described = {}
    for name, provider in PROVIDERS.items():
        ready, reason = provider.ready()
        described[name] = {"ready": ready, "reason": reason}
    return {"default": default(), "providers": described}
