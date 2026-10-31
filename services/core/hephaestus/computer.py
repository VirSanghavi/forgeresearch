"""Hephaestus's own computer: one persistent SuperServe VM it can run commands on.

It is deliberately separate from everything that matters. It holds no credential and no customer data,
and has no route to this Mac or the lab. It reaches only package and code hosts, so what it reads there
can't reach anything, and whatever it reads is treated as untrusted. It pauses itself when left active, so
it can't bill unattended, and it keeps its files through a pause.
"""

import os

# The sandbox's resolvers: a deny-all rule also blocks DNS, and domain rules need names to resolve.
DNS = ["1.1.1.1/32", "8.8.8.8/32"]
HOSTS = (
    "pypi.org,files.pythonhosted.org,github.com,codeload.github.com,raw.githubusercontent.com,huggingface.co"
)
MAX_OUTPUT = 4000
MAX_READ = 64 * 1024
SDK = None  # Tests set this to a double of (Sandbox, NetworkConfig).


class ComputerError(RuntimeError):
    """A failure whose message is ours, so it is safe to show. Provider messages can echo requests."""


def ready():
    return bool(SDK or os.getenv("SUPERSERVE_API_KEY"))


def _sdk():
    if SDK:
        return SDK
    if not os.getenv("SUPERSERVE_API_KEY"):
        raise ComputerError("Set SUPERSERVE_API_KEY so I can have a VM.")
    try:
        from superserve import NetworkConfig, Sandbox
    except ImportError as error:
        raise ComputerError("Install the superserve extra: uv sync --extra superserve") from error
    return Sandbox, NetworkConfig


def hosts():
    configured = os.environ.get("FORGE_HEPHAESTUS_VM_ALLOW") or HOSTS
    return [host.strip() for host in configured.split(",") if host.strip()]


def _guarded(action):
    try:
        return action()
    except ComputerError:
        raise
    except Exception as error:
        raise ComputerError(f"The VM failed ({type(error).__name__}).") from error


def create():
    sandbox, network = _sdk()

    def boot():
        return sandbox.create(
            name="forge-hephaestus",
            timeout_seconds=int(os.environ.get("FORGE_HEPHAESTUS_VM_ACTIVE_SECONDS") or 1800),
            auto_delete_seconds=int(os.environ.get("FORGE_HEPHAESTUS_VM_RETAIN_SECONDS") or 14 * 24 * 3600),
            metadata={"application": "forge-hephaestus"},
            network=network(allow_out=[*DNS, *hosts()], deny_out=["0.0.0.0/0"]),
            preview_access="private",
        ).id

    return _guarded(boot)


def _connect(vm_id):
    sandbox, _ = _sdk()
    try:
        return sandbox.connect(vm_id)
    except Exception as error:
        if type(error).__name__ == "NotFoundError":
            raise ComputerError("My VM is gone. Start a new one.") from error
        raise ComputerError(f"The VM failed ({type(error).__name__}).") from error


def status(vm_id):
    sandbox, _ = _sdk()

    def look():
        for vm in sandbox.list():
            if vm.id == vm_id:
                return str(getattr(vm, "status", "unknown")).rsplit(".", maxsplit=1)[-1].lower()
        return "gone"

    return _guarded(look)


def resume(vm_id):
    _connect(vm_id)
    return {"running": vm_id}


def run(vm_id, command, seconds=60):
    vm = _connect(vm_id)
    limit = max(1, min(int(seconds), 300))
    done = _guarded(lambda: vm.commands.run(command, timeout_seconds=limit))
    return {
        "exit_code": done.exit_code,
        "stdout": done.stdout[-MAX_OUTPUT:],
        "stderr": done.stderr[-MAX_OUTPUT:],
    }


def write(vm_id, path, text):
    vm = _connect(vm_id)
    _guarded(lambda: vm.files.write(path, text.encode(), timeout=30))
    return {"written": path, "bytes": len(text.encode())}


def read(vm_id, path):
    vm = _connect(vm_id)
    raw = _guarded(lambda: vm.files.read(path, max_bytes=MAX_READ, timeout=30))
    return {"path": path, "text": raw.decode("utf-8", "replace")[-MAX_OUTPUT:]}


def pause(vm_id):
    vm = _connect(vm_id)
    _guarded(lambda: vm.pause(wait=True))
    return {"paused": vm_id}


def destroy(vm_id):
    sandbox, _ = _sdk()
    _guarded(lambda: sandbox.kill_by_id(vm_id))
    return {"deleted": vm_id}
