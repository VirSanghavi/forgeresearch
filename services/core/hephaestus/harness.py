"""What keeps Hephaestus running: a supervisor that restarts it, a health check, and the lab's service.

The lab runs as its own launchd service, not from Terminal, so training never inherits Terminal's Full
Disk Access.
"""

import os
import plistlib
import stat
import subprocess
import sys
import time
from pathlib import Path

from services.core import agents, store
from services.core.hephaestus import brain, computer, imessage, ledger, tools

LABEL = "ai.forge.lab"
MAX_RESTARTS = 5
RESTART_WINDOW = 600


def _check(name, ok, detail):
    return {"name": name, "ok": bool(ok), "detail": detail}


def checks(lab=None, path=imessage.DATABASE):
    """Each thing Hephaestus needs, whether it's working, and what to do if it isn't."""
    found = []
    chat = os.environ.get("FORGE_HEPHAESTUS_CHAT")
    terse = _terse()
    found.append(terse)
    try:
        (lab or tools.Lab()).call("GET", "/health")
        found.append(_check("Lab", True, "answering"))
    except tools.ToolError:
        found.append(
            _check("Lab", False, "not running. Run: python -m services.core.hephaestus lab-service install")
        )
    if not chat:
        found.append(_check("Chat", False, "set FORGE_HEPHAESTUS_CHAT"))
    else:
        try:
            count = imessage.count(chat, path)
            found.append(
                _check("Chat", count > 0, "readable" if count else "not found: send a message in it")
            )
        except imessage.MessagesError:
            found.append(
                _check("Chat", False, "no access here. Run this in Terminal, the app with Full Disk Access")
            )
    found.append(_model())
    found.append(
        _check(
            "Research agents",
            agents.configured(),
            agents.provider() if agents.configured() else "local director",
        )
    )
    vm = tools.vm_id() if terse["ok"] else None
    if not computer.ready():
        found.append(_check("Computer", False, "set SUPERSERVE_API_KEY"))
    elif not vm:
        found.append(_check("Computer", True, "no VM yet: say computer start"))
    else:
        try:
            state = computer.status(vm)
            found.append(_check("Computer", state != "gone", state))
        except computer.ComputerError as error:
            found.append(_check("Computer", False, str(error)))
    free = tools.free_memory_percent()
    if free is not None:
        needed = int(os.environ.get("FORGE_HEPHAESTUS_MIN_FREE_MEMORY") or 35)
        found.append(_check("Memory", free >= needed, f"{free}% free, training needs {needed}%"))
    return found


def _terse():
    """Its memory, and whether the audit chain still holds together."""
    try:
        chain = ledger.Ledger().verify_audit()
    except (ledger.Unavailable, ValueError):
        return _check("Terse", False, "not answering. Start the lab: npm run lab")
    if not chain["ok"]:
        return _check("Terse", False, f"the audit chain is broken at row {chain['broken_at']}")
    return _check("Terse", True, f"answering; audit chain verified over {chain['rows']} rows")


def _model():
    if not brain.connected():
        return _check("Model", False, "none: only plain commands work")
    name = f"{brain.provider()}, {brain.model_name()}"
    known = agents.deployments(*brain.connection())
    if known is not None and brain.model_name() not in known:
        return _check("Model", False, f"{name} isn't deployed. Deployed: {', '.join(known)}")
    return _check("Model", True, name)


def report(found):
    return "\n".join(f"{'ok ' if c['ok'] else 'NO '} {c['name']}: {c['detail']}" for c in found)


def supervise(argv=None, spawn=subprocess.call, sleep=time.sleep, clock=time.monotonic):
    """Keep the bot running, restarting it after a crash. Ctrl+C stops it, and so does a crash loop."""
    command = argv or [sys.executable, "-m", "services.core.hephaestus", "run"]
    restarts = []
    while True:
        started = clock()
        try:
            code = spawn(command)
        except KeyboardInterrupt:
            return 0
        if code in (0, -2, 130):
            return 0
        now = clock()
        restarts = [at for at in restarts if now - at < RESTART_WINDOW] + [now]
        if len(restarts) > MAX_RESTARTS:
            print(
                f"It failed {MAX_RESTARTS} times in 10 minutes, so I'm stopping. See the error above.",
                flush=True,
            )
            return code
        print(f"It stopped (code {code}) after {round(now - started)}s. Restarting.", flush=True)
        sleep(min(60, 2 ** len(restarts)))


def _plist_path():
    return Path.home() / "Library/LaunchAgents" / f"{LABEL}.plist"


def lab_plist(root=None):
    root = Path(root or store.ROOT)
    return {
        "Label": LABEL,
        "ProgramArguments": [str(root / ".venv/bin/python"), str(root / "scripts/lab.py")],
        "WorkingDirectory": str(root),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 30,
        "StandardOutPath": str(root / "data/lab-service.log"),
        "StandardErrorPath": str(root / "data/lab-service.log"),
        "EnvironmentVariables": {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONPATH": str(root / "services/training-core/src"),
            **{
                name: os.environ[name]
                for name in (
                    "FORGE_TERSE_PORT",
                    "FORGE_API_PORT",
                    "FORGE_WEB_PORT",
                    "FORGE_RELAYCAST_PORT",
                    "FORGE_RELAY_BRIDGE_PORT",
                )
                if os.environ.get(name)
            },
        },
    }


def lab_service(action, run=subprocess.run):
    """Install or remove the lab's launchd service. Its plist holds no credential."""
    path = _plist_path()
    domain = f"gui/{os.getuid()}"
    if action == "install":
        path.parent.mkdir(parents=True, exist_ok=True)
        run(["launchctl", "bootout", f"{domain}/{LABEL}"], capture_output=True)
        path.write_bytes(plistlib.dumps(lab_plist()))
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        done = run(["launchctl", "bootstrap", domain, str(path)], capture_output=True)
        print("The lab is running as a service." if done.returncode == 0 else "launchctl refused it.")
        return done.returncode
    run(["launchctl", "bootout", f"{domain}/{LABEL}"], capture_output=True)
    path.unlink(missing_ok=True)
    print("The lab's service is removed.")
    return 0
