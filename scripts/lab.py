"""Start the complete local lab, reusing healthy services and owning only new processes."""

import os
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path
import httpx
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from services.core.chaos import RuntimeSupervisor  # noqa: E402

children = []
logs = []
(ROOT / "data").mkdir(exist_ok=True)
settings = dotenv_values(ROOT / ".env")
# Unset or loopback: this command runs a local Relaycast engine. Otherwise the bridge uses
# that engine, e.g. the hosted https://cast.agentrelay.com.
# FORGE_<NAME>_PORT runs a second lab beside the default one (children inherit the URLs).
PORT = {
    k: int(os.environ.get(f"FORGE_{k.upper()}_PORT", d))
    for k, d in {"terse": 7100, "api": 8000, "web": 3000, "relaycast": 7528, "relay_bridge": 7529}.items()
}
TERSE = os.environ.setdefault("DURABLE_ACTORS_CONTROL_PLANE_URL", f"http://127.0.0.1:{PORT['terse']}")
os.environ.setdefault("FORGE_RELAY_BRIDGE", f"http://127.0.0.1:{PORT['relay_bridge']}")
os.environ.setdefault("FORGE_API_URL", f"http://127.0.0.1:{PORT['api']}")
os.environ.update(FORGE_RELAY_BRIDGE_PORT=str(PORT["relay_bridge"]), FORGE_WEB_PORT=str(PORT["web"]))
RELAY_URL = (
    settings.get("FORGE_RELAY_URL")
    or os.environ.get("FORGE_RELAY_URL")
    # A workspace key with no URL means the hosted engine (services/relay/config.mjs).
    or (
        "https://cast.agentrelay.com"
        if settings.get("FORGE_RELAY_WORKSPACE_KEY") or os.environ.get("FORGE_RELAY_WORKSPACE_KEY")
        else f"http://127.0.0.1:{PORT['relaycast']}"
    )
)
LOCAL_RELAY = httpx.URL(RELAY_URL).host in ("127.0.0.1", "localhost")


def healthy(url):
    try:
        return httpx.get(url, timeout=1).status_code < 500
    except httpx.HTTPError:
        return False


def start(name, cmd, url, env=None):
    if healthy(url):
        print(f"{name}: using existing service", flush=True)
        return
    log = (ROOT / "data" / f"{name}.log").open("a")
    logs.append(log)
    child = subprocess.Popen(
        cmd, cwd=ROOT, stdout=log, stderr=log, start_new_session=True, env={**os.environ, **(env or {})}
    )
    children.append(child)
    for _ in range(90):
        if child.poll() is not None:
            raise RuntimeError(f"{name} failed. Read data/{name}.log")
        if healthy(url):
            print(f"{name}: ready", flush=True)
            return
        time.sleep(1)
    raise TimeoutError(f"{name} did not become ready")


def stop(_signal, _frame):
    raise KeyboardInterrupt


signal.signal(signal.SIGTERM, stop)


terse = None
try:
    if healthy(TERSE):
        print("terse: using existing service (not supervised; crash tests disabled)", flush=True)
    else:
        # Supervised: if the runtime dies it is restarted on the same data dir, and recorded.
        data_dir = Path(settings.get("FORGE_TERSE_DATA_DIR") or ROOT / "data/terse").resolve()
        # Hephaestus's memory lives here, including what people said to it: private to this account.
        data_dir.mkdir(parents=True, exist_ok=True)
        data_dir.chmod(0o700)
        log = (ROOT / "data" / "terse.log").open("a")
        logs.append(log)
        terse = RuntimeSupervisor(
            [
                str(ROOT / ".venv/bin/durable-actors"),
                *("dev", "--project", "services/actors", "--port", str(PORT["terse"])),
                *("--data-dir", str(data_dir), "--no-watch"),
            ],
            TERSE,
            data_dir,
            log,
            cwd=ROOT,
        )
        terse.start()
        print(f"terse: ready (supervised, pid {terse.process.pid})", flush=True)
    subprocess.run(
        [str(ROOT / ".venv/bin/durable-actors"), "generate", "--out-dir", "services/actors/generated"],
        check=True,
    )
    # Agent Relay. RELAY_URL picks the engine; FORGE_RELAY_WORKSPACE_KEY (env or .env) picks a
    # workspace on it and reaches only the bridge.
    relay_env = {k: v for k, v in settings.items() if k.startswith("FORGE_RELAY_") and v}
    relay_env["FORGE_RELAY_URL"] = RELAY_URL
    # Per-launch bridge secret: only the bridge and the API (and the campaign workers it
    # spawns) hold it. A bridge reused from an earlier launch keeps that launch's secret.
    token_file = ROOT / "data" / "relay-bridge.token"
    if not (healthy(os.environ["FORGE_RELAY_BRIDGE"] + "/health") and token_file.exists()):
        token_file.unlink(missing_ok=True)
        fd = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_urlsafe(32))
    bridge_token = token_file.read_text().strip()
    relay_env["FORGE_RELAY_BRIDGE_TOKEN"] = bridge_token
    if LOCAL_RELAY:
        start(
            "relaycast",
            [
                "node",
                "services/relay/server.mjs",
                "data/relaycast.db",
                str(PORT["relaycast"]),
            ],
            RELAY_URL,
        )
    start(
        "relay-bridge",
        ["node", "services/relay/bridge.mjs"],
        os.environ["FORGE_RELAY_BRIDGE"] + "/health",
        relay_env,
    )
    start(
        "api",
        [
            sys.executable,
            "-m",
            "uvicorn",
            "services.api:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT["api"]),
        ],
        os.environ["FORGE_API_URL"] + "/health",
        env=os.environ | {"FORGE_RELAY_BRIDGE_TOKEN": bridge_token},
    )
    web = f"http://127.0.0.1:{PORT['web']}"
    start("web", ["npx", "next", "dev", "--hostname", "127.0.0.1", "--port", str(PORT["web"])], web)
    print(
        f"\nForge is ready: lab {web}/lab, landing page {web}/\nCtrl+C stops only services started by this command.",
        flush=True,
    )
    # Open the lab for someone at a terminal; scripted and headless runs are left alone.
    if sys.stdout.isatty() and os.environ.get("FORGE_NO_BROWSER") != "1":
        import webbrowser

        webbrowser.open(f"{web}/lab")
    while True:
        restart = terse and terse.poll()
        if restart:
            print(
                f"terse: pid {restart['old_pid']} exited ({restart['signal'] or restart['returncode']}); "
                f"restarted as pid {restart['pid']} in {restart['restart_seconds']:.1f}s",
                flush=True,
            )
        for child in children:
            if child.poll() is not None:
                raise RuntimeError("A lab process exited. Inspect data/*.log")
        time.sleep(0.2)
except KeyboardInterrupt:
    pass
finally:
    if terse:
        terse.stop()
    for child in reversed(children):
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
    for log in logs:
        log.close()
