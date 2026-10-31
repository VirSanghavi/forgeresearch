"""Keep the local lab alive across terminal and chat interruptions, without a login item."""

import os
import plistlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LABEL = "com.forge.lab.338fedca"


def run():
    action = sys.argv[1] if len(sys.argv) > 1 else "status"
    if sys.platform != "darwin":
        raise SystemExit("Use npm run lab in a persistent terminal on this platform.")
    is_capture = action in {"capture", "record-after"}
    label = LABEL + (".queued-capture" if action == "record-after" else ".capture") if is_capture else LABEL
    domain = f"gui/{os.getuid()}"
    target = domain + "/" + label
    if action == "stop":
        subprocess.run(["launchctl", "bootout", target], check=True)
        return
    if action == "status":
        subprocess.run(["launchctl", "print", target], check=True)
        return
    if action not in {"start", "capture", "record-after"}:
        raise SystemExit("Use start, stop, status, capture, or record-after")
    status = subprocess.run(["launchctl", "print", target], capture_output=True)
    if status.returncode == 0:
        if is_capture and b"state = running" not in status.stdout:
            subprocess.run(["launchctl", "bootout", target], check=True)
        else:
            print(f"Already registered: {target}")
            return
    DATA.mkdir(exist_ok=True)
    command = [str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "scripts/lab.py")]
    if is_capture:
        command = [
            str(ROOT / ".venv/bin/python"),
            "-u",
            str(ROOT / "scripts/capture.py"),
            *(["--wait-for-idle"] if action == "record-after" else []),
            *(sys.argv[2:] or ["--launch", "--interval", "5"]),
        ]
    config = {
        "Label": label,
        "ProgramArguments": command,
        "WorkingDirectory": str(ROOT),
        "RunAtLoad": True,
        "KeepAlive": False if is_capture else {"SuccessfulExit": False},
        "ThrottleInterval": 5,
        "EnvironmentVariables": {
            "PATH": os.environ.get("PATH", "/opt/homebrew/bin:/usr/bin:/bin"),
            "HOME": str(Path.home()),
        },
        "StandardOutPath": str(DATA / (f"{action}.log" if is_capture else "supervisor.log")),
        "StandardErrorPath": str(DATA / (f"{action}-error.log" if is_capture else "supervisor-error.log")),
    }
    path = DATA / (label + ".plist")
    path.write_bytes(plistlib.dumps(config))
    subprocess.run(["launchctl", "bootstrap", domain, str(path)], check=True)
    print(f"Started {target}. This registration ends on logout; no login item was installed.")


if __name__ == "__main__":
    run()
