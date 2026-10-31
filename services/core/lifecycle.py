"""Reconcile persisted state against actual worker and recorder processes."""

import os
import subprocess
from services.core import store

TERMINAL = {"completed", "cancelled", "failed", "interrupted"}


def process_alive(pid, command):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        # -ww: Linux ps cuts piped output at 80 columns, which would hide the command we match.
        process = subprocess.run(
            ["ps", "-ww", "-p", str(pid), "-o", "stat=,command="], capture_output=True, text=True, timeout=2
        )
        value = process.stdout.strip()
        return bool(value and not value.startswith("Z") and command in value)
    except (OSError, subprocess.TimeoutExpired):
        return False


def worker_alive(pid, campaign_id):
    return process_alive(pid, f"services.core.runner {campaign_id}") or process_alive(
        pid, f"services.harness.campaign {campaign_id}"
    )


def reconcile():
    for snapshot in store.campaigns():
        camera = snapshot.get("recording", {})
        if (
            camera.get("status") == "recording"
            and isinstance(camera.get("pid"), int)
            and not process_alive(camera["pid"], "scripts/capture.py")
        ):
            updated = []

            def stop_camera(current, expected=camera["pid"], applied=updated):
                recording = current.get("recording", {})
                if recording.get("pid") == expected and recording.get("status") == "recording":
                    recording["status"] = "failed"
                    applied.append(True)

            store.patch(snapshot["id"], stop_camera)
            if updated:
                store.event(
                    snapshot["id"],
                    "recording_error",
                    "The recorder process stopped. Partial footage is preserved.",
                    "data",
                )
        if snapshot["status"] in TERMINAL:
            continue
        pid = snapshot.get("worker_pid")
        if snapshot.get("dispatch_pending") and snapshot.get("controller_pid") == os.getpid():
            continue
        if worker_alive(pid, snapshot["id"]):
            continue

        changed = []

        def interrupt(current, expected=pid, applied=changed):
            if current.get("worker_pid") != expected or current["status"] in TERMINAL:
                return
            current.update(
                status="interrupted",
                progress=None,
                error="The training worker stopped. Completed evidence is preserved; launch a new campaign to retry.",
            )
            applied.append(True)
            for experiment in current.get("experiments", []):
                if experiment["status"] not in TERMINAL and experiment["status"] != "rejected":
                    experiment["status"] = "interrupted"
            if current.get("recording", {}).get("status") == "recording":
                current["recording"]["status"] = "failed"

        store.patch(snapshot["id"], interrupt)
        if changed:
            store.event(
                snapshot["id"],
                "interrupted",
                "Worker process is no longer alive. The read model has been reconciled.",
                "trainer",
            )
