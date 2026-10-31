"""Faithful time-stamped screen capture of the live app, with no generated activity."""

import argparse
import json
import subprocess
import time
import shutil
import uuid
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SESSION = f"forge-live-capture-{os.getpid()}"


def browser(*args):
    result = subprocess.run(
        ["agent-browser", "--session", SESSION, *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    return result.stdout


def main():
    from services.core import store

    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument(
        "--production-scan",
        help="With --launch: start from the Production failures page and click 'Train on these failures'",
    )
    parser.add_argument(
        "--wait-for-idle", action="store_true", help="Queue one capture until the GPU campaign is terminal"
    )
    parser.add_argument("--no-tour", action="store_true", help="Keep the camera on the command center")
    parser.add_argument("--experiments", type=int, choices=[1, 2, 3, 4], default=4)
    parser.add_argument("--steps", type=int, choices=[24, 48, 96, 192], default=192)
    parser.add_argument("--final-samples", type=int, choices=[8, 24, 64, 100], default=64)
    parser.add_argument(
        "--export", action="store_true", help="Export evidence and accelerated videos after completion"
    )
    parser.add_argument("--interval", type=float, default=3)
    args = parser.parse_args()
    if args.wait_for_idle:
        for _ in range(2160):
            if all(
                s["status"] in {"completed", "failed", "cancelled", "interrupted"} for s in store.campaigns()
            ):
                break
            time.sleep(5)
        else:
            raise TimeoutError("The queued capture expired after three hours without launching compute")
    browser("--args", "--disable-gpu", "--idle-timeout", "0", "open", "http://127.0.0.1:3000/lab")
    browser("set", "viewport", "1440", "1000")
    browser("open", "http://127.0.0.1:3000/lab")
    browser("wait", "--fn", "document.body.innerText.includes('Systems connected')")
    # Recording creates a fresh browser context. Explicitly reopen the app before launch.
    raw = ROOT / "artifacts" / f"capture-{uuid.uuid4().hex[:8]}.webm"
    browser("record", "start", str(raw), "http://127.0.0.1:3000/lab")
    video_started = time.monotonic()
    browser("open", "http://127.0.0.1:3000/lab")
    browser("wait", "--fn", "document.body.innerText.includes('Systems connected')")
    start = datetime.now(timezone.utc).isoformat()
    client = httpx.Client(timeout=60)
    prelaunch = None
    if args.launch and args.production_scan:
        known = {s["id"] for s in client.get("http://127.0.0.1:8000/campaigns").json()}
        browser("find", "role", "button", "click", "--name", "Production failures", "--exact")
        browser("wait", "--fn", "document.body.innerText.includes('Train on these failures')")
        browser("wait", "--fn", f"document.body.innerText.includes({json.dumps(args.production_scan)})")
        time.sleep(6)  # let the viewer read the scan before training starts
        prelaunch = ROOT / "artifacts" / "prelaunch-recording.png"
        browser("screenshot", str(prelaunch))
        prelaunch_time = time.monotonic()
        browser("find", "role", "button", "click", "--name", "Train on these failures")
        for _ in range(60):
            fresh = [s for s in client.get("http://127.0.0.1:8000/campaigns").json() if s["id"] not in known]
            if fresh:
                args.campaign = fresh[0]["id"]
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("Campaign did not initialize; prelaunch capture is preserved")
    elif args.launch:
        known = {s["id"] for s in client.get("http://127.0.0.1:8000/campaigns").json()}
        browser("find", "role", "button", "click", "--name", "New campaign")
        for name, value in [
            ("budget", "120"),
            ("experiments", str(args.experiments)),
            ("steps", str(args.steps)),
            ("samples", "24"),
            ("final_samples", str(args.final_samples)),
            ("strategies", "sft_grpo"),
            ("final_backend", "lm_eval_azure"),
        ]:
            browser("select", f"select[name={name}]", value)
        browser("fill", "input[name=name]", "Mathematical reasoning · independent research")
        prelaunch = ROOT / "artifacts" / "prelaunch-recording.png"
        browser("screenshot", str(prelaunch))
        prelaunch_time = time.monotonic()
        browser("find", "role", "button", "click", "--name", "Launch campaign")
        for _ in range(60):
            fresh = [s for s in client.get("http://127.0.0.1:8000/campaigns").json() if s["id"] not in known]
            if fresh:
                args.campaign = fresh[0]["id"]
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("Campaign did not initialize; prelaunch capture is preserved")
        browser("open", "http://127.0.0.1:3000/lab")
    if not args.campaign:
        raise ValueError("Use --launch or --campaign")
    if not args.launch:
        browser("select", "#campaign-select", args.campaign)
        browser(
            "wait",
            "--fn",
            f"document.querySelector('.page-footer')?.innerText.includes({json.dumps(args.campaign)})",
        )
    out = ROOT / "artifacts" / args.campaign / "recording"
    if (out / "capture-manifest.json").exists():
        if args.launch:
            raise FileExistsError("A new campaign must not overwrite an existing recording")
        out = out / ("continuation-" + datetime.now(timezone.utc).strftime("%H%M%S"))
    frames = out / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    manifest = {
        "campaign_id": args.campaign,
        "method": "Continuous live browser WebM, plus time-stamped unmodified PNG evidence",
        "raw_recording": "training_raw.webm",
        "interval_seconds": args.interval,
        "full_lifecycle": args.launch,
        "frames": [],
        "started_at": start,
        "markers": [],
    }
    if prelaunch:
        first = frames / "prelaunch.png"
        shutil.copy2(prelaunch, first)
        manifest["frames"].append({"path": str(first), "captured_at": start, "monotonic": prelaunch_time})
    (ROOT / "data/recorded-campaign.json").write_text(
        json.dumps(
            {
                "campaign_id": args.campaign,
                "recording_manifest": str(out / "capture-manifest.json"),
                "full_lifecycle": args.launch,
            },
            indent=2,
        )
    )
    terminal_frames = 0
    probe_shown = False
    probe_at = None
    camera_view = "briefing"
    tour_step = 0
    previous_stage = None
    terminal_seen = False
    store.patch(
        args.campaign,
        lambda s: s.update(
            recording={"status": "recording", "started_at": manifest["started_at"], "pid": os.getpid()}
        ),
    )
    try:
        for index in range(3000):
            began = time.monotonic()
            current = browser("get", "url").strip()
            if not current.startswith("http://127.0.0.1:3000/lab"):
                browser("open", "http://127.0.0.1:3000/lab")
            shot = frames / f"{index:06d}.png"
            browser("screenshot", str(shot))
            manifest["frames"].append(
                {
                    "path": str(shot),
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                    "monotonic": time.monotonic(),
                }
            )
            (out / "capture-manifest.json").write_text(json.dumps(manifest, indent=2))
            # The recorder must outlive a dropped HTTP keep-alive connection. Read the
            # cross-process transactional snapshot, independent of the API connection.
            state = store.get(args.campaign)
            if state is None:
                raise RuntimeError("The recorded campaign disappeared from the evidence store")
            from services.core.briefing import build

            manifest["frames"][-1]["report"] = build(state, store.events(args.campaign))
            manifest["frames"][-1]["camera_view"] = camera_view
            if state["status"] != previous_stage:
                previous_stage = state["status"]
                manifest["markers"].append(
                    {
                        "stage": previous_stage,
                        "elapsed_seconds": time.monotonic() - video_started,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "experiment_id": state["experiments"][-1]["id"] if state["experiments"] else None,
                        "strategy": state["experiments"][-1]["strategy"] if state["experiments"] else None,
                    }
                )
            try:
                if (
                    not args.no_tour
                    and not probe_shown
                    and any(len(e.get("metrics", [])) >= 6 for e in state["experiments"])
                ):
                    browser("find", "role", "button", "click", "--name", "Governance", "--exact")
                    browser("wait", "--text", "Put the gate to the test")
                    browser("find", "role", "button", "click", "--name", "Run integrity probe", "--exact")
                    browser(
                        "wait", "--fn", "document.body.innerText.includes('Checkpoint promotion was blocked')"
                    )
                    probe_shown = True
                    camera_view = "governance"
                    probe_at = time.monotonic()
                    manifest["markers"].append(
                        {
                            "stage": "controlled_governance_probe",
                            "elapsed_seconds": probe_at - video_started,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                elif probe_at and time.monotonic() - probe_at > 20:
                    tours = [
                        ("Agent chat", "Research room", "chat"),
                        ("Durable actors", "The memory behind the research.", "terse"),
                        ("Training briefing", "What the research director sees", "briefing"),
                    ]
                    label, heading, camera_view = tours[tour_step]
                    browser("find", "role", "button", "click", "--name", label, "--exact")
                    browser("wait", "--text", heading)
                    tour_step += 1
                    probe_at = time.monotonic() if tour_step < len(tours) else None
                    manifest["markers"].append(
                        {
                            "stage": f"camera_{camera_view}",
                            "elapsed_seconds": time.monotonic() - video_started,
                            "timestamp": store.now(),
                        }
                    )
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                # An optional camera tour is never allowed to stop the continuous video.
                manifest.setdefault("navigation_errors", []).append(
                    {"timestamp": store.now(), "error": str(error)[:500]}
                )
                probe_shown = True
                probe_at = time.monotonic()
            if state["status"] in {"completed", "failed", "cancelled", "interrupted"}:
                terminal_seen = True
                if not terminal_frames and state["status"] == "completed":
                    browser("find", "role", "button", "click", "--name", "Results", "--exact")
                    browser("wait", "--text", "Read the model's work")
                    camera_view = "results"
                terminal_frames += 1
                if terminal_frames >= 4:
                    break
            time.sleep(max(0.1, args.interval - (time.monotonic() - began)))
    finally:
        manifest["ended_at"] = datetime.now(timezone.utc).isoformat()
        manifest["terminal_state_observed"] = terminal_seen
        manifest["full_lifecycle"] = bool(args.launch and terminal_seen)
        (out / "capture-manifest.json").write_text(json.dumps(manifest, indent=2))
        try:
            browser("record", "stop")
            shutil.move(raw, out / "training_raw.webm")
        finally:
            browser("close")
            store.patch(
                args.campaign,
                lambda s: s.update(
                    recording={
                        "status": "saved"
                        if terminal_seen and (out / "training_raw.webm").is_file()
                        else "failed",
                        "started_at": manifest["started_at"],
                        "artifact_path": str(
                            (out / "training_raw.webm").relative_to(ROOT / "artifacts" / args.campaign)
                        ),
                    }
                ),
            )
    print(
        json.dumps({"frames": len(manifest["frames"]), "manifest": str(out / "capture-manifest.json")}),
        flush=True,
    )
    if args.export and terminal_seen and store.get(args.campaign)["status"] == "completed":
        for script in ["export-evidence.py", "render-video.py", "capture-demo.py", "build-demo.py"]:
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / script), "--campaign", args.campaign], check=True
            )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # A prelaunch failure must not leave an unbounded orphan camera context.
        for command in [("record", "stop"), ("close",)]:
            try:
                browser(*command)
            except Exception:
                pass
        raise
