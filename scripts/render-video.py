"""Export faithful, playable full-speed and accelerated versions of the actual capture."""

import argparse
import json
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def inspect(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_name,width,height",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def render(campaign_id):
    from dotenv import load_dotenv
    from services.core import store
    from services.core.engine import artifact
    from services.core.cloud import mirror
    from forge_training.artifacts import file_sha256

    load_dotenv(ROOT / ".env")
    folder = store.ARTIFACTS / campaign_id / "recording"
    source = folder / "training_raw.webm"
    manifest = json.loads((folder / "capture-manifest.json").read_text())
    if not manifest.get("full_lifecycle") or not manifest.get("terminal_state_observed"):
        raise ValueError("An incomplete capture cannot be exported as a full research-session recording")
    original = inspect(source)
    duration = float(original["format"]["duration"])
    wall = (
        datetime.fromisoformat(manifest["ended_at"]) - datetime.fromisoformat(manifest["started_at"])
    ).total_seconds()
    native_duration = duration
    timed_frames = abs(duration - wall) > max(5, wall * 0.03)
    inputs = ["-i", str(source)]
    timing = []
    if timed_frames:
        # Native screen streams can omit idle frame durations. Reconstruct the exact
        # wall-time axis from observed PNGs, holding real captures without interpolation.
        frames = manifest["frames"]
        if len(frames) < 2:
            raise ValueError("Insufficient time-stamped observations to recover the recording clock")
        listing = []
        for index, frame in enumerate(frames):
            path = Path(frame["path"])
            if not path.is_file() or not path.resolve().is_relative_to(folder.resolve()):
                raise ValueError("A source capture is missing or outside the recording directory")
            interval = (
                (frames[index + 1]["monotonic"] - frame["monotonic"])
                if index + 1 < len(frames)
                else max(0.04, wall - (frame["monotonic"] - frames[0]["monotonic"]))
            )
            if interval <= 0:
                raise ValueError("Capture timestamps are not strictly ordered")
            listing.extend([f"file '{path}'", f"duration {interval:.9f}"])
            timing.append(
                {
                    "file": str(path.relative_to(folder)),
                    "duration": interval,
                    "captured_at": frame["captured_at"],
                    "sha256": file_sha256(path),
                    "report": frame.get("report"),
                    "camera_view": frame.get("camera_view"),
                }
            )
        listing.append(f"file '{frames[-1]['path']}'")
        concat = folder / "wall-time.ffconcat"
        concat.write_text("\n".join(listing) + "\n")
        inputs = ["-f", "concat", "-safe", "0", "-i", str(concat)]
        duration = sum(f["duration"] for f in timing)
        (folder / "frame-timing.json").write_text(json.dumps(timing, indent=2))
        with zipfile.ZipFile(folder / "source-frames.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(folder / "frame-timing.json", "frame-timing.json")
            for frame in timing:
                archive.write(folder / frame["file"], frame["file"])
    if duration < 1 or original["streams"][0]["width"] < 1280:
        raise ValueError("The source recording failed its duration or resolution check")
    commands, outputs = [], []
    for speed, name in [(1, "training_raw.mp4"), (100, "training_100x.mp4"), (120, "training_120x.mp4")]:
        target = folder / name
        if target.exists():
            raise FileExistsError(f"Preserve existing recording: {target}")
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-n",
            *inputs,
            "-an",
            "-vf",
            f"setpts=(PTS-STARTPTS)/{speed}" + ("" if speed == 1 and timed_frames else ",fps=30"),
            "-fps_mode",
            "vfr" if speed == 1 and timed_frames else "cfr",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(target),
        ]
        subprocess.run(command, check=True)
        result = inspect(target)
        actual = float(result["format"]["duration"])
        if abs(actual - duration / speed) > max(0.2, duration / speed * 0.002):
            raise ValueError(f"Unexpected video speed or duration: {name}")
        outputs.append({"file": name, "speed": speed, "sha256": file_sha256(target), "probe": result})
        commands.append(command)
    movie_manifest = {
        "campaign_id": campaign_id,
        "source_sha256": file_sha256(source),
        "capture_started_at": manifest["started_at"],
        "capture_ended_at": manifest["ended_at"],
        "full_lifecycle": manifest["full_lifecycle"],
        "native_stream_seconds": native_duration,
        "wall_seconds": wall,
        "source": "Actual live browser capture, 1440 by 1000",
        "timing_method": "Time-stamped screenshots held for their observed intervals, nominal 5-second cadence"
        if timed_frames
        else "Native continuous stream clock",
        "native_clock_drift_detected": timed_frames,
        "largest_capture_interval_seconds": max((f["duration"] for f in timing), default=0),
        "transformation": "Actual images only, no generated intermediate states. Accelerated exports use measured wall time. The unmodified native stream is retained separately.",
        "commands": commands,
        "outputs": outputs,
    }
    (folder / "video-manifest.json").write_text(json.dumps(movie_manifest, indent=2))
    for name, title in [
        ("training_raw.webm", "Original live screen recording"),
        ("training_raw.mp4", "Full-speed training recording"),
        ("training_100x.mp4", "Training recording at 100×"),
        ("training_120x.mp4", "Training recording at 120×"),
        ("capture-manifest.json", "Capture timestamps and stage markers"),
        ("video-manifest.json", "Video provenance and export commands"),
        *(
            [
                ("source-frames.zip", "Original time-stamped capture frames"),
                ("frame-timing.json", "Measured source frame durations"),
            ]
            if timed_frames
            else []
        ),
    ]:
        artifact(campaign_id, folder / name, "recording", title)
    store.patch(
        campaign_id,
        lambda s: s.update(recording={"status": "saved", "artifact_path": "recording/training_raw.mp4"}),
    )
    mirror(campaign_id)
    print(json.dumps({"campaign_id": campaign_id, "source_seconds": duration, "exports": outputs}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    render(parser.parse_args().campaign)
