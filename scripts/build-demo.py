"""A factual two-minute edit made exclusively from recorded browser footage."""

import argparse
import json
import subprocess
import sys
import base64
import html
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def duration(path):
    return float(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(path),
            ],
            text=True,
        ).strip()
    )


def caption_png(text, target):
    """Render exact text using the app font, without optional FFmpeg text libraries."""
    font = base64.b64encode((ROOT / "public/fonts/ibm-plex-sans-latin.woff2").read_bytes()).decode()
    page = target.with_suffix(".html")
    page.write_text(f"""<!doctype html><html><meta charset="utf-8"><style>
@font-face{{font-family:Plex;src:url(data:font/woff2;base64,{font}) format('woff2')}}
*{{box-sizing:border-box}}html,body{{margin:0;width:1440px;height:118px;background:#111513;color:#e9eee9}}
body{{display:flex;align-items:center;padding:14px 28px;font:22px/1.35 Plex,sans-serif}}
main{{white-space:pre-line}}</style><main>{html.escape(text)}</main></html>""")
    session = f"forge-captions-{os.getpid()}"

    def browser(*args):
        subprocess.run(
            ["agent-browser", "--session", session, *args], check=True, capture_output=True, timeout=30
        )

    browser("--allow-file-access", "open", page.as_uri())
    browser("set", "viewport", "1440", "118")
    browser("open", page.as_uri())
    browser("wait", "--fn", "document.fonts.status === 'loaded'")
    browser("screenshot", str(target))


def main(campaign_id):
    from dotenv import load_dotenv
    from services.core import store
    from services.core.engine import artifact
    from services.core.cloud import mirror

    load_dotenv(ROOT / ".env")
    state = store.get(campaign_id)
    if not any(
        r["decision"] == "rejected" and r["evidence"].get("controlled_probe") for r in state["reviews"]
    ):
        raise ValueError("The controlled governance claim requires a recorded rejected probe")
    final = state["final"]
    folder = store.ARTIFACTS / campaign_id / "recording"
    raw, fast, tour = [
        folder / name for name in ["training_raw.mp4", "training_100x.mp4", "evidence-tour.webm"]
    ]
    capture = json.loads((folder / "capture-manifest.json").read_text())
    if not capture.get("full_lifecycle"):
        raise ValueError("The demo must be backed by a complete authentic session")
    markers = json.loads((folder / "evidence-tour.json").read_text())["markers"]
    total, accelerated = duration(raw), duration(fast)
    clip = min(25, accelerated)
    training = [m["elapsed_seconds"] for m in capture["markers"] if m["stage"] == "training"]
    if not training:
        raise ValueError("No actual training stage was recorded")
    first = final["baseline"]
    last = final["checkpoint"]
    prior = next(
        (
            s["final"]
            for s in store.campaigns()
            if s["id"] != campaign_id
            and s.get("config", {}).get("final_backend") == "lm_eval_azure"
            and s.get("final")
        ),
        None,
    )
    context = (
        f" Earlier independent run: {prior['baseline']['correct']}/{prior['baseline']['n']} to {prior['checkpoint']['correct']}/{prior['checkpoint']['n']}."
        if prior
        else ""
    )
    segments = [
        (
            tour,
            markers["comparison"],
            15,
            f"The same held-out questions, before and after training.\nOriginal {first['correct']}/{first['n']}. Frozen candidate {last['correct']}/{last['n']}. Actual outputs, preserved in the report.",
        ),
        (
            raw,
            0,
            15,
            "Give Forge a model, a benchmark and a compute budget.\nRecorded launch. The research director chooses experiments from measured evidence.",
        ),
        (
            fast,
            0,
            clip,
            "Actual autonomous research at 100 times speed.\nThe unedited source, timestamps, losses and Relay message IDs are preserved.",
        ),
        (
            raw,
            training[0] + 2,
            (35 - clip) / 2,
            "Actual training footage at normal speed.\nTerse persists coordination. The GPU updates real LoRA parameters.",
        ),
        (
            raw,
            training[-1] + 2,
            (35 - clip) / 2,
            "Later research, selected using earlier measurements.\nEvery attempted strategy, failure and corrective decision remains in the trace.",
        ),
        (
            tour,
            markers["governance"],
            20,
            "Independent governance exercises a real veto.\nThe controlled integrity fixture changes the reward and requests sealed data. Promotion is blocked.",
        ),
        (
            raw,
            max(0, total - 20),
            20,
            f"Selection froze before the independent Azure benchmark.\n{first['correct']}/{first['n']} to {last['correct']}/{last['n']}, {final['improvement_pp']:+.1f} percentage points on this subset.{context}\nSmall samples. Uncertainty and prior results are preserved. No general superiority claim.",
        ),
        (
            tour,
            markers["artifacts"],
            15,
            "A checkpoint, a measurement and the complete paper trail.\nDownload the adapter, predictions, governance decisions, trajectories and reproduction manifest.",
        ),
    ]
    edit = folder / "demo-edit"
    edit.mkdir(exist_ok=False)
    commands, elapsed, subtitles = [], 0.0, []
    for index, (source, start, seconds, caption) in enumerate(segments):
        text = edit / f"{index:02d}.txt"
        text.write_text(caption)
        picture = edit / f"{index:02d}.png"
        caption_png(caption, picture)
        target = edit / f"{index:02d}.mp4"
        filters = "[0:v]scale=1440:1000,fps=30[screen];[screen][1:v]overlay=0:H-h:format=auto[v]"
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-n",
            "-ss",
            str(start),
            "-i",
            str(source),
            "-i",
            str(picture),
            "-t",
            str(seconds),
            "-an",
            "-filter_complex",
            filters,
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            str(target),
        ]
        subprocess.run(command, check=True)
        if abs(duration(target) - seconds) > 0.2:
            raise ValueError("A source clip is too short; do not silently loop research footage")
        commands.append(command)
        subtitles.append(
            {
                "from": elapsed,
                "to": elapsed + seconds,
                "text": caption,
                "source": str(source.relative_to(folder)),
                "source_start": start,
            }
        )
        elapsed += seconds
    concat = edit / "concat.txt"
    concat.write_text("\n".join(f"file '{edit / f'{i:02d}.mp4'}'" for i in range(len(segments))))
    output = folder / "forge_demo_2min.mp4"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-n",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(command, check=True)
    commands.append(command)
    if abs(duration(output) - 120) > 0.35:
        raise ValueError("Two-minute demo duration validation failed")
    provenance = folder / "demo-manifest.json"
    provenance.write_text(
        json.dumps(
            {
                "campaign_id": campaign_id,
                "duration_seconds": duration(output),
                "chapters": subtitles,
                "commands": commands,
                "claims": "Measured subset results only. No synthetic research activity.",
            },
            indent=2,
        )
    )
    artifact(campaign_id, output, "recording", "Forge two-minute demonstration")
    artifact(campaign_id, tour, "recording", "Recorded historical evidence tour")
    artifact(campaign_id, provenance, "recording", "Demo chapters and exact edit commands")
    mirror(campaign_id)
    print(json.dumps({"video": str(output), "seconds": duration(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    main(parser.parse_args().campaign)
