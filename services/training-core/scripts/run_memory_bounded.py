"""Run a command with system-memory monitoring and cleanup of its worker children."""

from __future__ import annotations

import argparse
import contextlib
import json
import signal
import subprocess
import time
from pathlib import Path

import psutil


def stop_tree(process: subprocess.Popen) -> None:
    try:
        children = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    # Let the worker's signal handler preserve failure evidence and stop its phases.
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    for child in children:
        with contextlib.suppress(psutil.NoSuchProcess):
            if child.is_running():
                child.terminate()
    _, remaining = psutil.wait_procs(children, timeout=2)
    for child in remaining:
        with contextlib.suppress(psutil.NoSuchProcess):
            child.kill()
    psutil.wait_procs(remaining, timeout=2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--minimum-available-mb", type=int, default=2048)
    parser.add_argument("--start-available-mb", type=int, default=6144)
    parser.add_argument("--max-swap-growth-mb", type=int, default=256)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if min(args.minimum_available_mb, args.start_available_mb, args.max_swap_growth_mb) < 1:
        parser.error("Memory thresholds must be positive")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A command is required after --")
    # Claim the report path before launching anything; never overwrite evidence.
    with args.report.open("x", encoding="utf-8") as stream:
        mib = 1024**2
        starting_swap = psutil.swap_memory().used
        available = psutil.virtual_memory().available / mib
        report = {
            "limits": {
                "start_available_mb": args.start_available_mb,
                "minimum_available_mb": args.minimum_available_mb,
                "max_swap_growth_mb": args.max_swap_growth_mb,
            },
            "command": command,
            "minimum_available_mb": available,
            "peak_swap_growth_mb": 0.0,
            "status": "aborted",
            "reason": None,
        }
        process = None

        def terminate(_signal, _frame):
            raise KeyboardInterrupt("Memory monitor received a termination signal")

        previous_handler = signal.signal(signal.SIGTERM, terminate)
        try:
            if available < args.start_available_mb:
                raise MemoryError("Insufficient available memory to start safely")
            process = subprocess.Popen(command, start_new_session=True)
            while process.poll() is None:
                available = psutil.virtual_memory().available / mib
                swap_growth = (psutil.swap_memory().used - starting_swap) / mib
                report["minimum_available_mb"] = min(report["minimum_available_mb"], available)
                report["peak_swap_growth_mb"] = max(report["peak_swap_growth_mb"], swap_growth)
                if available < args.minimum_available_mb or swap_growth > args.max_swap_growth_mb:
                    raise MemoryError("System memory reserve or swap-growth limit reached")
                time.sleep(1)
            report.update(
                status="completed" if process.returncode == 0 else "failed",
                returncode=process.returncode,
            )
            return process.returncode
        except (Exception, KeyboardInterrupt) as error:
            report["reason"] = str(error) or type(error).__name__
            if process is not None:
                stop_tree(process)
            print(json.dumps(report), flush=True)
            return 1
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
            json.dump(report, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
