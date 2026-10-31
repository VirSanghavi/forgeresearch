"""Monitor only our child process group. Never evict caches, delete artifacts, or kill peers."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from .harness_config import ResourceLimits

GIB = 1024**3


def output_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def sample_resources(root: Path, pid: int | None = None, *, count_output: bool = True) -> dict:
    import psutil

    rss = 0
    if pid is not None:
        try:
            process = psutil.Process(pid)
            for child in [process, *process.children(recursive=True)]:
                try:
                    rss += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
        except psutil.NoSuchProcess:
            pass
    return {
        "time": time.time(),
        "rss_bytes": rss,
        "available_bytes": psutil.virtual_memory().available,
        "swap_bytes": psutil.swap_memory().used,
        "disk_free_bytes": shutil.disk_usage(root).free,
        "output_bytes": output_bytes(root) if count_output else 0,
    }


def violation(sample: dict, initial_swap: int, elapsed: float, limits: ResourceLimits):
    checks = [
        (elapsed > limits.max_seconds_per_model, "time limit"),
        (sample["rss_bytes"] > limits.max_process_gib * GIB, "process memory limit"),
        (sample["available_bytes"] < limits.min_available_gib * GIB, "available memory reserve"),
        (sample["swap_bytes"] - initial_swap > limits.max_swap_growth_gib * GIB, "swap growth"),
        (sample["disk_free_bytes"] < limits.min_disk_free_gib * GIB, "disk reserve"),
        (sample["output_bytes"] > limits.max_output_mib * 1024**2, "output size limit"),
    ]
    return next((reason for failed, reason in checks if failed), None)


def preflight_resources(root: Path, limits: ResourceLimits, *, count_output: bool = True) -> dict:
    sample = sample_resources(root, count_output=count_output)
    reason = violation(sample, sample["swap_bytes"], 0, limits)
    if reason:
        raise RuntimeError(f"Resource preflight refused: {reason}")
    return sample


def stop_child(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def supervised(command: list[str], root: Path, label: str, limits: ResourceLimits) -> dict:
    initial = preflight_resources(root, limits)
    environment = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO": str(limits.mps_memory_fraction),
        "PYTORCH_MPS_LOW_WATERMARK_RATIO": str(limits.mps_memory_fraction * 0.8),
    }
    started = time.monotonic()
    peak_rss = initial["rss_bytes"]
    with (
        (root / f"{label}.log").open("x") as log,
        (root / f"{label}.resources.jsonl").open("x") as telemetry,
    ):
        process = subprocess.Popen(
            command, stdout=log, stderr=log, env=environment, start_new_session=True
        )
        try:
            while True:
                sample = sample_resources(root, process.pid)
                peak_rss = max(peak_rss, sample["rss_bytes"])
                telemetry.write(json.dumps(sample) + "\n")
                telemetry.flush()
                reason = violation(
                    sample, initial["swap_bytes"], time.monotonic() - started, limits
                )
                if reason:
                    raise RuntimeError(f"Resource guard stopped {label}: {reason}")
                code = process.poll()
                if code is not None:
                    if code:
                        raise RuntimeError(f"{label} failed (exit {code}); see {label}.log")
                    return {
                        "peak_rss_bytes": peak_rss,
                        "elapsed_seconds": time.monotonic() - started,
                    }
                time.sleep(0.5)
        finally:
            stop_child(process)
