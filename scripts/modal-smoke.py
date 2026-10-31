"""Smoke-test the Modal GPU backend end to end: evaluate, train a few steps, reload, evaluate.

    FORGE_BACKEND=modal .venv/bin/python scripts/modal-smoke.py [rows] [steps] [model id]

Spends a few minutes of GPU time. Prints one JSON summary; exits nonzero if any check fails.
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["FORGE_BACKEND"] = "modal"

import modal  # noqa: E402

from services.core import compute, training  # noqa: E402

rows_n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 8
model_id = sys.argv[3] if len(sys.argv) > 3 else training.MODEL_ID
config = {"seed": 42, "dev_samples": 24, "train_steps": steps, "final_samples": 24, "model_id": model_id}
train_rows, dev_rows, _sealed = training.benchmark_data(config)
rows = dev_rows[:rows_n]
timings = {}


def quiet(*_args):
    pass


def never():
    return False


with modal.enable_output():
    started = time.monotonic()
    backend = compute.open_backend()
    timings["open_session"] = time.monotonic() - started
    try:
        base, _ = backend.load(model_id=model_id)
        started = time.monotonic()
        one = backend.evaluate(base, None, rows, quiet, never, split="smoke")
        timings[f"evaluate_{len(rows)}_rows"] = time.monotonic() - started
        proposal = {"strategy": "broad_sft", "lora_rank": 8, "learning_rate": 2e-4}
        folder = Path(tempfile.mkdtemp(prefix="forge-smoke-"))
        started = time.monotonic()
        trained = backend.train(proposal, config, train_rows, folder, quiet, never)
        timings[f"train_{steps}_steps"] = time.monotonic() - started
        adapter, _ = backend.load(folder, model_id)
        started = time.monotonic()
        tuned = backend.evaluate(adapter, None, rows[:4], quiet, never, split="smoke adapter")
        timings["evaluate_adapter_4_rows"] = time.monotonic() - started
    finally:
        backend.close()

record = json.loads((folder / "training.json").read_text())
report = {
    "device": one["device"],
    "training_device": trained["device"],
    "evaluation": {
        "correct": one["correct"],
        "n": one["n"],
        "gpu_seconds": round(one["gpu_seconds"], 1),
        "gpu_seconds_per_row": round(one["gpu_seconds"] / one["n"], 2),
        "peak_gpu_memory_gb": one.get("peak_gpu_memory_gb"),
        "length_truncated": sum(p["finish_reason"] == "length" for p in one["predictions"]),
        "decoding": one["decoding"],
        "model": one["model_id"],
        "answers": [p["answer"] for p in one["predictions"]],
    },
    "samples": [
        {k: p[k] for k in ("id", "answer", "expected", "correct", "finish_reason", "output")}
        for p in one["predictions"][:3]
    ],
    "training": {
        "steps": len(trained["metrics"]),
        "first_loss": trained["metrics"][0]["loss"],
        "last_loss": trained["metrics"][-1]["loss"],
        "before_hash": trained["before_hash"][:16],
        "after_hash": trained["after_hash"][:16],
        "weights_changed": trained["weights_changed"],
        "adapter_sha256": trained["sha256"][:16],
        "local_adapter_sha256": training.sha(folder / "adapter_model.safetensors")[:16],
        "record_device": record["device"]["name"],
        "gpu_seconds": round(trained["seconds"], 2),
        "peak_gpu_memory_gb": trained.get("peak_gpu_memory_gb"),
        "lora_targets": record["lora_targets"],
    },
    "adapter_evaluation": {"correct": tuned["correct"], "n": tuned["n"], "device": tuned["device"]["name"]},
    "finish_reasons": sorted({p["finish_reason"] for p in one["predictions"]}),
    "timings_seconds": {k: round(v, 1) for k, v in timings.items()},
    "adapter_folder": str(folder),
}
print(json.dumps(report, indent=2))
checks = [
    one["device"]["type"] == "cuda",
    "H100" in one["device"]["name"],
    trained["weights_changed"],
    trained["sha256"] == training.sha(folder / "adapter_model.safetensors"),
    record["device"] == trained["device"],
]
sys.exit(0 if all(checks) else 1)
