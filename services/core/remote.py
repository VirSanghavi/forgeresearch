"""Training and evaluation on a Modal NVIDIA H100, with the same inputs and outputs as local runs.

The orchestrator, Relay, Terse governance and the UI stay on this machine. Only the GPU work
crosses the boundary: rows (and adapter bytes) go out, predictions, metrics and adapter files
come back. The Modal app is ephemeral and lives exactly as long as one campaign worker, so no
deployed endpoint or warm container outlives the run.
"""

import os
import queue
import signal
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2]
GPU = os.environ.get("FORGE_MODAL_GPU", "H100")
CACHE = "/cache"
# Seconds an idle container stays up between calls of one campaign. The app stops with the
# campaign, which terminates the container regardless.
IDLE_SECONDS = 180

# Pinned to the versions in uv.lock so remote numerics match the local stack.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.14.1",
        "transformers==5.21.0",
        "peft==0.21.2",
        "datasets==5.1.0",
        "safetensors==0.8.0",
        "pydantic==2.13.5",
        "numpy==2.3.4",
        "tokenizers==0.23.3",
        "huggingface-hub==2.3.0",
        "accelerate==1.15.0",
    )
    # Triton kernels for Qwen3.5's gated delta (linear attention) layers. Without them transformers
    # runs a pure-torch fallback at about 20 s per 4B answer. Dense models never call them.
    .uv_pip_install("flash-linear-attention==0.5.2", "fla-core==0.5.2", "einops==0.8.1")
    .env(
        {
            "HF_HOME": f"{CACHE}/hf",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "FORGE_EXECUTION_BACKEND": "modal",
        }
    )
)
weights = modal.Volume.from_name("forge-hf-cache", create_if_missing=True)
app = modal.App("forge-training", image=image)

_loaded = {"key": None, "model": None, "tokenizer": None}


def _model(adapter, model_id=None):
    """Keep one model resident per container; reload only when the adapter changes."""
    import tempfile

    from services.core import training

    key = (model_id, adapter["sha256"] if adapter else "base")
    if _loaded["key"] != key:
        if _loaded["model"] is not None:
            training.unload(_loaded["model"])
            _loaded.update(key=None, model=None, tokenizer=None)
        folder = None
        if adapter:
            folder = Path(tempfile.mkdtemp(prefix="adapter-"))
            for name, data in adapter["files"].items():
                (folder / name).write_bytes(data)
            if training.sha(folder / "adapter_model.safetensors") != adapter["sha256"]:
                raise RuntimeError("Adapter checksum changed in transfer; refusing to evaluate")
        model, tokenizer = training.load(folder, model_id)
        weights.commit()
        _loaded.update(key=key, model=model, tokenizer=tokenizer)
    return _loaded["model"], _loaded["tokenizer"]


@app.function(
    gpu=GPU,
    # Greedy decoding one row at a time is bound by Python dispatch, not the GPU. Modal's default
    # fractional CPU made each H100 token about four times slower than the M3, so reserve cores.
    cpu=8.0,
    memory=16384,
    volumes={CACHE: weights},
    timeout=3600,
    scaledown_window=IDLE_SECONDS,
    max_containers=1,
    retries=0,
)
def execute(job, events):
    """Run one evaluation or training job. Progress streams through `events` as it happens."""
    import tempfile

    from forge_training.tasks import GSM8K_TASK, TaskSpec
    from services.core import contracts, training

    import torch

    if job.get("model"):
        contracts.adopt_model(job["model"])
    task = TaskSpec.model_validate(job["task"]) if job.get("task") else GSM8K_TASK
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    device = training.device_info()
    events.put(("device", device))  # The command center names the GPU before the first result.
    if job["kind"] == "evaluate":
        model, tokenizer = _model(job.get("adapter"), job.get("model_id"))
        count = 0
        # Batch size 1, exactly the local protocol: batched greedy decoding changes predictions.
        for prediction in training.generate(model, tokenizer, job["rows"], lambda: False, task=task):
            events.put(("prediction", prediction))
            count += 1
        return {
            "device": device,
            "seconds": time.monotonic() - started,
            "count": count,
            "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }
    if job["kind"] == "train":
        if _loaded["model"] is not None:
            training.unload(_loaded["model"])
            _loaded.update(key=None, model=None, tokenizer=None)
        folder = Path(tempfile.mkdtemp(prefix="checkpoint-"))
        result = training.train(
            job["proposal"],
            job["config"],
            job["rows"],
            folder,
            lambda metric: events.put(("metric", metric)),
            lambda: False,  # A local stop cancels this function call outright.
            task=task,
        )
        files = {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}
        result["peak_gpu_memory_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        return {"result": result, "files": files, "device": device}
    raise ValueError(f"Unknown job kind {job['kind']!r}")


class Session:
    """One ephemeral Modal app per campaign worker. Closing it stops every container."""

    def __init__(self, poll_seconds=0.5):
        self.poll_seconds = poll_seconds
        self._context = None

    def open(self):
        if not (os.environ.get("MODAL_TOKEN_ID") or (Path.home() / ".modal.toml").exists()):
            raise RuntimeError("no Modal credentials (run `modal token new` or set MODAL_TOKEN_ID)")
        self._context = app.run()
        self._context.__enter__()
        # A deadline kill sends SIGTERM; unwind so the app (and its GPU) stops with the worker.
        try:
            signal.signal(signal.SIGTERM, _terminate)
        except ValueError:
            pass  # Not the main thread; the app still stops when the client disconnects.
        return self

    def close(self):
        context, self._context = self._context, None
        if context is not None:
            context.__exit__(None, None, None)

    def call(self, job, cancelled):
        """Yield ("prediction"|"metric", value) events as they arrive, then ("result", value)."""
        if self._context is None:
            raise RuntimeError("Modal session is not open")
        with modal.Queue.ephemeral() as events:
            call = execute.spawn(job, events)
            finished = False
            try:
                while True:
                    try:
                        for item in events.get_many(256, timeout=self.poll_seconds):
                            yield item
                    except queue.Empty:
                        pass
                    if cancelled():
                        raise InterruptedError("Campaign stopped")
                    try:
                        result = call.get(timeout=0)
                        break
                    except modal.exception.OutputExpiredError:
                        raise
                    except (TimeoutError, modal.exception.TimeoutError):
                        continue
                finished = True
            finally:
                if not finished:
                    # Stopped, failed locally, or abandoned: release the GPU now, not at timeout.
                    call.cancel(terminate_containers=True)
            # Every event was enqueued before the function returned; deliver the remainder.
            while batch := events.get_many(256, block=False):
                yield from batch
        yield ("result", result)


def _terminate(_signal, _frame):
    raise SystemExit(143)
