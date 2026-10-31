"""Where a campaign's model work executes: this machine (local) or a Modal H100 (modal).

FORGE_BACKEND selects it, default local. Both backends take and return exactly what
services.core.training takes and returns, so the runner, governance, artifacts and the command
center cannot tell them apart except through the recorded device provenance.

If Modal is selected but unavailable the campaign fails with a clear error. It falls back to local
execution only when FORGE_BACKEND_FALLBACK=local says so, and then it says so in the evidence.
"""

import json
import os
import time
from pathlib import Path

from services.core import contracts, packs, protocol

BACKENDS = ("local", "modal")
# Files a saved adapter needs to be reloaded elsewhere. Tokenizer files are recreated from the
# pinned base model revision.
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def configured():
    name = os.environ.get("FORGE_BACKEND", "local").strip().lower() or "local"
    if name not in BACKENDS:
        raise ValueError(f"FORGE_BACKEND must be one of {', '.join(BACKENDS)}, not {name!r}")
    return name


def requested_gpu():
    return os.environ.get("FORGE_MODAL_GPU", "H100")


def describe():
    """What the configured backend will run on, without contacting it. Used by /health."""
    if configured() == "modal":
        return {"backend": "modal", "label": f"Modal NVIDIA {requested_gpu()}", "gpu": requested_gpu()}
    return {"backend": "local", "label": None, "gpu": None}


class Local:
    name = "local"

    def __init__(self, on_hardware=None):
        from services.core import training

        self.training = training
        self.hardware = training.device_info()
        if on_hardware:
            on_hardware(self.hardware)

    def load(self, adapter=None, model_id=None):
        return self.training.load(adapter, model_id)

    def unload(self, model):
        self.training.unload(model)

    def evaluate(
        self, model, tokenizer, rows, progress, cancelled, split="development", worlds=None, pack=None
    ):
        if pack is not None and pack.kind == "preference":
            from services.core.preference import score_pairs

            return score_pairs(model, tokenizer, rows, progress, cancelled, split=split, pack=pack)
        return self.training.evaluate(
            model, tokenizer, rows, progress, cancelled, split=split, worlds=worlds, pack=pack
        )

    def train(self, proposal, config, rows, folder, progress, cancelled, pack=None):
        task = (pack or packs.get()).task
        return self.training.train(proposal, config, rows, folder, progress, cancelled, task=task)

    def close(self):
        pass


class RemoteModel:
    """A handle for a model that lives on the remote GPU: the base model or a saved adapter."""

    def __init__(self, adapter=None, model_id=None):
        self.path = str(adapter) if adapter else None
        self.model_id = model_id
        self.adapter = None
        if adapter:
            folder = Path(adapter)
            files = {name: (folder / name).read_bytes() for name in ADAPTER_FILES}
            self.adapter = {"files": files, "sha256": protocol.sha(folder / "adapter_model.safetensors")}


class Remote:
    name = "modal"

    def __init__(self, client, on_hardware=None):
        self.client = client
        self.on_hardware = on_hardware
        self.hardware = {"backend": "modal", "type": "cuda", "name": None, "requested": requested_gpu()}
        if on_hardware:
            on_hardware(self.hardware)

    def _measured(self, device):
        if device and device != self.hardware:
            self.hardware = device
            if self.on_hardware:
                self.on_hardware(device)

    def load(self, adapter=None, model_id=None):
        return RemoteModel(adapter, model_id), None

    def unload(self, model):
        pass

    def evaluate(
        self, model, tokenizer, rows, progress, cancelled, split="development", worlds=None, pack=None
    ):
        pack = pack or packs.get()
        if pack.kind == "preference":
            raise RuntimeError("Preference evaluation runs on the local backend only")
        started = time.monotonic()
        job = {
            "kind": "evaluate",
            "rows": [{k: r[k] for k in ("id", "question", "answer")} for r in rows],
            "adapter": model.adapter,
            "model_id": model.model_id,
            "model": contracts.model_spec(model.model_id),
            # The prompt travels with the job; the checker stays here and scores what returns.
            "task": pack.task.model_dump(mode="json"),
        }
        done = {}

        def stream():
            for kind, value in self.client.call(job, cancelled):
                if kind == "prediction":
                    yield value
                elif kind == "device":
                    self._measured(value)
                elif kind == "result":
                    done.update(value)

        events = stream()
        evaluation = protocol.collect(
            rows, events, progress, split, worlds, started, None, model_id=model.model_id, pack=pack
        )
        for _ in events:  # Drain to the final result; a surplus prediction is a protocol error.
            raise RuntimeError("Remote evaluation returned more predictions than rows")
        if not done:
            raise RuntimeError("Remote evaluation ended without a result")
        self._measured(done["device"])
        evaluation["device"] = done["device"]
        evaluation["gpu_seconds"] = done["seconds"]
        if "peak_gpu_memory_gb" in done:
            evaluation["peak_gpu_memory_gb"] = done["peak_gpu_memory_gb"]
        return evaluation

    def train(self, proposal, config, rows, folder, progress, cancelled, pack=None):
        job = {
            "kind": "train",
            "proposal": proposal,
            "config": config,
            "model": contracts.model_spec(config.get("model_id")),
            "rows": [{k: r[k] for k in ("id", "question", "answer")} for r in rows],
            "task": (pack or packs.get()).task.model_dump(mode="json"),
        }
        done = {}
        for kind, value in self.client.call(job, cancelled):
            if kind == "metric":
                progress(value)
            elif kind == "device":
                self._measured(value)
            elif kind == "result":
                done.update(value)
        if not done:
            raise RuntimeError("Remote training ended without a result")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for name, data in done["files"].items():
            if Path(name).name != name:
                raise RuntimeError(f"Refusing remote file name {name!r}")
            (folder / name).write_bytes(data)
        result = done["result"]
        if protocol.sha(folder / "adapter_model.safetensors") != result["sha256"]:
            raise RuntimeError("Adapter checksum changed in transfer from Modal; checkpoint rejected")
        record = json.loads((folder / "training.json").read_text())
        if record.get("device") != done["device"]:
            raise RuntimeError("Remote training record disagrees with the reported device")
        self._measured(done["device"])
        return result

    def close(self):
        self.client.close()


def open_backend(on_hardware=None, on_notice=None, client=None):
    """Start the configured backend for one campaign worker. Close it when the worker ends."""
    name = configured()
    if name == "local":
        return Local(on_hardware)
    try:
        if client is None:
            from services.core import remote

            client = remote.Session().open()
    except Exception as error:
        if os.environ.get("FORGE_BACKEND_FALLBACK", "").strip().lower() == "local":
            if on_notice:
                on_notice(
                    f"Modal {requested_gpu()} unavailable ({error}). FORGE_BACKEND_FALLBACK=local, "
                    "so this campaign runs on the local device instead."
                )
            return Local(on_hardware)
        raise RuntimeError(
            f"Modal {requested_gpu()} backend unavailable: {error}. "
            "Fix Modal access or set FORGE_BACKEND=local."
        ) from error
    return Remote(client, on_hardware)
