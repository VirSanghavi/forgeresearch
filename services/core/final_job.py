"""Independent evaluator entry point. Receives frozen inputs, never research credentials."""

import json
import sys
import urllib.request
from pathlib import Path
from services.core.final_evaluator import evaluate
from services.core.training import sha


def main(path):
    job_path = Path(path).resolve()
    job = json.loads(job_path.read_text())
    root = job_path.parent
    adapter = root / "checkpoint" if job.get("checkpoint_sha256") else None
    if adapter and sha(adapter / "adapter_model.safetensors") != job["checkpoint_sha256"]:
        raise ValueError("Independent evaluator refused a checkpoint checksum mismatch")

    phase = "Final baseline"

    def progress(done, total, prediction):
        print(json.dumps({"stage": "evaluation", "done": done, "total": total}), flush=True)
        if job.get("progress_upload_url"):
            request = urllib.request.Request(
                job["progress_upload_url"],
                data=json.dumps({"stage": phase, "done": done, "total": total}).encode(),
                headers={"Content-Type": "application/json", "x-upsert": "true"},
                method="PUT",
            )
            try:
                urllib.request.urlopen(request, timeout=10).close()
            except Exception as error:
                print(f"PROGRESS_UPLOAD_FAILED: {type(error).__name__}", flush=True)

    base = evaluate(job["rows"], job["config"], root / "baseline-harness.json", progress, lambda: False)
    phase = "Final checkpoint"
    candidate = (
        evaluate(
            job["rows"], job["config"], root / "candidate-harness.json", progress, lambda: False, adapter
        )
        if adapter
        else base
    )
    (root / "result.json").write_text(
        json.dumps(
            {
                "baseline": base,
                "checkpoint": candidate,
                "input_sha256": sha(job_path),
                "environment": "Azure Linux / isolated CPU evaluator",
                "harness": "lm-evaluation-harness 0.4.13",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main(sys.argv[1])
