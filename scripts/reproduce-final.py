"""Re-run a previously frozen benchmark without reopening checkpoint selection."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from services.core.final_evaluator import evaluate
    from forge_training.artifacts import file_sha256

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    evidence = json.loads(args.manifest.read_text())
    job = evidence["frozen_evaluator_job"]
    if not job:
        raise ValueError("This run did not use the official independent harness")
    expected = job.get("checkpoint_sha256")
    if expected and (
        not args.checkpoint or file_sha256(args.checkpoint / "adapter_model.safetensors") != expected
    ):
        raise ValueError("Provide the exact frozen adapter; checkpoint checksum mismatch")
    if not expected and args.checkpoint:
        raise ValueError("This campaign selected the baseline; do not substitute an adapter")
    args.output.mkdir(parents=True, exist_ok=False)

    def progress(done, total, _prediction):
        print(f"Historical frozen evaluation: {done}/{total}", flush=True)

    baseline = evaluate(
        job["rows"], job["config"], args.output / "baseline-harness.json", progress, lambda: False
    )
    candidate = (
        evaluate(
            job["rows"],
            job["config"],
            args.output / "candidate-harness.json",
            progress,
            lambda: False,
            args.checkpoint,
        )
        if expected
        else baseline
    )
    (args.output / "reproduced-result.json").write_text(
        json.dumps(
            {
                "campaign_id": job["campaign_id"],
                "purpose": "Historical reproduction, no research or selection",
                "baseline": baseline,
                "checkpoint": candidate,
                "matches_recorded_correct_counts": (
                    baseline["correct"] == evidence["final"]["baseline"]["correct"]
                    and candidate["correct"] == evidence["final"]["checkpoint"]["correct"]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
