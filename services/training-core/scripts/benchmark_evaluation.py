"""Compare frozen development evaluation variants in fresh, bounded processes."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from forge_training.artifacts import write_json
from forge_training.config import ExperimentConfig, load_config
from forge_training.worker import run_phase, validate_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--expected-hash")
    parser.add_argument("--repeats", type=int, choices=range(1, 11), default=2)
    parser.add_argument("--attention", choices=("eager", "sdpa", "both"), default="both")
    parser.add_argument("--batch-size", type=int, choices=range(2, 17), default=4)
    args = parser.parse_args()
    if bool(args.checkpoint) != bool(args.expected_hash):
        parser.error("--checkpoint and --expected-hash must be provided together")
    config = load_config(args.config)
    validate_inputs(config, check_output=False)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    variants = [
        ("serial-eager", "eager", 1),
        ("serial-sdpa", "sdpa", 1),
        (f"batch{args.batch_size}-eager", "eager", args.batch_size),
        (f"batch{args.batch_size}-sdpa", "sdpa", args.batch_size),
    ]
    if args.attention != "both":
        variants = [v for v in variants if v[1] == args.attention]
    records = []
    baseline_samples = None
    for repeat in range(args.repeats):
        # Alternate order to reduce warm-cache/thermal ordering bias.
        for name, attention, batch_size in variants[:: 1 if repeat % 2 == 0 else -1]:
            key = f"{name}-{repeat + 1}"
            body = config.model_dump() | {
                "attention_implementation": attention,
                "evaluation": config.evaluation.model_dump() | {"batch_size": batch_size},
            }
            frozen = ExperimentConfig.model_validate(body)
            config_path = output / f"{key}.config.json"
            report_path = output / f"{key}.report.json"
            write_json(config_path, frozen.model_dump())
            command = ["evaluate", "--config", str(config_path), "--output", str(report_path)]
            if args.checkpoint:
                command += [
                    "--checkpoint",
                    str(args.checkpoint.resolve()),
                    "--expected-hash",
                    args.expected_hash,
                ]
            started = time.monotonic()
            run_phase(command, config.evaluation.max_seconds, output / f"{key}.log")
            wall_seconds = time.monotonic() - started
            report = json.loads(report_path.read_text())
            samples = report["samples"]
            if baseline_samples is None:
                baseline_samples = samples
            tokens = sum(s["generated_tokens"] for s in samples)
            row = {
                "variant": name,
                "repeat": repeat + 1,
                "wall_seconds": wall_seconds,
                "evaluation_seconds": report["elapsed_seconds"],
                "generated_tokens": tokens,
                "tokens_per_second": tokens / wall_seconds,
                "num_correct": report["num_correct"],
                "num_examples": report["num_examples"],
                "identical_samples_to_first_serial": samples == baseline_samples,
                "changed_predictions": sum(
                    a["prediction"] != b["prediction"]
                    for a, b in zip(samples, baseline_samples, strict=True)
                ),
            }
            records.append(row)
            write_json(output / f"{key}.measurement.json", row)
            print(json.dumps(row), flush=True)
    summary = []
    for name, _, _ in variants:
        runs = [r for r in records if r["variant"] == name]
        summary.append(
            {
                "variant": name,
                "median_wall_seconds": statistics.median(r["wall_seconds"] for r in runs),
                "correct_counts": [r["num_correct"] for r in runs],
                "changed_prediction_counts": [r["changed_predictions"] for r in runs],
            }
        )
    for row in summary:
        row["speedup_vs_serial_reference"] = (
            summary[0]["median_wall_seconds"] / row["median_wall_seconds"]
        )
    write_json(
        output / "benchmark.json",
        {
            "runs": records,
            "summary": summary,
            "scope": (
                "Local development diagnostic; batching/kernels can alter floating-point results"
            ),
        },
    )


if __name__ == "__main__":
    main()
