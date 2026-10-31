"""Record stand-in production traffic from the real base model, then scan it.

The prompts are a fixed slice of GSM8K's official train split that is never a
campaign's development data (test stays sealed). Outputs are real Qwen2.5-0.5B
generations. Traces carry no answers; the reference answers go to a separate
labels file that plays the role of a human reviewer, used only to audit the
detectors and to label failures for training. No user feedback is invented.

Usage: .venv/bin/python scripts/simulate-production.py [--count 64]
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.core import production, store, training  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--name", default="Simulated production traffic")
    args = parser.parse_args()
    rows = production.simulated_traffic(args.count)
    model, tokenizer = training.load()

    def progress(done, total, _prediction):
        print(f"\r{done}/{total} answered", end="", flush=True)

    result = training.evaluate(model, tokenizer, rows, progress, lambda: False, split="production")
    training.unload(model)
    print()
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    traces = "\n".join(
        json.dumps(
            {
                "id": p["id"],
                "timestamp": stamp,
                "prompt": p["question"],
                "output": p["output"],
                "finish_reason": p["finish_reason"],
            }
        )
        for p in result["predictions"]
    )
    labels = "\n".join(json.dumps({"id": r["id"], "answer": r["answer"]}) for r in rows)
    report = production.save_scan(
        f"{args.name} · {args.count} real Qwen2.5-0.5B answers", traces, labels, source="simulated"
    )
    audit = report["audit"]
    print(f"Scan {report['id']}: {report['flagged']} of {report['traces']} flagged")
    for c in report["categories"]:
        print(f"  {c['count']:>3}  {c['kind']}")
    print(
        f"Actually wrong: {audit['actually_wrong']}. Detector precision {audit['precision']:.2f}, "
        f"recall {audit['recall']:.2f}."
    )
    print(f"Base model accuracy on this traffic: {result['correct']}/{result['n']}")
    print(f"Saved under {store.DATA / 'production'}")


if __name__ == "__main__":
    main()
