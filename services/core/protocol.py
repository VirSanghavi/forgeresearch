"""The evaluation protocol shared by every backend. Pure Python: no torch, no model."""

import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

from forge_training.artifacts import file_sha256
from forge_training.scoring import extract_answer, wilson_interval
from services.core.contracts import model_spec

DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
PROMPT_VERSION = "math_completion_v1"


def sha(path):
    return file_sha256(Path(path))


def extract(text):
    value = extract_answer(text)
    return str(value) if value is not None else ""


def equal(a, b):
    try:
        return Decimal(a) == Decimal(b)
    except InvalidOperation:
        return False


def score(checker, row, prediction):
    """A prediction with its answer, expected answer and correctness from the task's checker."""
    scored = checker.score(prediction["output"], row["answer"])
    head = {k: prediction[k] for k in ("id", "question", "output")}
    rest = {k: v for k, v in prediction.items() if k not in head and k not in scored}
    extra = {"detail": scored["detail"]} if "detail" in scored else {}
    return head | {k: scored[k] for k in ("answer", "expected", "correct")} | rest | extra


def collect(
    rows,
    predictions,
    progress,
    split,
    worlds,
    started,
    device,
    batch_size=1,
    seconds=None,
    model_id=None,
    pack=None,
):
    """Score streamed predictions in row order: checker, Terse worlds, progress and summary stay local."""
    from services.core import packs

    pack = pack or packs.get()
    checker = pack.task.make_checker()
    results = []
    # Not strict: a remote stream carries its final result after the last prediction.
    for i, (row, p) in enumerate(zip(rows, predictions, strict=False)):
        if p["id"] != row["id"]:
            raise ValueError(f"Prediction {p['id']} arrived for row {row['id']}")
        p = score(checker, row, p)
        env = worlds(row, p["answer"], i) if worlds else None
        p = {**p, "correct": env["reward"] == 1 if env is not None else p["correct"]}
        p["environment_id"] = env["id"] if env else None
        results.append(p)
        progress(i + 1, len(rows), p)
    if len(results) != len(rows):
        raise RuntimeError(f"Evaluation returned {len(results)} of {len(rows)} predictions")
    return summary(results, split, started, device, batch_size, seconds, model_id, pack)


def summary(
    results, split, started, device, batch_size=1, seconds=None, model_id=None, pack=None, decoding=None
):
    """The evaluation record every backend and task returns, from scored predictions."""
    from services.core import packs

    pack = pack or packs.get()
    spec = model_spec(model_id)
    n = len(results)
    correct = sum(p["correct"] for p in results)
    return {
        "score": correct / n if n else 0,
        "n": n,
        "correct": correct,
        # Wilson interval for the subset. No claim of statistical significance.
        "ci95": wilson_interval(correct, n),
        "predictions": results,
        "split": split,
        "benchmark": pack.display.benchmark_label,
        "dataset_revision": pack.source.revision,
        "scoring": pack.display.scoring_label,
        "prompt_version": pack.task.prompt_version,
        "decoding": decoding
        or {
            "do_sample": False,
            "max_new_tokens": pack.task.max_new_tokens or spec["max_new_tokens"],
            "batch_size": batch_size,
            **spec["chat_template_kwargs"],
        },
        "seconds": time.monotonic() - started if seconds is None else seconds,
        "model_id": spec["id"],
        "model_revision": spec["revision"],
        "device": device,
    }
