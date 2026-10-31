"""Hard-example training rows: spend steps on the questions the base model actually gets wrong.

A strong model already answers most training questions, so training on every reference teaches
little and pulls it toward the references' terse style. Here the base model answers the training
questions first. Questions it misses train on the reference without its calculator annotations;
questions it gets right train on its own checker-verified answer, which holds its existing
behaviour in place. Only training questions are ever read.
"""

import random
import re

ANNOTATION = re.compile(r"<<[^<>]*>>")
# Rows per evaluation job, so a long mining pass cannot outlive one remote function's timeout.
CHUNK = 100
EASY_PER_HARD = 2


def clean_reference(text):
    """GSM8K marks arithmetic as <<4*20=80>>80. A model has no calculator behind that syntax."""
    return ANNOTATION.sub("", text)


def mine(backend, config, rows, progress, cancelled, pack):
    """The base model's greedy predictions for training rows, in row order."""
    model, tokenizer = backend.load(model_id=config["model_id"])
    predictions = []
    for start in range(0, len(rows), CHUNK):

        def report(done, _total, prediction, offset=start):
            progress(offset + done, len(rows), prediction)

        predictions += backend.evaluate(
            model,
            tokenizer,
            rows[start : start + CHUNK],
            report,
            cancelled,
            split="training mining",
            pack=pack,
        )["predictions"]
    backend.unload(model)
    return predictions


def build(rows, predictions, seed):
    """Training rows in order: each hard row is followed by easy rows, with hard rows repeating."""
    hard, easy = [], []
    for row, prediction in zip(rows, predictions, strict=True):
        if row["id"] != prediction["id"]:
            raise ValueError(f"Prediction {prediction['id']} arrived for row {row['id']}")
        if prediction["correct"] and prediction.get("finish_reason") == "stop":
            easy.append(row | {"answer": prediction["output"]})
        else:
            hard.append(row | {"answer": clean_reference(row["answer"])})
    if not hard:
        raise ValueError("The base model answered every training question; nothing to mine")
    random.Random(seed).shuffle(hard)
    random.Random(seed).shuffle(easy)
    slots = max(len(hard), -(-len(easy) // EASY_PER_HARD))
    mixed = []
    for i in range(slots):
        mixed.append(hard[i % len(hard)])
        mixed.extend(easy[i * EASY_PER_HARD : (i + 1) * EASY_PER_HARD])
    return {
        "rows": mixed,
        "mined": len(rows),
        "hard": len(hard),
        "easy": len(easy),
        "hard_ids": sorted(row["id"] for row in hard),
    }
