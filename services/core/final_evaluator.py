"""Pinned established harness on an explicitly disclosed, one-shot GSM8K subset."""

import json
import time
from pathlib import Path
import lm_eval
import yaml
from datasets import Dataset
from lm_eval import evaluator
from lm_eval.api.task import ConfigurableTask
from lm_eval.models.huggingface import HFLM
from forge_training.scoring import wilson_interval
from services.core import training


def evaluate(rows, config, output, progress, cancelled, adapter=None):
    # Official task definition, filters, metrics, 5-shot prompt and version are preserved.
    definition_path = Path(lm_eval.__file__).parent / "tasks/gsm8k/gsm8k.yaml"
    definition = yaml.safe_load(definition_path.read_text())
    definition["dataset_kwargs"] = {"revision": training.DATASET_REVISION}
    definition["generation_kwargs"]["max_gen_toks"] = 256

    class SealedTask(ConfigurableTask):
        EVAL_HARNESS_NAME = "gsm8k"

        def test_docs(self):
            # Only the frozen, unqueried subset allocated after checkpoint selection.
            return Dataset.from_list(rows)

    task = SealedTask(config=definition)

    class ObservableModel(HFLM):
        def generate_until(self, requests, disable_tqdm=False):
            result = []
            for i, request in enumerate(requests):
                if cancelled():
                    raise InterruptedError("Final benchmark cancelled at generation boundary")
                result.extend(super().generate_until([request], disable_tqdm=True))
                progress(i + 1, len(requests), {"correct": None})
            return result

    started = time.monotonic()
    model_args = {
        "pretrained": training.MODEL_ID,
        "revision": training.MODEL_REVISION,
        "dtype": "float16" if training.device_name() != "cpu" else "float32",
        "device": training.device_name(),
        "batch_size": 1,
        "trust_remote_code": False,
    }
    if adapter:
        model_args["peft"] = str(adapter)
    model = ObservableModel(**model_args)
    result = evaluator.simple_evaluate(
        model=model,
        tasks=[task],
        num_fewshot=5,
        batch_size=1,
        bootstrap_iters=200,
        log_samples=True,
        random_seed=config["seed"],
        numpy_random_seed=config["seed"],
        torch_random_seed=config["seed"],
        fewshot_random_seed=config["seed"],
    )
    output.write_text(json.dumps(result, indent=2, default=str))
    samples = [s for s in result["samples"]["gsm8k"] if s["filter"] == "strict-match"]
    predictions = []
    for sample in samples:
        doc = sample["doc"]
        predictions.append(
            {
                "id": doc["id"],
                "question": doc["question"],
                "output": sample["resps"][0][0],
                "expected": training.extract(doc["answer"]),
                "answer": str(sample["filtered_resps"][0]),
                "correct": bool(sample["exact_match"]),
                "doc_hash": sample["doc_hash"],
                "prompt_hash": sample["prompt_hash"],
            }
        )
    correct = sum(p["correct"] for p in predictions)
    n = len(predictions)
    score = result["results"]["gsm8k"]["exact_match,strict-match"]
    if abs(score - correct / n) > 1e-9:
        raise ValueError("Official harness aggregate does not match sample-level results")
    training.unload(model.model)
    return {
        "score": score,
        "n": n,
        "correct": correct,
        "ci95": wilson_interval(correct, n),
        "predictions": predictions,
        "split": "sealed final",
        "benchmark": "GSM8K subset / lm-evaluation-harness",
        "harness_version": lm_eval.__version__,
        "task_version": 3.0,
        "task_definition_sha256": training.sha(definition_path),
        "dataset_revision": training.DATASET_REVISION,
        "scoring": "official gsm8k exact_match,strict-match",
        "decoding": {"do_sample": False, "max_new_tokens": 256},
        "num_fewshot": 5,
        "seconds": time.monotonic() - started,
        "model_revision": training.MODEL_REVISION,
    }
