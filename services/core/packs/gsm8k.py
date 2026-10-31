"""The built-in GSM8K math pack: Forge's original task, kept byte-identical."""

import random

from forge_training.config import Example
from forge_training.data import verify_separation
from forge_training.tasks import GSM8K_TASK

from services.core.packs.spec import Capabilities, Display, Source, TaskPack

DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
TRAIN_ROWS = 384

PACK = TaskPack(
    id="gsm8k",
    kind="qa",
    task=GSM8K_TASK,
    source=Source(kind="hf_parquet", repo="openai/gsm8k", revision=DATASET_REVISION, license="MIT"),
    display=Display(
        name="Mathematical reasoning",
        objective="Improve multi-step mathematical reasoning while preserving answer format and arithmetic regressions.",
        benchmark_label="GSM8K subset",
        scoring_label="forge_numeric_v1, strict final #### numeric line (shared training-core scorer)",
        unit="problems",
    ),
    # A small declared math regression set. This is not a broad safety evaluation.
    regression=(
        {"id": "reg-1", "question": "What is 6 times 7?", "answer": "#### 42"},
        {"id": "reg-2", "question": "What is 81 divided by 9?", "answer": "#### 9"},
        {"id": "reg-3", "question": "What is 125 minus 37?", "answer": "#### 88"},
        {"id": "reg-4", "question": "What is 15 plus 26?", "answer": "#### 41"},
    ),
    regression_title="Arithmetic regression review",
    regression_label="Four arithmetic questions. This is a limited regression check, not a broad safety evaluation.",
    detectors=("arithmetic_slip", "missing_final_answer", "truncated", "user_reported"),
    focus_tags=("arithmetic", "multi_step"),
    limitations=(
        "GSM8K subset only, not the full leaderboard evaluation; exact harness and scorer versions are in each report",
        "Possible base model pretraining contamination",
        "Four math regression questions, not a broad safety claim",
    ),
    capabilities=Capabilities(terse_world=True, grpo=True, dpo=False, lm_eval_final=True, remote=True),
)


def _parquet(split, data_key):
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "openai/gsm8k",
        repo_type="dataset",
        revision=DATASET_REVISION,
        filename=f"main/{split}-00000-of-00001.parquet",
    )
    return load_dataset("parquet", data_files={data_key: path}, split=data_key)


def benchmark_data(config):
    # Training and dev are disjoint slices of the official train split. Test stays sealed.
    ds = _parquet("train", "train")
    ids = list(range(len(ds)))
    random.Random(config["seed"]).shuffle(ids)
    dev_ids = ids[: config["dev_samples"]]
    train_ids = ids[config["dev_samples"] : config["dev_samples"] + TRAIN_ROWS]
    train = [dict(ds[i]) | {"id": f"train-{i}"} for i in train_ids]
    dev = [dict(ds[i]) | {"id": f"train-{i}"} for i in dev_ids]
    verify_separation([Example(**r) for r in train], [Example(**r) for r in dev])

    # Only the final evaluator calls this closure after checkpoint selection is frozen.
    def sealed():
        from services.core import store

        test = _parquet("test", "sealed")
        # Every GSM8K campaign shares one test split, so exposure is counted across all of them.
        campaigns = [s for s in store.campaigns() if s.get("config", {}).get("task_pack", "gsm8k") == PACK.id]
        exposed = {
            p["id"] for s in campaigns for p in s.get("final", {}).get("baseline", {}).get("predictions", [])
        }
        exposed.update(p for campaign in campaigns for p in campaign.get("sealed_ids", []))
        available = [i for i in range(len(test)) if f"test-{i}" not in exposed]
        if len(available) < config["final_samples"]:
            raise ValueError("Insufficient unqueried final examples; refuse test-set reuse")
        rng = random.Random(config["seed"] + 1)
        chosen = rng.sample(available, config["final_samples"])
        return [dict(test[i]) | {"id": f"test-{i}"} for i in chosen]

    return train, dev, sealed
