"""Expand training after reserving development rows in the original seeded partition."""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from forge_training.artifacts import write_json
from forge_training.config import Example
from forge_training.data import verify_separation, write_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--original-train-size", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--dev-size", type=int, default=64)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be a pinned Hub commit")
    if not 1 <= args.original_train_size <= args.train_size or args.dev_size < 1:
        parser.error("Require 1 <= original-train-size <= train-size and dev-size >= 1")
    path = hf_hub_download(
        repo_id="openai/gsm8k",
        repo_type="dataset",
        revision=args.revision,
        filename="main/train-00000-of-00001.parquet",
    )
    dataset = load_dataset("parquet", data_files={"train": path}, split="train")
    if args.train_size + args.dev_size > len(dataset):
        parser.error("Requested sizes exceed the official training split")
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    start = args.original_train_size
    end = start + args.dev_size
    selected = {
        "train": indices[:start] + indices[end : args.train_size + args.dev_size],
        "development": indices[start:end],
    }
    rows = {
        purpose: [Example(id=f"gsm8k-train-{i}", **dict(dataset[i])) for i in positions]
        for purpose, positions in selected.items()
    }
    verify_separation(rows["train"], rows["development"])
    args.output.mkdir(parents=True, exist_ok=False)
    for purpose, examples in rows.items():
        write_dataset(
            args.output,
            purpose,
            examples,
            source="openai/gsm8k",
            revision=args.revision,
            source_split="train",
            license_name="MIT",
        )
    write_json(
        args.output / "partition.json",
        {
            "source": "openai/gsm8k",
            "revision": args.revision,
            "seed": args.seed,
            "original_train_size": start,
            "train_indices": selected["train"],
            "development_indices": selected["development"],
            "official_test_loaded": False,
        },
    )


if __name__ == "__main__":
    main()
