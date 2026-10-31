"""Prepare teacher datasets for a task pack: a background data job, not campaign compute.

A job writes the pack's training questions as a training-core dataset, excludes the development
questions and the sealed question keys, and runs `forge-distill` in a subprocess. The finished
dataset lives at data/packs/<pack>/synthetic/syn-<8 hex>/output and is re-verified from bytes
before any campaign trains on it.
"""

import hashlib
import json
import re
import subprocess
import sys
import uuid

from forge_training.config import Example
from forge_training.data import question_key, write_dataset
from forge_training.distill import DistillConfig, Teacher, is_loopback, probe_teacher, verify

from services.core import packs, store
from services.core.lifecycle import process_alive

SYN_ID = re.compile(r"^syn-[0-9a-f]{8}$")
TERMINAL = {"completed", "cancelled", "failed", "interrupted"}


def folder(pack_id, syn_id=None):
    root = packs.folder(pack_id) / "synthetic"
    return root / syn_id if syn_id else root


def sealed_keys(pack):
    """sha256(question_key) for every sealed question, cached beside the pack."""
    path = packs.folder(pack.id) / "sealed.keys"
    if not path.exists():
        if pack.id == packs.DEFAULT:
            from services.core.packs import gsm8k

            questions = [row["question"] for row in gsm8k._parquet("test", "sealed")]
        else:
            questions = [row["question"] for row in packs.read(pack.id, "sealed")]
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({hashlib.sha256(question_key(q).encode()).hexdigest() for q in questions})
        path.write_text("".join(k + "\n" for k in keys))
    return path


def probe(teacher):
    return probe_teacher(Teacher.model_validate(teacher))


def start(
    pack_id, teacher, sampling=None, budget=None, selection="shortest_correct", seed=42, dev_samples=12
):
    """Write the job inputs and launch forge-distill. Returns the job summary."""
    pack = packs.get(pack_id)
    if pack.kind == "preference":
        raise ValueError("Teacher distillation applies to Q&A and extraction tasks")
    teacher = Teacher.model_validate(teacher)
    if is_loopback(teacher.base_url) and any(s["status"] not in TERMINAL for s in store.campaigns()):
        raise RuntimeError("A campaign is using this machine; a local teacher would compete for memory")
    if running():
        raise RuntimeError("Another teacher job is running")
    syn_id = f"syn-{uuid.uuid4().hex[:8]}"
    job = folder(pack_id, syn_id)
    inputs = job / "inputs"
    inputs.mkdir(parents=True)
    config = {"seed": seed, "dev_samples": dev_samples, "final_samples": 4}
    train, dev, _ = packs.data(pack, config)
    for name, rows, purpose in (("train", train, "train"), ("development", dev, "development")):
        write_dataset(
            inputs,
            name,
            [Example(id=r["id"], question=r["question"], answer=r["answer"]) for r in rows],
            source=f"pack:{pack.id}",
            revision=pack.source.revision,
            source_split="train",
            license_name=pack.source.license,
            purpose=purpose,
        )
    spec = DistillConfig.model_validate(
        {
            "teacher": teacher.model_dump(),
            "task": pack.task.model_dump(mode="json"),
            "prompts": {
                "records": str(inputs / "train.jsonl"),
                "manifest": str(inputs / "train.manifest.json"),
            },
            "exclude": [
                {
                    "records": str(inputs / "development.jsonl"),
                    "manifest": str(inputs / "development.manifest.json"),
                }
            ],
            "exclude_keys": [str(sealed_keys(pack))],
            "sampling": sampling or {},
            "budget": budget or {},
            "selection": selection,
            "output_dir": str(job / "output"),
        }
    )
    (job / "config.json").write_text(json.dumps(spec.model_dump(mode="json"), indent=2))
    log = (job / "distill.log").open("a")
    process = subprocess.Popen(
        [sys.executable, "-m", "forge_training.distill_cli", "run", str(job / "config.json")],
        cwd=store.ROOT,
        stdout=log,
        stderr=log,
        start_new_session=True,
    )
    log.close()
    meta = {
        "id": syn_id,
        "pack": pack.id,
        "pid": process.pid,
        "teacher": teacher.model,
        "created_at": store.now(),
    }
    meta |= {"seed": seed, "dev_samples": dev_samples, "local": is_loopback(teacher.base_url)}
    (job / "job.json").write_text(json.dumps(meta, indent=2))
    return status(pack.id, syn_id)


def status(pack_id, syn_id):
    if not SYN_ID.match(syn_id or ""):
        return None
    job = folder(pack_id, syn_id)
    if not (job / "job.json").exists():
        return None
    meta = json.loads((job / "job.json").read_text())
    record_path = job / "output" / "distillation.json"
    progress = None
    for line in reversed(
        (job / "distill.log").read_text().splitlines() if (job / "distill.log").exists() else []
    ):
        if line.startswith('{"progress"'):
            progress = json.loads(line)["progress"]
            break
    if record_path.exists():
        record = json.loads(record_path.read_text())
        state = "completed" if record["status"] == "complete" else record["status"]
        return meta | {
            "status": state,
            "counts": record["counts"],
            "usage": record["usage"],
            "progress": progress,
        }
    alive = process_alive(meta["pid"], "forge_training.distill_cli")
    tail = (job / "distill.log").read_text()[-600:] if (job / "distill.log").exists() else ""
    return meta | {
        "status": "running" if alive else "failed",
        "progress": progress,
        "error": None if alive else tail,
    }


def running():
    root = packs.folder()
    return [
        s
        for path in (root.glob("*/synthetic/syn-*/job.json") if root.exists() else [])
        if (s := status(path.parent.parent.parent.name, path.parent.name)) and s["status"] == "running"
    ]


def jobs(pack_id):
    root = folder(pack_id)
    found = [status(pack_id, p.name) for p in sorted(root.glob("syn-*"))] if root.exists() else []
    return [j for j in found if j]


def find(pack_id, syn_id):
    """The verified output folder of a finished teacher dataset, or None."""
    job = status(pack_id, syn_id)
    if not job or job["status"] != "completed":
        return None
    return folder(pack_id, syn_id) / "output"


def campaign_rows(pack, syn_id, train_rows, protected_keys):
    """Teacher rows for this campaign's own training questions, with governance evidence.

    Rows whose question is not in this campaign's training set, or that collides with any
    development, production held-out or sealed question, are dropped and counted.
    """
    output = find(pack.id, syn_id)
    checked = verify(output)
    sealed = {line.strip() for line in sealed_keys(pack).read_text().splitlines() if line.strip()}
    allowed = {question_key(r["question"]) for r in train_rows}
    rows, dropped = [], 0
    for line in (output / "train.jsonl").read_text().splitlines():
        row = json.loads(line)
        key = question_key(row["question"])
        hashed = hashlib.sha256(key.encode()).hexdigest()
        if key not in allowed or key in protected_keys or hashed in sealed:
            dropped += 1
            continue
        rows.append({"id": f"syn-{row['id']}", "question": row["question"], "answer": row["answer"]})
    evidence = {
        "dataset_id": syn_id,
        "provenance_sha256": checked["provenance_sha256"],
        "provenance_verified": checked["status"] == "verified",
        "teacher_terms_present": checked["teacher_terms_present"],
        "checker_matches_pack": checked["checker_id"] == pack.task.make_checker().checker_id,
        "separation_verified": checked["separation_verified"],
        "verification": checked["verification"],
        "teacher": checked["teacher"],
        "rows": len(rows),
        "overlap_dropped": dropped,
    }
    return rows, evidence
