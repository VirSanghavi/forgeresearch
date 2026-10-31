"""Export completed-run evidence using the shared training core's integrity primitives."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def export(campaign_id):
    from dotenv import load_dotenv
    from services.core import store
    from services.core.engine import artifact, governance
    from services.core.cloud import mirror
    from forge_training.artifacts import file_sha256, manifest_sha256, runtime_versions
    from forge_training.scoring import paired_bootstrap

    load_dotenv(ROOT / ".env")
    snapshot = store.get(campaign_id)
    if not snapshot or snapshot["status"] != "completed":
        raise ValueError("Only completed campaigns can produce a final evidence bundle")
    folder = store.ARTIFACTS / campaign_id
    final = snapshot["final"]
    paired = []
    after = {p["id"]: p for p in final["checkpoint"]["predictions"]}
    for before in final["baseline"]["predictions"]:
        candidate = after[before["id"]]
        if before["question"] != candidate["question"] or before["expected"] != candidate["expected"]:
            raise ValueError("Cannot compare different final questions or reference answers")
        paired.append(int(candidate["correct"]) - int(before["correct"]))
    public_job = None
    job = folder / "independent-final/job.json"
    if job.exists():
        private = json.loads(job.read_text())
        public_job = {k: v for k, v in private.items() if k != "progress_upload_url"}
    sources = {
        str(path.relative_to(ROOT)): file_sha256(path)
        for base in ["services/core", "services/actors/src", "services/training-core/src/forge_training"]
        for path in (ROOT / base).glob("*.py")
    }
    evidence = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "exported_at": store.now(),
        "config": snapshot["config"],
        "model_revision": final["checkpoint"]["model_revision"],
        "selected_checkpoint": snapshot.get("best"),
        "experiments": snapshot["experiments"],
        "final": final,
        "paired_comparison": {
            "improved": paired.count(1),
            "regressed": paired.count(-1),
            "paired_bootstrap_95_pp": paired_bootstrap(paired, snapshot["config"]["seed"]),
            "resamples": 2000,
            "implementation": "shared forge_training.scoring.paired_bootstrap",
            "limitation": "A subset estimate with possible pretraining contamination; no broad reasoning or safety claim.",
        },
        "frozen_evaluator_job": public_job,
        "runtime": runtime_versions(),
        "source_sha256": manifest_sha256(sources),
        "source_files": sources,
        "git_commit_at_export": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "governance": json.loads(governance(campaign_id).snapshot()),
        "trust_boundary": "Trusted local orchestrator; deterministic vetoes, but role credentials are not process-isolated.",
    }
    outputs = [
        ("reproducibility.json", evidence, "report", "Reproducibility and paired comparison"),
        (
            "environment-trajectories.json",
            snapshot["environments"],
            "trajectory",
            "All Terse environment trajectories",
        ),
        ("research-trace.json", store.events(campaign_id), "trace", "Correlated research and Relay trace"),
    ]
    for name, value, kind, title in outputs:
        path = folder / name
        if path.exists():
            raise FileExistsError(f"Evidence is immutable: {path}")
        path.write_text(json.dumps(value, indent=2, allow_nan=False))
        artifact(campaign_id, path, kind, title)
    mirror(campaign_id)
    print(json.dumps({"campaign_id": campaign_id, "paired_comparison": evidence["paired_comparison"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    export(parser.parse_args().campaign)
