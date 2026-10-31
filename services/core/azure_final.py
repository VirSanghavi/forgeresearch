"""Independent Azure evaluator with object-scoped storage grants and no inbound access."""

import json
import os
import shlex
import subprocess
import tarfile
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
import httpx
from services.core import store
from forge_training.artifacts import file_sha256 as sha


def ensure_worker(worker, cancelled):
    """Wake only the lab-owned evaluator, with a fresh bounded shutdown schedule."""
    if worker.get("resource_group") != "forge-eval-338fedca" or worker.get("name") != "forge-evaluator":
        raise ValueError("Evaluator resource is outside the authorized Forge worker")

    def az(*arguments):
        response = subprocess.run(
            ["az", "vm", *arguments, "-g", worker["resource_group"], "-n", worker["name"], "-o", "json"],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if response.returncode:
            raise RuntimeError(
                "Azure evaluator startup failed. Check the configured subscription and VM access."
            )
        return json.loads(response.stdout) if response.stdout.strip() else None

    shutdown = (datetime.now(timezone.utc) + timedelta(hours=2)).strftime("%H%M")
    az("auto-shutdown", "--location", "centralus", "--time", shutdown)
    started = False
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        if cancelled():
            az("deallocate", "--no-wait")
            raise InterruptedError("Evaluator startup cancelled")
        state = az(
            "get-instance-view",
            "--query",
            "{power:instanceView.statuses,agent:instanceView.vmAgent.statuses}",
        )
        running = any(
            s.get("code") == "PowerState/running" for s in (state.get("power") or []) if isinstance(s, dict)
        )
        ready = any(
            s.get("code") == "ProvisioningState/succeeded"
            for s in (state.get("agent") or [])
            if isinstance(s, dict)
        )
        if running and ready:
            return
        if not running and not started:
            az("start")
            started = True
        time.sleep(2)
    raise TimeoutError("Azure evaluator did not become ready within its startup window")


def evaluate(campaign_id, rows, config, adapter, expected_sha, cancelled):
    if not campaign_id.startswith("forge-") or not campaign_id[6:].isalnum():
        raise ValueError("Invalid evaluator job identifier")
    metadata = store.DATA / "azure-worker.json"
    if not metadata.is_file():
        raise RuntimeError("Azure evaluator is not provisioned. Run scripts/azure-worker.py first.")
    worker = json.loads(metadata.read_text())
    if worker["resource_group"] != "forge-eval-338fedca":
        raise ValueError("Evaluator resource is outside the authorized Forge group")
    ensure_worker(worker, cancelled)
    url, key = os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    headers = {"apikey": key, "Authorization": "Bearer " + key}
    progress_path = f"{campaign_id}/independent-progress.json"
    with httpx.Client(headers=headers, timeout=30) as client:
        progress_grant = client.post(
            f"{url}/storage/v1/object/upload/sign/forge-artifacts/{progress_path}",
            headers={"x-upsert": "true"},
            json={},
        )
        progress_grant.raise_for_status()
        progress_url = url + "/storage/v1" + progress_grant.json()["url"]
    folder = store.ARTIFACTS / campaign_id / "independent-final"
    folder.mkdir(parents=True, exist_ok=False)
    job = {
        "campaign_id": campaign_id,
        "rows": rows,
        "config": config,
        "checkpoint_sha256": expected_sha,
        "selection": "frozen before this job was created",
        "progress_upload_url": progress_url,
    }
    (folder / "job.json").write_text(json.dumps(job, indent=2))
    bundle = folder / "input.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        for name in [
            "services/__init__.py",
            "services/core/__init__.py",
            "services/core/contracts.py",
            "services/core/store.py",
            "services/core/protocol.py",
            "services/core/training.py",
            "services/core/final_evaluator.py",
            "services/core/final_job.py",
        ]:
            archive.add(store.ROOT / name, arcname=name)
        for file in (store.ROOT / "services/training-core/src/forge_training").glob("*.py"):
            archive.add(file, arcname=str(file.relative_to(store.ROOT)))
        archive.add(folder / "job.json", arcname="job/job.json")
        if adapter:
            for name in ["adapter_model.safetensors", "adapter_config.json"]:
                archive.add(Path(adapter) / name, arcname=f"job/checkpoint/{name}")
    input_path = f"{campaign_id}/independent-input.tar.gz"
    output_path = f"{campaign_id}/independent-output.tar.gz"
    with httpx.Client(headers=headers, timeout=120) as client:
        response = client.post(
            f"{url}/storage/v1/object/forge-artifacts/{input_path}",
            content=bundle.read_bytes(),
            headers={"Content-Type": "application/gzip"},
        )
        response.raise_for_status()
        signed_read = client.post(
            f"{url}/storage/v1/object/sign/forge-artifacts/{input_path}", json={"expiresIn": 7200}
        )
        signed_read.raise_for_status()
        signed_write = client.post(
            f"{url}/storage/v1/object/upload/sign/forge-artifacts/{output_path}", json={}
        )
        signed_write.raise_for_status()
        download_url = url + "/storage/v1" + signed_read.json()["signedURL"]
        upload_url = url + "/storage/v1" + signed_write.json()["url"]
    # Object-specific, expiring grants cross the trust boundary. No account keys.
    remote_script = folder / "dispatch.sh"
    remote_script.write_text(f"""#!/bin/bash
set -eu
test -f /opt/forge-ready
workdir=/home/forge/run-{campaign_id}
mkdir -p "$workdir"
cd "$workdir"
curl -fsSL {shlex.quote(download_url)} -o input.tar.gz
echo {shlex.quote(sha(bundle) + "  input.tar.gz")} | sha256sum -c -
tar -xzf input.tar.gz
set +e
OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false timeout 2700 /opt/forge-evaluator/bin/python -m services.core.final_job job/job.json > job/evaluation.log 2>&1
code=$?
set -e
tar -czf output.tar.gz -C job .
curl -fsS -X PUT -H 'Content-Type: application/gzip' --data-binary @output.tar.gz {shlex.quote(upload_url)}
echo "FORGE_EVALUATOR_EXIT=$code"
""")
    remote_script.chmod(0o600)
    store.event(
        campaign_id,
        "stage",
        "Dispatching the frozen checkpoint to an isolated Azure Linux evaluator using two object-scoped grants.",
        "evaluator",
    )
    command = [
        "az",
        "vm",
        "run-command",
        "invoke",
        "--resource-group",
        worker["resource_group"],
        "--name",
        worker["name"],
        "--command-id",
        "RunShellScript",
        "--scripts",
        "@" + str(remote_script),
        "-o",
        "json",
    ]
    with (
        (folder / "azure-command.json").open("w") as log,
        (folder / "azure-command-error.log").open("w") as errors,
    ):
        process = subprocess.Popen(command, stdout=log, stderr=errors)
        began = time.monotonic()
        while process.poll() is None:
            if cancelled() or time.monotonic() - began > 3000:
                process.terminate()
                # The VM belongs exclusively to this service. Deallocation enforces cancellation
                # even though Azure action Run Command itself cannot cancel an active script.
                subprocess.run(
                    [
                        "az",
                        "vm",
                        "deallocate",
                        "-g",
                        worker["resource_group"],
                        "-n",
                        worker["name"],
                        "--no-wait",
                    ],
                    capture_output=True,
                    timeout=60,
                )
                raise InterruptedError("Independent evaluator stopped at its compute boundary")
            try:
                update = httpx.get(
                    f"{url}/storage/v1/object/forge-artifacts/{progress_path}", headers=headers, timeout=10
                )
                if update.status_code == 200:
                    progress = update.json()
                    if (
                        progress.get("stage") in {"Final baseline", "Final checkpoint"}
                        and progress.get("total") == len(rows)
                        and 0 <= progress.get("done", -1) <= len(rows)
                    ):
                        snapshot = store.get(campaign_id)
                        if snapshot and snapshot.get("progress") != progress:
                            store.patch(campaign_id, lambda s, value=progress: s.update(progress=value))
                            store.event(
                                campaign_id,
                                "evaluation_sample",
                                f"{progress['stage']}: {progress['done']}/{progress['total']} generated by the independent harness.",
                                "evaluator",
                            )
            except (httpx.HTTPError, ValueError, TypeError):
                pass
            time.sleep(2)
    if process.returncode:
        raise RuntimeError(
            "Azure evaluator dispatch failed. Inspect independent-final/azure-command-error.log."
        )
    with httpx.Client(headers=headers, timeout=120) as client:
        response = client.get(f"{url}/storage/v1/object/forge-artifacts/{output_path}")
        response.raise_for_status()
        output = folder / "output.tar.gz"
        output.write_bytes(response.content)
    with tarfile.open(output) as archive:
        archive.extractall(folder, filter="data")
    if not (folder / "result.json").is_file():
        raise RuntimeError("Independent evaluator failed. Its complete evaluation.log has been preserved.")
    result = json.loads((folder / "result.json").read_text())
    if result["input_sha256"] != sha(folder / "job.json"):
        raise ValueError("Independent evaluator input identity did not match")
    return result
