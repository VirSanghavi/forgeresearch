"""Provision an isolated, bounded CPU evaluator. Never changes existing resources."""

import json
import os
import subprocess
from pathlib import Path
from datetime import datetime, timedelta, timezone

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
GROUP = "forge-eval-338fedca"
REGION = "centralus"
NSG = "forge-evaluator-centralus-nsg"


def az(*args):
    r = subprocess.run(["az", *args, "-o", "json"], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-5000:])
    return json.loads(r.stdout) if r.stdout.strip() else None


def provision():
    DATA.mkdir(exist_ok=True)
    saved = DATA / "azure-worker.json"
    key = DATA / "azure-forge-ed25519"
    try:
        existing = az("vm", "show", "-g", GROUP, "-n", "forge-evaluator", "--show-details")
    except RuntimeError:
        existing = None
    if existing and existing.get("publicIps"):
        if existing.get("powerState") != "VM running":
            az("vm", "start", "-g", GROUP, "-n", "forge-evaluator")
        shutdown = (datetime.now(timezone.utc) + timedelta(hours=2)).strftime("%H%M")
        az(
            "vm",
            "auto-shutdown",
            "-g",
            GROUP,
            "-n",
            "forge-evaluator",
            "--location",
            existing["location"],
            "--time",
            shutdown,
        )
        state = {
            "resource_group": GROUP,
            "name": "forge-evaluator",
            "ip": existing["publicIps"],
            "ssh_key": str(key),
            "size": existing["hardwareProfile"]["vmSize"],
            "auto_shutdown_utc": shutdown,
            "purpose": "Isolated CPU final benchmark, no research credentials",
        }
        saved.write_text(json.dumps(state, indent=2))
        os.chmod(saved, 0o600)
        return state
    if saved.exists():
        state = json.loads(saved.read_text())
        try:
            vm = az("vm", "show", "-g", GROUP, "-n", "forge-evaluator", "--show-details")
            if vm:
                return state
        except RuntimeError:
            pass
    az(
        "group",
        "create",
        "--name",
        GROUP,
        "--location",
        "eastus",
        "--tags",
        "purpose=forge-ephemeral-evaluation",
        "owner=forge-lab",
    )
    az("network", "nsg", "create", "-g", GROUP, "-n", NSG, "--location", REGION)
    if not key.exists():
        subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-f", str(key), "-N", "", "-C", "forge-ephemeral-evaluator"],
            check=True,
            capture_output=True,
        )
    cloud_init = DATA / "forge-cloud-init.yaml"
    cloud_init.write_text("""#cloud-config
package_update: true
packages:
  - python3-venv
  - python3-pip
runcmd:
  - [python3, -m, venv, /opt/forge-evaluator]
  - [/opt/forge-evaluator/bin/pip, install, torch==2.14.1, --index-url, https://download.pytorch.org/whl/cpu]
  - [/opt/forge-evaluator/bin/pip, install, transformers==5.21.0, peft==0.21.2, 'lm-eval[hf]==0.4.13', pydantic==2.13.5, pyyaml]
  - [touch, /opt/forge-ready]
""")
    vm = az(
        "vm",
        "create",
        "-g",
        GROUP,
        "-n",
        "forge-evaluator",
        "--location",
        REGION,
        "--image",
        "Canonical:ubuntu-24_04-lts:server-arm64:latest",
        "--size",
        "Standard_D8ps_v5",
        "--admin-username",
        "forge",
        "--ssh-key-values",
        str(key) + ".pub",
        "--public-ip-sku",
        "Standard",
        "--nsg",
        NSG,
        "--nsg-rule",
        "NONE",
        "--os-disk-size-gb",
        "32",
        "--storage-sku",
        "Standard_LRS",
        "--custom-data",
        str(cloud_init),
    )
    shutdown = (datetime.now(timezone.utc) + timedelta(hours=2)).strftime("%H%M")
    az("vm", "auto-shutdown", "-g", GROUP, "-n", "forge-evaluator", "--location", REGION, "--time", shutdown)
    state = {
        "resource_group": GROUP,
        "name": "forge-evaluator",
        "ip": vm["publicIpAddress"],
        "ssh_key": str(key),
        "size": "Standard_D8ps_v5",
        "auto_shutdown_utc": shutdown,
        "purpose": "Isolated CPU final benchmark, no Azure OpenAI or Supabase credentials",
    }
    saved.write_text(json.dumps(state, indent=2))
    os.chmod(saved, 0o600)
    return state


if __name__ == "__main__":
    print(json.dumps(provision(), indent=2))
