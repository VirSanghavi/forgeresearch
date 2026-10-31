"""Kill the lab's Terse runtime and print the recovery, the same path as the command center button.

FORGE_CHAOS=1 npm run lab          # in one terminal
.venv/bin/python scripts/chaos.py  # in another: SIGKILL, then wait for the proof
"""

import argparse
import os
import sys
import time
import httpx

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--api", default=os.environ.get("FORGE_API_URL", "http://127.0.0.1:8000"))
parser.add_argument("--campaign", help="Campaign to prove state against (default: the running one)")
args = parser.parse_args()

with httpx.Client(base_url=args.api, timeout=30) as api:
    r = api.post("/chaos/actors/kill", json={"campaign_id": args.campaign} if args.campaign else None)
    if r.status_code != 202:
        sys.exit(f"Refused ({r.status_code}): {r.json().get('detail')}")
    incident = r.json()["incidents"][0]
    kill = incident["kill"]
    print(f"{kill['signal']} pid {kill['pid']} (campaign {incident['campaign_id']})", flush=True)
    while incident["phase"] in {"killed", "fenced"}:
        time.sleep(0.25)
        incident = api.get("/chaos/actors").json()["incidents"][0]
    if incident["phase"] == "failed":
        sys.exit(f"Recovery failed: {incident.get('error')}")
    proof, restart = incident["proof"], incident["restart"]
    print(
        f"Runtime back in {incident['downtime_seconds']:.1f}s "
        f"(pid {restart['pid']} listening after {incident['process_seconds']:.1f}s, "
        f"owner lease fenced {incident['fence_seconds']:.1f}s)"
    )
    same = "state identical" if proof["identical"] == proof["total"] else "some actors advanced"
    print(
        f"{proof['identical']} of {proof['total']} actors restored, {same} (hash {proof['hash_after'][:12]})"
    )
    budget = proof["budget"]
    print(
        f"Budget {budget['consumed_after']:.1f}s settled = sum of leases {budget['settled_sum']:.1f}s; "
        f"double counted: {budget['double_counted']}"
    )
    for _ in range(240):
        incident = api.get("/chaos/actors").json()["incidents"][0]
        if incident.get("resumed"):
            print(f"Campaign resumed: {incident['resumed']['text']}")
            break
        if incident.get("campaign_status") in {"completed", "failed", "cancelled", "interrupted"}:
            break  # no worker left to resume
        time.sleep(0.5)
    print(f"Campaign status: {incident.get('campaign_status')}")
