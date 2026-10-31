"""A Linux VM for each research agent, on Freestyle (freestyle.sh).

With FREESTYLE_API_KEY set, every role in a campaign (director, data researcher, training engineer,
evaluator, governance council) gets its own VM the first time it runs code, and its subagents work in
the same VM, each in its own folder (~/work/<name>). Agents use it through the run_python tool: the evidence they were
given is at ~/work/evidence.json, and whatever the script prints comes back to them.

Nothing is left running. A VM has no network access, pauses after FORGE_VM_IDLE_SECONDS without
activity and again after every decision, deletes itself after FORGE_VM_AUTO_DELETE_SECONDS paused or
when the campaign's deadline passes, and is deleted outright when the campaign ends. Final test data
never reaches it: agents only ever receive development evidence.
"""

import base64
import contextvars
import hashlib
import json
import os
import re
import threading
import time

import httpx

API = "https://api.freestyle.sh"
MAX_STDOUT = 6000
MAX_STDERR = 2000
# Commands run as the unprivileged `ubuntu` user, so the workspace lives in its home folder.
WORK = "/home/ubuntu/work"

# The campaign whose agents are running on this thread. Sessions capture it when they start, so
# subagent threads keep using it.
current = contextvars.ContextVar("workbench", default=None)


def configured():
    return bool(os.getenv("FREESTYLE_API_KEY")) and os.getenv("FORGE_AGENT_VMS", "1") != "0"


def limits():
    return {
        "runs_per_agent": int(os.getenv("FORGE_VM_RUNS_PER_AGENT", "3")),
        "run_seconds": int(os.getenv("FORGE_VM_RUN_SECONDS", "60")),
        "idle_seconds": int(os.getenv("FORGE_VM_IDLE_SECONDS", "60")),
        "auto_delete_seconds": int(os.getenv("FORGE_VM_AUTO_DELETE_SECONDS", "1800")),
    }


class Client:
    """The few Freestyle REST calls Forge needs (https://api.freestyle.sh, API v5)."""

    def __init__(self, key=None, base=API):
        self.key = key or os.environ["FREESTYLE_API_KEY"]
        self.base = base

    def _call(self, method, path, body=None, timeout=60):
        reply = httpx.request(
            method,
            self.base + path,
            headers={"Authorization": f"Bearer {self.key}"},
            json=body,
            timeout=timeout,
        )
        if reply.status_code >= 400:
            # Freestyle's message, without the request: it never carries the key.
            raise RuntimeError(
                f"Freestyle {method} {path.split('?')[0]}: {reply.status_code} {reply.text[:200]}"
            )
        return reply.json() if reply.content else {}

    def create(self, slug, ttl_seconds, idle_seconds, auto_delete_seconds, metadata):
        return self._call(
            "POST",
            "/v5/vms",
            {
                "slug": slug,
                # No rules: the VM reaches nothing, the Internet included.
                "firewall": {"rules": []},
                "idleTimeoutSeconds": idle_seconds,
                "autoDeleteSeconds": auto_delete_seconds,
                "ttlSeconds": ttl_seconds,
                "metadata": metadata,
                # A slug left by a crashed run of the same campaign is taken over, not duplicated.
                "reassignSlug": True,
            },
            timeout=120,
        )

    def exec(self, vm_id, command, timeout_seconds, stdin=None):
        body = {"command": command, "timeoutMs": timeout_seconds * 1000}
        if stdin is not None:
            body["stdin"] = base64.b64encode(stdin).decode()
        return self._call("POST", f"/v5/vms/{vm_id}/exec-await", body, timeout=timeout_seconds + 30)

    def get(self, vm_id):
        return self._call("GET", f"/v5/vms/{vm_id}")

    def pause(self, vm_id):
        return self._call("POST", f"/v5/vms/{vm_id}/pause")

    def delete(self, vm_id):
        return self._call("DELETE", f"/v5/vms/{vm_id}")


def slug(cid, role):
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9-]", "-", f"forge-{cid}-{role}".lower())).strip("-")[:63]


class Workbenches:
    """The VMs of one campaign, one per role, created on first use and deleted by close()."""

    def __init__(self, cid, deadline_seconds, client=None, clock=time.monotonic, on_event=None):
        self.cid = cid
        self.client = client or Client()
        self.limits = limits()
        self.clock = clock
        self.started = clock()
        # Freestyle deletes a VM this long after it is created, whatever happens to this process.
        self.ttl = int(deadline_seconds) + 1800
        self.on_event = on_event or (lambda kind, text, role: None)
        self.vms = {}
        self.locks = {}
        self.evidence = {}
        self._lock = threading.Lock()

    def _role_lock(self, role):
        with self._lock:
            return self.locks.setdefault(role, threading.Lock())

    def _vm(self, role):
        with self._role_lock(role):
            vm = self.vms.get(role)
            if vm is None:
                created = self.client.create(
                    slug(self.cid, role),
                    ttl_seconds=self.ttl,
                    idle_seconds=self.limits["idle_seconds"],
                    auto_delete_seconds=self.limits["auto_delete_seconds"],
                    metadata={"project": "forge", "campaign": self.cid[:63], "role": role},
                )
                vm = self.vms[role] = {
                    "id": created["id"],
                    "slug": created.get("slug"),
                    "role": role,
                    "resources": created.get("resources"),
                    "runs": 0,
                    "seconds": 0.0,
                    "deleted": False,
                }
                self.on_event("workbench", f"Started a Freestyle VM for the {role} ({vm['id']}).", role)
            return vm

    def _ensure_evidence(self, vm, workdir, evidence):
        data = json.dumps(evidence, indent=1, default=str).encode()
        digest = hashlib.sha256(data).hexdigest()
        if self.evidence.get((vm["id"], workdir)) == digest:
            return
        written = self.client.exec(
            vm["id"],
            f"mkdir -p {workdir} && cat > {workdir}/evidence.json",
            self.limits["run_seconds"],
            stdin=data,
        )
        if written.get("statusCode"):
            raise RuntimeError(f"Could not write the evidence: {(written.get('stderr') or '')[:200]}")
        self.evidence[(vm["id"], workdir)] = digest

    def run_python(self, role, agent, code, evidence):
        """Runs `code` for `agent` in `role`'s VM, next to its evidence. Returns what it printed."""
        vm = self._vm(role)
        folder = re.sub(r"[^a-z0-9-]", "-", agent.lower())[:40] or "agent"
        workdir = WORK if agent == role else f"{WORK}/{folder}"
        started = self.clock()
        with self._role_lock(role):
            vm["runs"] += 1
            n = vm["runs"]
        self._ensure_evidence(vm, workdir, evidence)
        result = self.client.exec(
            vm["id"],
            f"cd {workdir} && cat > run-{n}.py && timeout {self.limits['run_seconds']} python3 run-{n}.py",
            self.limits["run_seconds"] + 5,
            stdin=code.encode(),
        )
        seconds = round(self.clock() - started, 2)
        with self._role_lock(role):
            vm["seconds"] += seconds
        return {
            "exit_code": result.get("statusCode"),
            "stdout": (result.get("stdout") or "")[-MAX_STDOUT:],
            "stderr": (result.get("stderr") or "")[-MAX_STDERR:],
            "seconds": seconds,
            "vm": vm["id"],
        }

    def pause(self, role):
        """Called when a decision ends: a paused VM bills only for its disk."""
        vm = self.vms.get(role)
        if vm and not vm["deleted"]:
            try:
                self.client.pause(vm["id"])
            except Exception:
                pass  # The idle timeout pauses it anyway.

    def close(self):
        """Deletes every VM this campaign started. Returns what each one did."""
        summary = []
        for vm in list(self.vms.values()):
            if vm["deleted"]:
                continue
            try:
                self.client.delete(vm["id"])
                vm["deleted"] = True
            except Exception as error:
                # It still deletes itself at its TTL; say so instead of failing the campaign.
                vm["error"] = f"{type(error).__name__}: {error}"[:200]
            summary.append({k: v for k, v in vm.items()})
        if summary:
            runs = sum(v["runs"] for v in summary)
            self.on_event(
                "workbench",
                f"Deleted {sum(v['deleted'] for v in summary)} of {len(summary)} Freestyle VMs after {runs} "
                "agent runs.",
                None,
            )
        return summary
