"""Crash recovery for the Terse actor runtime: supervise, kill on request, prove state survived.

Three processes share the files under data/chaos:
  runtime.json     the lab supervisor's current runtime (pid, port, data dir)
  runtime.jsonl    supervisor journal: every exit and restart, with signal and timings
  incidents.json   kills requested through the API, with before/after actor proof
  retries.jsonl    actor calls that needed retries (written by services.core.actor_client)
"""

import base64
import hashlib
import json
import os
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "data" / "chaos"


def read_json(path, default):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)


def read_lines(path):
    rows = []
    try:
        with Path(path).open() as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return rows


def append_line(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(value) + "\n")
        f.flush()
        os.fsync(f.fileno())


def listening(url):
    try:
        return httpx.get(url, timeout=1).status_code < 500
    except httpx.HTTPError:
        return False


def signal_name(returncode):
    if returncode is not None and returncode < 0:
        try:
            return signal.Signals(-returncode).name
        except ValueError:
            return f"signal {-returncode}"
    return None


class RuntimeSupervisor:
    """Owns the `durable-actors dev` process group and restarts it, same data dir, if it dies."""

    def __init__(self, command, url, data_dir, log, env=None, cwd=None, folder=FOLDER, max_restarts=5):
        self.command, self.url, self.data_dir, self.log = command, url, str(data_dir), log
        self.env, self.cwd, self.folder = env, cwd, Path(folder)
        self.max_restarts, self.process, self.restarts = max_restarts, None, []

    def _spawn(self):
        self.process = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            stdout=self.log,
            stderr=self.log,
            start_new_session=True,
            env=self.env,
        )
        return self.process

    def _wait_listening(self, timeout=90):
        for _ in range(int(timeout / 0.1)):
            if self.process.poll() is not None:
                return False
            if listening(self.url):
                return True
            time.sleep(0.1)
        return False

    def _state(self):
        write_json(
            self.folder / "runtime.json",
            {
                "supervisor_pid": os.getpid(),
                "pid": self.process.pid,
                "url": self.url,
                "data_dir": self.data_dir,
                "restarts": len(self.restarts),
                "started_at": time.time(),
            },
        )

    def start(self):
        self._spawn()
        if not self._wait_listening():
            raise RuntimeError("Terse runtime failed to start")
        self._state()
        return self.process

    def poll(self):
        """Call from the supervising loop. Returns a restart record when a restart happened."""
        if self.process is None or self.process.poll() is None:
            return None
        old, returncode, exited_at = self.process.pid, self.process.returncode, time.time()
        # Reap any host processes still in the dead runtime's group before reusing its data dir.
        try:
            os.killpg(old, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        append_line(
            self.folder / "runtime.jsonl",
            {
                "type": "exited",
                "pid": old,
                "returncode": returncode,
                "signal": signal_name(returncode),
                "at": exited_at,
            },
        )
        recent = [r for r in self.restarts if exited_at - r["exited_at"] < 120]
        if len(recent) >= self.max_restarts:
            raise RuntimeError(f"Terse runtime exited {len(recent) + 1} times in two minutes; not restarting")
        self._spawn()
        ready = self._wait_listening()
        record = {
            "type": "restarted" if ready else "restart_failed",
            "old_pid": old,
            "pid": self.process.pid,
            "signal": signal_name(returncode),
            "returncode": returncode,
            "exited_at": exited_at,
            "listening_at": time.time(),
            "restart_seconds": round(time.time() - exited_at, 3),
            "data_dir": self.data_dir,
        }
        self.restarts.append(record)
        if ready:
            self._state()  # before the journal line: a reader that sees the restart can kill the new pid
        append_line(self.folder / "runtime.jsonl", record)
        if not ready:
            raise RuntimeError("Terse runtime did not come back after a restart")
        return record

    def stop(self):
        (self.folder / "runtime.json").unlink(missing_ok=True)
        if self.process and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=15)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                os.killpg(self.process.pid, signal.SIGKILL)


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, TypeError):
        return False


def command_of(pid):
    """Full command line. Linux ps truncates piped output without -ww, so prefer /proc."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        return raw.replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        pass
    try:
        return subprocess.run(
            ["ps", "-ww", "-p", str(pid), "-o", "command="], capture_output=True, text=True, timeout=2
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def runtime_problem(folder=FOLDER):
    """Why there is no supervised runtime to act on, or None when there is one."""
    state = read_json(Path(folder) / "runtime.json", None)
    if not isinstance(state, dict):
        return "no runtime.json"
    if not alive(state.get("supervisor_pid")):
        return f"supervisor {state.get('supervisor_pid')} is not running"
    if not alive(state.get("pid")):
        return f"runtime {state.get('pid')} is not running"
    command = command_of(state["pid"])
    if "durable-actors" not in command or state["data_dir"] not in command:
        return (
            f"pid {state['pid']} is not the durable-actors runtime for {state['data_dir']}: {command[:300]!r}"
        )
    return None


def supervised_runtime(folder=FOLDER):
    """The runtime the lab supervisor owns, or None when nothing would bring it back."""
    if runtime_problem(folder):
        return None
    return read_json(Path(folder) / "runtime.json", None)


def kill_runtime(folder=FOLDER):
    problem = runtime_problem(folder)
    if problem:
        raise RuntimeError(f"No supervised Terse runtime ({problem}). Start the lab with scripts/lab.py.")
    state = read_json(Path(folder) / "runtime.json", None)
    pid = state["pid"]
    if os.getpgid(pid) != pid:
        raise RuntimeError("Terse runtime is not a process group leader; refusing to signal")
    os.killpg(pid, signal.SIGKILL)
    return {"pid": pid, "signal": "SIGKILL", "signum": int(signal.SIGKILL), "at": time.time()}


# Proof: what the actors held before the kill, and what the restarted runtime serves.


def unix(stamp):
    from datetime import datetime

    try:
        return datetime.fromisoformat(stamp).timestamp()
    except (TypeError, ValueError):
        return None


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def campaign_actors(campaign_id, experiments):
    rows = [
        ("ResearchCampaignActor", campaign_id),
        ("GovernanceActor", campaign_id),
        ("ArtifactRegistryActor", campaign_id),
    ]
    return rows + [("ExperimentActor", f"{campaign_id}-{e}") for e in experiments]


def read_actors(client, rows):
    def one(row):
        return client.invoke(row[0], row[1], "snapshot", [])

    with ThreadPoolExecutor(len(rows)) as pool:
        values = list(pool.map(one, rows))
    return [
        {"actor": name, "id": identity, "snapshot": value, "sha256": digest(value)}
        for (name, identity), value in zip(rows, values, strict=True)
    ]


def owner_record(data_dir, actor, identity):
    """Terse's own records in local storage: the current owner epoch, and the latest commit.

    Terse writes one snapshot object per committed invocation (stateVersion, the ownerEpoch that
    committed it, and its requestId) before the call returns. The owner record holds the lease
    and the epoch; a restarted runtime takes ownership with a higher epoch.
    """

    def b64(value):
        return base64.urlsafe_b64encode(value.encode()).decode().rstrip("=")

    def load(path):
        raw = path.read_bytes()
        return json.loads(raw[raw.index(b"{") :])

    key = f"{b64('local')}/{b64(actor)}/{b64(identity)}"
    root = Path(data_dir) / "objects/durable-actors/v3"
    try:
        owner = next(iter(root.glob(f"owners/*/{key}.json")), None)
        commits = [p for p in root.glob(f"snapshots/*/{key}/*/*.json") if p.stem.isdigit()]
        if not owner or not commits:
            return None
        latest = load(max(commits, key=lambda p: int(p.stem)))
        return {
            "epoch": load(owner).get("epoch"),
            "state_version": latest.get("stateVersion"),
            "committed_by_epoch": latest.get("ownerEpoch"),
            "request_id": latest.get("requestId"),
        }
    except (OSError, ValueError):
        return None


def combined(rows):
    return digest("".join(f"{r['actor']}/{r['id']}={r['sha256']}\n" for r in rows))


def compare(before, after):
    """Per-actor identity plus the invariants a double-applied retry would break."""
    after_by_key = {(r["actor"], r["id"]): r for r in after}
    actors = []
    for b in before:
        a = after_by_key.get((b["actor"], b["id"]))
        actors.append(
            {
                "actor": b["actor"],
                "id": b["id"],
                "before": b["sha256"],
                "after": a["sha256"] if a else None,
                "identical": bool(a and a["sha256"] == b["sha256"]),
                "owner_before": b.get("owner"),
                "owner_after": a.get("owner") if a else None,
            }
        )
    campaign_before = json.loads(before[0]["snapshot"])
    campaign_after = json.loads(after[0]["snapshot"])
    settled = sum(
        r.get("actual_seconds", 0)
        for r in campaign_after["reservations"].values()
        if r["status"] == "settled"
    )
    held = [k for k, r in campaign_after["reservations"].items() if r["status"] == "reserved"]
    leases_kept = all(
        campaign_after["reservations"].get(k) == r for k, r in campaign_before["reservations"].items()
    )
    gov_before, gov_after = json.loads(before[1]["snapshot"]), json.loads(after[1]["snapshot"])
    reviews_kept = all(gov_after["reviews"].get(k) == r for k, r in gov_before["reviews"].items())
    return {
        "actors": actors,
        "identical": sum(a["identical"] for a in actors),
        "total": len(actors),
        "hash_before": combined(before),
        "hash_after": combined(after),
        "budget": {
            "consumed_before": campaign_before["consumed_seconds"],
            "consumed_after": campaign_after["consumed_seconds"],
            "settled_sum": settled,
            "max_seconds": campaign_after["config"]["max_seconds"],
            "double_counted": abs(settled - campaign_after["consumed_seconds"]) > 1e-6,
        },
        "leases": {
            "before": len(campaign_before["reservations"]),
            "after": len(campaign_after["reservations"]),
            "held": len(held),
            "unchanged": leases_kept,
        },
        "governance": {
            "reviews_before": len(gov_before["reviews"]),
            "reviews_after": len(gov_after["reviews"]),
            "retained": reviews_kept,
            "promoted_before": gov_before["promoted"],
            "promoted_after": gov_after["promoted"],
        },
        "campaign": {
            "version_before": campaign_before["version"],
            "version_after": campaign_after["version"],
            "decisions_before": len(campaign_before["decisions"]),
            "decisions_after": len(campaign_after["decisions"]),
            "status_after": campaign_after["status"],
        },
    }


class Incidents:
    """Kill, then watch the supervisor bring the runtime back and prove the actor state."""

    def __init__(self, folder=FOLDER, url=None, client_factory=None):
        self.folder = Path(folder)
        self.url = url or os.environ.get("DURABLE_ACTORS_CONTROL_PLANE_URL", "http://127.0.0.1:7100")
        self.client_factory = client_factory
        self.lock = threading.Lock()

    @property
    def path(self):
        return self.folder / "incidents.json"

    def all(self):
        rows = read_json(self.path, [])
        return rows if isinstance(rows, list) else []

    def save(self, incident):
        with self.lock:
            rows = [r for r in self.all() if r["id"] != incident["id"]]
            write_json(self.path, [incident, *rows][:20])

    def active(self):
        return next((r for r in self.all() if r.get("phase") not in {"restored", "failed"}), None)

    def client(self):
        from durable_actors.client import Client

        if self.client_factory:
            return self.client_factory()
        return Client(self.url, http=httpx.Client(timeout=5))

    def begin(self, campaign_id, experiments):
        if self.active():
            raise RuntimeError("A crash test is already in progress")
        problem = runtime_problem(self.folder)
        if problem:
            raise RuntimeError(f"No supervised Terse runtime ({problem}). Start the lab with scripts/lab.py.")
        state = read_json(self.folder / "runtime.json", None)
        rows = campaign_actors(campaign_id, experiments)
        with self.client() as client:
            before = read_actors(client, rows)
        for r in before:
            r["owner"] = owner_record(state["data_dir"], r["actor"], r["id"])
        killed = kill_runtime(self.folder)
        incident = {
            "id": f"crash-{int(killed['at'] * 1000)}",
            "campaign_id": campaign_id,
            "phase": "killed",
            "kill": killed,
            "data_dir": state["data_dir"],
            "before": before,
        }
        self.save(incident)
        threading.Thread(target=self.recover, args=(incident, rows), daemon=True).start()
        return incident

    def recover(self, incident, rows, timeout=180):
        killed_at = incident["kill"]["at"]
        deadline = time.time() + timeout
        try:
            restart = None
            while time.time() < deadline and not restart:
                restart = next(
                    (
                        r
                        for r in read_lines(self.folder / "runtime.jsonl")
                        if r.get("type") == "restarted" and r.get("old_pid") == incident["kill"]["pid"]
                    ),
                    None,
                )
                time.sleep(0.05)
            if not restart:
                raise TimeoutError("The supervisor did not restart the runtime")
            incident.update(phase="fenced", restart=restart)
            self.save(incident)
            after, attempts = None, 0
            with self.client() as client:
                while time.time() < deadline:
                    attempts += 1
                    try:
                        after = read_actors(client, rows)
                        break
                    except Exception:
                        time.sleep(0.1)
            if after is None:
                raise TimeoutError("Actors did not become readable after the restart")
            restored_at = time.time()
            for r in after:
                r["owner"] = owner_record(incident["data_dir"], r["actor"], r["id"])
            proof = compare(incident["before"], after)
            incident.update(
                phase="restored",
                restored_at=restored_at,
                downtime_seconds=round(restored_at - killed_at, 3),
                process_seconds=round(restart["listening_at"] - killed_at, 3),
                fence_seconds=round(restored_at - restart["listening_at"], 3),
                read_attempts=attempts,
                proof=proof,
            )
            incident.pop("before", None)
        except Exception as error:
            incident.update(phase="failed", error=str(error))
        self.save(incident)
        return incident

    def view(self, campaign_lookup=None, events_lookup=None):
        """Incidents enriched with what the campaign did after the restart."""
        retries = read_lines(self.folder / "retries.jsonl")
        rows = []
        for incident in self.all()[:5]:
            item = {k: v for k, v in incident.items() if k != "before"}
            cid, killed_at = incident["campaign_id"], incident["kill"]["at"]
            # The campaign's own calls: not this API process (its reads serve the UI), not the
            # reads that gave up; each one is a call the worker blocked on and then completed.
            item["retries"] = [
                r
                for r in retries
                if r.get("at", 0) >= killed_at
                and str(r.get("actor_id", "")).startswith(cid)
                and (r["role"] == "worker" if "role" in r else r.get("pid") != os.getpid())
                and r.get("outcome") == "completed"
            ][:12]
            if campaign_lookup:
                campaign = campaign_lookup(cid) or {}
                item["campaign_status"] = campaign.get("status")
            if incident.get("restored_at"):
                # Resumed: the worker's first actor call that completed after the kill, or the
                # first campaign event after the restore, whichever came first.
                marks = [
                    {
                        "at": r["at"],
                        "text": f"Worker call {r['actor']}.{r['method']} completed after "
                        f"{r['attempts']} attempts under one request ID",
                    }
                    for r in item["retries"]
                ]
                for e in events_lookup(cid) if events_lookup else []:
                    at = unix(e.get("timestamp"))
                    if at and at > incident["restored_at"]:
                        marks.append({"at": at, "text": e.get("text", ""), "kind": e.get("kind")})
                        break
                item["resumed"] = min(marks, key=lambda m: m["at"]) if marks else None
            rows.append(item)
        return rows
