"""Model-backed research agents that can spin up subagents.

Each Forge role (director, data researcher, training engineer, evaluator, governance council)
reasons with its own model call. During that call it may start subagents: focused helpers with
their own brief and the same evidence, run in parallel, whose findings come back to the parent
before it answers. Subagents may start their own helpers down to FORGE_AGENT_DEPTH levels; every
decision is capped at FORGE_AGENT_MAX_CALLS model calls.

Agents advise and propose. Deterministic gates (budgets, provenance, sealed data, integrity and
regression checks) stay authoritative. Final test data is never passed to an agent.

Keys: AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY (Azure's OpenAI-compatible v1 endpoint), or
OPENAI_API_KEY. Models: FORGE_AGENT_MODEL_<ROLE> > FORGE_AGENT_MODEL > FORGE_DIRECTOR_MODEL. Hephaestus
can change a role's model and limits for one campaign, up to fixed ceilings (services/core/team.py).
"""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from services.core import workbench

# Worth trying again: the provider is busy, or the connection dropped. Anything else won't change.
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
MODEL_ATTEMPTS = 4
MAX_RETRY_WAIT = 30
DEPLOYMENTS_SECONDS = 300
_deployments = {}

ROLES = {
    "director": "Research director",
    "data": "Data researcher",
    "trainer": "Training engineer",
    "evaluator": "Experiment evaluator",
    "governance": "Governance council",
}

SPAWN_TOOL = {
    "type": "function",
    "name": "spawn_subagents",
    "description": (
        "Start focused subagents that work in parallel on parts of your task. Each gets its own "
        "brief and the same evidence, and returns a written finding. Use them when the evidence "
        "has separable parts (failure clusters, competing hypotheses, independent checks)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "agents": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Short role, e.g. 'arithmetic-slip analyst'",
                        },
                        "brief": {
                            "type": "string",
                            "description": "Exactly what this subagent must find out",
                        },
                    },
                    "required": ["name", "brief"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["agents"],
        "additionalProperties": False,
    },
    "strict": True,
}

RUN_TOOL = {
    "type": "function",
    "name": "run_python",
    "description": (
        "Run a Python 3 script in your own Linux VM to check the evidence instead of estimating: "
        "count failure patterns, compare answers, recompute a number. The evidence you were given "
        "is at ./evidence.json. The VM has no network. Whatever the script prints comes back to you."
    ),
    "parameters": {
        "type": "object",
        "properties": {"code": {"type": "string", "description": "A complete Python 3 script"}},
        "required": ["code"],
        "additionalProperties": False,
    },
    "strict": True,
}

FINDING = {
    "type": "object",
    "properties": {"finding": {"type": "string"}},
    "required": ["finding"],
    "additionalProperties": False,
}

SUBAGENT_INSTRUCTIONS = (
    "You are a subagent started by Forge's {parent}. Work only on your brief, using only the "
    "evidence given. Cite question ids or numbers from the evidence. Be concrete and brief: at "
    "most 120 words. Never ask for final test data."
)


def configured():
    return bool(os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"))


def model(role):
    return (
        os.getenv(f"FORGE_AGENT_MODEL_{role.upper()}")
        or os.getenv("FORGE_AGENT_MODEL")
        or os.getenv("FORGE_DIRECTOR_MODEL", "gpt-5-mini")
    )


def provider():
    return "Azure OpenAI" if os.getenv("AZURE_OPENAI_API_KEY") else "OpenAI"


def limits(settings=None):
    found = {
        "max_subagents": int(os.getenv("FORGE_AGENT_MAX_SUBAGENTS", "4")),
        "depth": int(os.getenv("FORGE_AGENT_DEPTH", "2")),
        "max_calls": int(os.getenv("FORGE_AGENT_MAX_CALLS", "12")),
    }
    return found | {k: settings[k] for k in found if k in (settings or {})}


def status():
    """What the API reports: which roles reason with a model, and with which one."""
    return {
        "connected": configured(),
        "provider": provider() if configured() else None,
        "models": {role: model(role) for role in ROLES} if configured() else {},
        "subagents": limits() if configured() else None,
        "vms": {"provider": "Freestyle", **workbench.limits()}
        if configured() and workbench.configured()
        else None,
    }


def _connection():
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or "https://api.openai.com/v1"
    key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    headers = {"api-key": key} if os.getenv("AZURE_OPENAI_API_KEY") else {"Authorization": f"Bearer {key}"}
    return endpoint, headers


def _retry_wait(reply, attempt):
    """How long to wait before trying again, as the provider asks when it says."""
    headers = reply.headers if reply is not None else {}
    try:
        if headers.get("retry-after-ms"):
            return min(MAX_RETRY_WAIT, float(headers["retry-after-ms"]) / 1000)
        if headers.get("retry-after"):
            return min(MAX_RETRY_WAIT, float(headers["retry-after"]))
    except ValueError:
        pass
    return min(MAX_RETRY_WAIT, 2.0**attempt)


def responses(endpoint, headers, payload, timeout=120, sleep=time.sleep):
    """One Responses API call, through a busy provider or a dropped connection. A rate limit or a passing
    outage shouldn't cost a research decision or a reply, and a refused request isn't worth repeating."""
    for attempt in range(MODEL_ATTEMPTS):
        last = attempt == MODEL_ATTEMPTS - 1
        try:
            reply = httpx.post(
                endpoint.rstrip("/") + "/responses", headers=headers, json=payload, timeout=timeout
            )
        except httpx.TransportError:
            if last:
                raise
            sleep(_retry_wait(None, attempt))
            continue
        if reply.status_code not in RETRY_STATUS or last:
            reply.raise_for_status()
            return reply.json()
        sleep(_retry_wait(reply, attempt))
    raise RuntimeError("Unreachable")


def _post(payload):
    endpoint, headers = _connection()
    return responses(endpoint, headers, payload)


def deployments(endpoint=None, headers=None):
    """The model names that will really answer on this connection, or None when that can't be read.

    Azure's catalog lists hundreds of models it could deploy, so a name is only good if it's in the
    resource's own deployment list. A wrong name fails every call with DeploymentNotFound."""
    if endpoint is None:
        if not configured():
            return None
        endpoint, headers = _connection()
    key = endpoint.rstrip("/")
    cached = _deployments.get(key)
    if cached and time.monotonic() - cached[0] < DEPLOYMENTS_SECONDS:
        return cached[1]
    try:
        if "api-key" in (headers or {}):
            reply = httpx.get(
                key.split("/openai")[0] + "/openai/deployments",
                params={"api-version": "2023-03-15-preview"},
                headers=headers,
                timeout=20,
            )
        else:
            reply = httpx.get(key + "/models", headers=headers, timeout=20)
        reply.raise_for_status()
        found = sorted(row["id"] for row in reply.json()["data"] if row.get("id"))
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return None
    _deployments[key] = (time.monotonic(), found)
    return found


class Session:
    """One decision: the agent, its subagents and their subagents, with a shared call budget."""

    def __init__(self, role, settings=None):
        self.role = role
        self.limits = limits(settings)
        self.model = (settings or {}).get("model") or model(role)
        self.calls = 0
        self.usage = {"input_tokens": 0, "output_tokens": 0}
        self.nodes = []
        # Calls promised but not yet made: each spawning agent's answer and each started
        # subagent's first call. Delegation can never spend the budget an answer needs.
        self.held = 0
        self._lock = threading.Lock()
        # This campaign's VMs (services/core/workbench.py), or None without Freestyle.
        self.bench = workbench.current.get()

    def _charge(self, raw):
        with self._lock:
            usage = raw.get("usage") or {}
            for k in self.usage:
                self.usage[k] += int(usage.get(k) or 0)

    def _reserve(self, held=False):
        with self._lock:
            if held:
                self.held -= 1
            elif self.calls + self.held >= self.limits["max_calls"]:
                return False
            self.calls += 1
            return True

    def _free(self):
        with self._lock:
            return self.limits["max_calls"] - self.calls - self.held

    def run(self, instructions, evidence, schema, name, depth=0, parent=None, brief=None, prepaid=False):
        node = {
            "agent": name,
            "parent": parent,
            "depth": depth,
            "brief": brief,
            "model": self.model,
            "response_ids": [],
            "children": [],
            "runs_left": self.bench.limits["runs_per_agent"] if self.bench else 0,
        }
        with self._lock:
            self.nodes.append(node)
        # Offering a tool holds this agent's answer-after-tools call at once, so siblings spending
        # the budget can never leave it unable to answer. Released if it answers directly.
        with self._lock:
            free = self.limits["max_calls"] - self.calls - self.held - (0 if prepaid else 1)
            spawn = depth < self.limits["depth"] and free >= 3
            code = node["runs_left"] > 0 and free >= 1
            if spawn or code:
                self.held += 1
                node["answer_held"] = True
        offered = ([SPAWN_TOOL] if spawn else []) + ([RUN_TOOL] if code else [])
        payload = {
            "model": node["model"],
            "max_output_tokens": 4000,
            "instructions": instructions
            + (
                f" You can call spawn_subagents (at most {self.limits['max_subagents']} at a time). "
                "Delegate when the evidence has many items (8 or more failures or examples) or "
                "several independent questions: give each subagent one slice or one question, "
                "then combine their findings in your answer. Answer directly when it is small."
                if spawn
                else ""
            )
            + (
                f" You have your own Linux VM: call run_python (at most {node['runs_left']} runs) to "
                "count or recompute from ./evidence.json instead of estimating, and cite what it printed."
                if code
                else ""
            ),
            "input": json.dumps({"task": brief, "evidence": evidence} if brief else evidence),
            "text": {"format": {"type": "json_schema", "name": "answer", "strict": True, "schema": schema}},
        }
        if offered:
            payload["tools"] = offered
        held = prepaid
        for _round in range(3 + node["runs_left"]):
            if not self._reserve(held) and not (
                node.get("answer_held") and self._answer_now(payload, node) and self._reserve(True)
            ):
                raise RuntimeError(f"{ROLES[self.role]} exceeded {self.limits['max_calls']} model calls")
            raw = _post(payload)
            self._charge(raw)
            node["response_ids"].append(raw.get("id"))
            calls = [o for o in raw.get("output", []) if o.get("type") == "function_call"]
            if not calls and node.pop("answer_held", False):
                with self._lock:
                    self.held -= 1  # Answered without tools: the held call is not needed.
            if not calls:
                text = "".join(
                    c["text"]
                    for o in raw.get("output", [])
                    for c in o.get("content", []) or []
                    if c.get("type") == "output_text"
                )
                if not text:
                    raise RuntimeError(f"{name} returned no answer")
                node["output"] = json.loads(text)
                return node["output"]
            if payload.get("tool_choice") == "none":
                break
            outputs = [
                {
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": json.dumps(self._tool(call, evidence, name, depth, node)),
                }
                for call in calls
            ]
            tools = [t for t in offered if t is RUN_TOOL or not node.get("spawned")]
            payload = {
                "model": node["model"],
                "max_output_tokens": 4000,
                "previous_response_id": raw["id"],
                "input": outputs,
                "text": payload["text"],
                "tools": offered,
            }
            # Another tool round only while code runs remain and a call beyond the held answer is
            # free; otherwise the agent answers now, on the call held for it.
            if code and node["runs_left"] > 0 and self._free() >= 1:
                payload["tools"] = tools
                held = False
            else:
                held = self._answer_now(payload, node)
        raise RuntimeError(f"{name} did not answer")

    @staticmethod
    def _answer_now(payload, node):
        payload["tool_choice"] = "none"
        return node.pop("answer_held", False)

    def _tool(self, call, evidence, name, depth, node):
        if call.get("name") == "spawn_subagents":
            return self._spawn(call, evidence, name, depth, node)
        if call.get("name") == "run_python":
            return self._run_code(call, evidence, name, depth, node)
        return {"error": f"Unknown tool {call.get('name')!r}."}

    def _run_code(self, call, evidence, name, depth, node):
        if not self.bench:
            return {"error": "No VM is available. Answer from the evidence."}
        with self._lock:
            if node["runs_left"] <= 0:
                return {"error": "No runs left. Answer with what you have."}
            node["runs_left"] -= 1
        try:
            code = json.loads(call.get("arguments") or "{}")["code"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return {"error": "Arguments were not valid JSON with a code field."}
        try:
            result = self.bench.run_python(self.role, self.role if depth == 0 else name, code, evidence)
        except Exception as error:  # A VM failure is the agent's to work around, not fatal.
            result = {"error": f"{type(error).__name__}: {error}"[:300]}
        node.setdefault("vm_runs", []).append(
            {"code": code[:4000]} | {k: (v[:2000] if isinstance(v, str) else v) for k, v in result.items()}
        )
        return result

    def _spawn(self, call, evidence, parent, depth, node):
        try:
            wanted = json.loads(call.get("arguments") or "{}").get("agents") or []
        except json.JSONDecodeError:
            return {"error": "Arguments were not valid JSON; answer with what you have."}
        if node.get("spawned"):
            return {"error": "Only one spawn_subagents call per answer."}
        node["spawned"] = True
        with self._lock:
            # This agent's answer is already held; the subagents' first calls are held now.
            room = self.limits["max_calls"] - self.calls - self.held
            wanted = wanted[: max(0, min(self.limits["max_subagents"], room))]
            self.held += len(wanted)
        if not wanted:
            return {
                "error": "No subagents started (none requested, or the call budget is spent). Answer now."
            }
        instructions = SUBAGENT_INSTRUCTIONS.format(parent=ROLES[self.role].lower())

        def work(spec):
            try:
                found = self.run(
                    instructions,
                    evidence,
                    FINDING,
                    spec["name"][:60],
                    depth + 1,
                    parent=parent,
                    brief=spec["brief"][:1000],
                    prepaid=True,
                )
                return {"name": spec["name"][:60], "finding": found["finding"]}
            except Exception as error:  # A failed helper is reported to its parent, not fatal.
                return {"name": spec["name"][:60], "error": f"{type(error).__name__}: {error}"[:300]}

        with ThreadPoolExecutor(max_workers=len(wanted)) as pool:
            results = list(pool.map(work, wanted))
        node["children"].extend(results)
        return {"results": results}

    def transcript(self):
        return {
            "role": self.role,
            "provider": provider(),
            "model": self.model,
            "calls": self.calls,
            "usage": self.usage,
            "limits": self.limits,
            "agents": self.nodes,
            "vm_runs": sum(len(n.get("vm_runs", [])) for n in self.nodes),
        }

    def subagents(self):
        """Helpers started during this decision, in start order, with their briefs and findings."""
        found = {(c["name"], n["agent"]): c for n in self.nodes for c in n["children"]}
        return [
            {
                "name": n["agent"],
                "parent": n["parent"],
                "depth": n["depth"],
                "brief": n["brief"],
                **{k: v for k, v in found.get((n["agent"], n["parent"]), {}).items() if k != "name"},
            }
            for n in self.nodes
            if n["depth"] > 0
        ]


def consult(role, instructions, evidence, schema, settings=None):
    """Runs one agent decision. Returns (answer, session). `settings` are this campaign's for the role."""
    session = Session(role, settings)
    try:
        answer = session.run(instructions, evidence, schema, ROLES[role])
    finally:
        if session.bench:
            session.bench.pause(role)
    return answer, session


# ----- Role briefs used by the campaign runner -------------------------------------------------


def _text(schema_fields):
    return {
        "type": "object",
        "properties": {k: v for k, v in schema_fields.items()},
        "required": list(schema_fields),
        "additionalProperties": False,
    }


DATA = (
    "You are Forge's data researcher. Diagnose why the model fails on development questions, "
    "from its actual answers. Group failures into concrete patterns (for example a misread "
    "quantity, an arithmetic slip, a missing final answer, running out of tokens), count them and "
    "cite question ids. Say which training data would address the largest pattern. You cannot "
    "change the objective or data splits."
)
DATA_SCHEMA = _text(
    {
        "summary": {"type": "string"},
        "patterns": {
            "type": "array",
            "items": _text(
                {
                    "pattern": {"type": "string"},
                    "count": {"type": "integer"},
                    "question_ids": {"type": "array", "items": {"type": "string"}},
                }
            ),
        },
        "recommendation": {"type": "string"},
    }
)

TRAINER = (
    "You are Forge's training engineer. Review a finished LoRA training run from its recorded "
    "loss per step and configuration. Say whether it trained normally (loss fell and settled), "
    "diverged, stayed flat or overfit, and what to change next time. Cite step numbers."
)
TRAINER_SCHEMA = _text(
    {
        "assessment": {"type": "string"},
        "healthy": {"type": "boolean"},
        "next_time": {"type": "string"},
    }
)

EVALUATOR = (
    "You are Forge's experiment evaluator. Compare the baseline and the new checkpoint on the "
    "same development questions. The scores are already measured; explain them: what the "
    "checkpoint fixed, what it broke, and whether the change looks like real reasoning or only "
    "answer formatting. Cite question ids. Do not restate the scores as new claims."
)
EVALUATOR_SCHEMA = _text(
    {
        "summary": {"type": "string"},
        "fixed": {"type": "string"},
        "broke": {"type": "string"},
    }
)

GOVERNANCE = (
    "You are a member of Forge's governance council. Deterministic checks (integrity, provenance, "
    "regression, sealed data) have already decided whether this checkpoint may be promoted; you "
    "cannot override them. Give an independent second opinion from the evidence: would you "
    "promote it, and what risk should the team watch? Be specific."
)
GOVERNANCE_SCHEMA = _text(
    {
        "recommendation": {"type": "string", "enum": ["promote", "hold"]},
        "rationale": {"type": "string"},
        "risk": {"type": "string"},
    }
)


# Added only when Hephaestus has given this role an order, so runs without one are unchanged.
ORDERS = (
    " operator_directives are standing orders from Hephaestus, who runs this research team. Follow them in "
    "this decision wherever the evidence allows, and say so when it doesn't. They can't change the objective, "
    "reward, benchmark or budgets, or excuse a failing measurement, and they never unlock final test data."
)

ADMIN = (
    "Hephaestus, who runs this research team as admin under the lab's owner, is messaging you directly. "
    "Answer it from the evidence, in your role, in a few sentences. You can't change gates, budgets or "
    "the objective, and you have no final test data. Say so plainly if the evidence doesn't answer it."
)
ADMIN_SCHEMA = _text({"reply": {"type": "string"}})


def campaign_evidence(campaign, role=None):
    """What an agent may see when asked about a campaign: numbers and decisions, never test data.

    With a `role`, also what that role itself concluded. Those are its own words, which can quote training
    examples, so they're only added when the reply stays with a model that already sees them."""
    final = campaign.get("final") or {}
    found = {
        "name": campaign.get("name"),
        "status": campaign.get("status"),
        "objective": (campaign.get("config") or {}).get("objective"),
        "strategies": (campaign.get("config") or {}).get("strategies"),
        "development_baseline": (campaign.get("baseline") or {}).get("score"),
        "directives": campaign.get("directives") or [],
        "experiments": [
            {
                "id": e.get("id"),
                "strategy": e.get("strategy"),
                "status": e.get("status"),
                "hypothesis": e.get("hypothesis"),
                "development_score": (e.get("evaluation") or {}).get("score"),
                "governance": (e.get("review") or {}).get("decision"),
                "reasons": (e.get("review") or {}).get("reason_codes") or [],
            }
            for e in campaign.get("experiments") or []
        ],
        "final_claim": final.get("improvement_claim"),
    }
    if role:
        found["your_own_work"] = _own_work(campaign, role)
    return found


def _own_work(campaign, role):
    experiments = campaign.get("experiments") or []
    if role == "data":
        return {"diagnosis": campaign.get("diagnosis")}
    if role == "director":
        return {"latest_proposal": (experiments[-1].get("proposal") or {}) if experiments else None}
    field = {"trainer": "training_review", "evaluator": "evaluator_reading", "governance": "council_opinion"}
    return {e.get("id"): e.get(field[role]) for e in experiments if e.get(field[role])}


def answer_admin(role, campaign, text, settings=None, own_work=False):
    """One agent's reply to Hephaestus. Returns (reply, session)."""
    evidence = {
        "message_from_admin": text,
        "campaign": campaign_evidence(campaign, role if own_work else None),
    }
    answer, session = consult(role, ADMIN, evidence, ADMIN_SCHEMA, settings)
    return answer["reply"], session


def clip(text, n=400):
    text = str(text or "")
    return text if len(text) <= n else text[:n] + "..."


def failures(evaluation, limit=24):
    return [
        {
            "id": p["id"],
            "question": clip(p["question"]),
            "model_answer": clip(p["output"]),
            "expected": clip(p["expected"], 80),
            "finish_reason": p.get("finish_reason"),
        }
        for p in evaluation["predictions"]
        if not p["correct"]
    ][:limit]


def changes(before, after, limit=12):
    pairs = list(zip(before["predictions"], after["predictions"], strict=True))
    row = lambda b, a: {  # noqa: E731
        "id": a["id"],
        "question": clip(a["question"], 300),
        "before": clip(b["output"], 300),
        "after": clip(a["output"], 300),
        "expected": clip(a["expected"], 80),
    }
    return {
        "fixed": [row(b, a) for b, a in pairs if a["correct"] and not b["correct"]][:limit],
        "broke": [row(b, a) for b, a in pairs if b["correct"] and not a["correct"]][:limit],
        "still_wrong": sum(1 for b, a in pairs if not a["correct"] and not b["correct"]),
    }
