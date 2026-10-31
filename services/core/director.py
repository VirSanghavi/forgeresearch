"""A bounded, evidence-driven research decision. Final test data is never passed here."""

from typing import Literal
import httpx  # noqa: F401 (tests patch httpx.post through this module)
from pydantic import BaseModel, ConfigDict, Field

FOCUS = ("arithmetic", "multi_step", "formatting", "general", "production_failures")


class Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy: Literal["broad_sft", "targeted_sft", "distill_sft", "hard_sft", "grpo", "dpo"]
    learning_rate: float = Field(ge=0.00001, le=0.001)
    lora_rank: Literal[8, 16]
    hypothesis: str = Field(min_length=10, max_length=1000)
    focus: Literal[FOCUS]


def build_input(config, development, previous, production=None, production_train=()):
    context = {
        "objective": config["objective"],
        "development_score": development["score"],
        "failures": [
            {"question": p["question"], "prediction": p["output"], "expected": p["expected"]}
            for p in development["predictions"]
            if not p["correct"]
        ][:6],
        "previous_experiments": previous,
    }
    if production:
        context["production_failures"] = {
            "flagged": production["flagged"],
            "traces": production["traces"],
            "categories": {c["kind"]: c["count"] for c in production["categories"] if c["count"]},
            "labeled_training_rows": len(production_train),
            "note": "targeted_sft trains these labeled production failures first.",
            "examples": [
                {"question": r["question"][:300], "kinds": r["kinds"]} for r in list(production_train)[:4]
            ],
        }
    return context


GSM8K_INSTRUCTIONS = "You are Forge's research director. Choose ONE bounded LoRA experiment based on measured development evidence. You cannot change the objective, reward or benchmark. Never ask for final test data. Use only allowed strategies. broad_sft trains the permitted corpus; targeted_sft reweights its longer multi-step examples. If grpo is allowed, it uses 8 TRL steps with deterministic Terse rewards. Begin with SFT to learn answer format, then compare GRPO after a valid SFT candidate. GRPO warm-starts the best approved SFT adapter, so its lora_rank MUST match that candidate's recorded rank. Return a concrete hypothesis. If production_failures is present, those are failures found in deployment traffic: targeted_sft trains their labeled examples first, and their held-out half measures the fix, so name the failure kinds your experiment addresses. Prefer a different strategy when prior experiments fail. Learning rate 1e-5 to 1e-3, rank 8 or 16."


# Appended to either set of instructions when the director reasons with a model.
DELEGATION = (
    " Before choosing, you may spawn subagents to analyze separable parts of the evidence in "
    "parallel, for example one per failure pattern, one to weigh previous experiments, or one to "
    "check the production failures. Use their findings in your hypothesis. Training is "
    "deterministic: repeating a previous experiment's strategy, learning rate and LoRA rank "
    "reproduces the same weights, so never repeat one."
)


# Added only when a directive exists, so runs without one are unchanged.
DIRECTIVES = (
    " operator_directives are instructions from Hephaestus, who runs this research team. Weigh them "
    "when choosing the experiment. They can't change the objective, reward, benchmark or budgets, or "
    "excuse a failing measurement, and they never unlock final test data."
)


def repeats(proposal, previous):
    """Names the earlier experiment this proposal would reproduce byte for byte, if any."""
    for p in previous:
        if (p.get("strategy"), p.get("learning_rate"), p.get("lora_rank")) == (
            proposal["strategy"],
            proposal["learning_rate"],
            proposal["lora_rank"],
        ):
            return (
                f"the earlier {p['strategy']} run (learning rate {p['learning_rate']}, rank {p['lora_rank']})"
            )
    return None


def instructions(config):
    """The original math instructions, unchanged, unless the campaign uses a new method or task."""
    from services.core import packs, strategies

    pack = packs.get(config.get("task_pack"))
    if pack.id == packs.DEFAULT and set(config["strategies"]) <= {"broad_sft", "targeted_sft", "grpo"}:
        return GSM8K_INSTRUCTIONS
    methods = " ".join(strategies.STRATEGIES[name]["director"] for name in config["strategies"])
    return (
        "You are Forge's research director. Choose ONE bounded LoRA experiment based on measured "
        "development evidence. You cannot change the objective, reward or benchmark. Never ask for "
        f"final test data. Task: {pack.display.name}. Objective: {pack.display.objective} Answers are "
        f"scored by {pack.display.scoring_label}. Allowed methods: {methods} Return a concrete "
        "hypothesis. If production_failures is present, those are failures found in deployment "
        "traffic: targeted_sft trains their labeled examples first, and their held-out half measures "
        "the fix, so name the failure kinds your experiment addresses. Prefer a different strategy "
        "when prior experiments fail. Learning rate 1e-5 to 1e-3, rank 8 or 16."
    )


def propose(config, development, previous, context=None, settings=None):
    failures = [p for p in development["predictions"] if not p["correct"]]
    fallback = {
        "strategy": "broad_sft" if not previous else "targeted_sft",
        "learning_rate": 0.0002 if not previous else 0.0001,
        "lora_rank": 8,
        "hypothesis": f"{len(failures)} of {development['n']} development answers are incorrect. Train verified worked examples, then compare an independently loaded adapter.",
        "focus": "general" if not previous else "multi_step",
    }
    production = (context or {}).get("production_failures")
    if production and production["labeled_training_rows"] and not previous:
        kinds = ", ".join(f"{n} {k.replace('_', ' ')}" for k, n in production["categories"].items())
        fallback |= {
            "strategy": "targeted_sft",
            "hypothesis": f"Production flagged {production['flagged']} of {production['traces']} answers ({kinds}). Train the {production['labeled_training_rows']} labeled production failures first, then check the held-out production failures and development set.",
            "focus": "production_failures",
        }
    if "hard_sft" in config["strategies"]:
        # A second run must differ: back off after a drop, otherwise push harder.
        scored = [p["score"] for p in previous if p.get("score") is not None]
        rate = 0.0001 if not scored else 0.00005 if scored[-1] < development["score"] else 0.0002
        fallback |= {
            "strategy": "hard_sft",
            "learning_rate": rate,
            "lora_rank": 16,
            "hypothesis": f"{len(failures)} of {development['n']} development answers are incorrect. Train the training questions the base model misses, holding its verified answers in place, at learning rate {rate}.",
            "focus": "general",
        }
    if fallback["strategy"] not in config["strategies"]:
        fallback["strategy"] = config["strategies"][0]
    from services.core import agents

    settings = settings or {}
    # Hephaestus can stand the director's model down for a campaign; the local director then decides.
    if not agents.configured() or settings.get("enabled") is False:
        return Proposal.model_validate(fallback).model_dump() | {"director": "Evidence-driven local director"}
    schema = {
        "type": "object",
        "properties": {
            "strategy": {"type": "string", "enum": config["strategies"]},
            "learning_rate": {"type": "number"},
            "lora_rank": {"type": "integer", "enum": [8, 16]},
            "hypothesis": {"type": "string"},
            "focus": {"type": "string", "enum": list(FOCUS)},
        },
        "required": ["strategy", "learning_rate", "lora_rank", "hypothesis", "focus"],
        "additionalProperties": False,
    }
    evidence = context if context is not None else build_input(config, development, previous)
    told = DIRECTIVES if isinstance(evidence, dict) and evidence.get("operator_directives") else ""
    answer, session = agents.consult(
        "director", instructions(config) + DELEGATION + told, evidence, schema, settings
    )
    proposal = Proposal.model_validate(answer).model_dump()
    repeated = repeats(proposal, previous)
    if repeated:
        # One chance to choose again, told why; a second repeat is refused rather than rerun.
        answer, retry = agents.consult(
            "director",
            instructions(config) + DELEGATION,
            evidence
            | {
                "rejected_proposal": proposal,
                "rejection": f"Identical to {repeated}; it would reproduce the same weights. "
                "Change the strategy, learning rate or rank.",
            },
            schema,
            settings,
        )
        proposal = Proposal.model_validate(answer).model_dump()
        if repeats(proposal, previous):
            raise ValueError(f"Director repeated {repeated} twice; refusing to rerun identical training")
        retry.nodes = session.nodes + retry.nodes
        retry.calls += session.calls
        for k in retry.usage:
            retry.usage[k] += session.usage[k]
        session = retry
    if proposal["strategy"] not in config["strategies"]:
        raise ValueError("Director selected an unauthorized strategy")
    return proposal | {
        "director": f"{agents.provider()} · {settings.get('model') or agents.model('director')}",
        "response_id": session.nodes[0]["response_ids"][-1],
        "usage": session.usage,
        "subagents": session.subagents(),
        "agent_session": session.transcript(),
    }
