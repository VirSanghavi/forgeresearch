"""Every training method a campaign may choose, with what it needs to run.

The director, the API's launch checks, the runner and the command center read this one table.
"""

STRATEGIES = {
    "broad_sft": {
        "label": "Broad SFT",
        "director": "broad_sft trains the permitted corpus (for preference tasks: the chosen replies).",
        "backends": ("local", "modal"),
        "kinds": ("qa", "extraction", "preference"),
    },
    "targeted_sft": {
        "label": "Targeted SFT",
        "director": "targeted_sft trains labeled production failures first, then the longer worked examples.",
        "backends": ("local", "modal"),
        "kinds": ("qa", "extraction"),
    },
    "distill_sft": {
        "label": "Teacher distillation",
        "director": "distill_sft trains verified teacher-written solutions to the training questions instead of the original references.",
        "backends": ("local", "modal"),
        "kinds": ("qa", "extraction"),
        "needs": "synthetic_dataset",
    },
    "hard_sft": {
        "label": "Hard examples",
        "director": "hard_sft first measures the base model on the training questions, then trains the ones it misses (cleaned references) mixed with its own verified answers to the rest, with minibatches and a cosine learning rate.",
        "backends": ("local", "modal"),
        "kinds": ("qa",),
    },
    "grpo": {
        "label": "GRPO",
        "director": "grpo runs 8 TRL steps with deterministic Terse rewards and warm-starts the best approved SFT adapter, so its lora_rank MUST match that candidate's.",
        "backends": ("local",),
        "kinds": ("qa",),
        "needs": "grpo",
    },
    "dpo": {
        "label": "DPO alignment",
        "director": "dpo aligns the model to preferred over rejected replies (TRL DPO) and warm-starts the best approved SFT adapter when one exists, so its lora_rank MUST match that candidate's.",
        "backends": ("local",),
        "kinds": ("preference",),
        "needs": "dpo",
    },
}
NAMES = tuple(STRATEGIES)


def label(name):
    return STRATEGIES[name]["label"]


def problems(config, pack, backend):
    """Reasons this campaign cannot run its requested strategies. Empty when it can."""
    found = []
    for name in config["strategies"]:
        strategy = STRATEGIES[name]
        if pack.kind not in strategy["kinds"]:
            found.append(f"{strategy['label']} does not apply to {pack.kind} tasks.")
        if backend not in strategy["backends"]:
            found.append(f"{strategy['label']} runs only on the local backend. Remove it for Modal.")
        needs = strategy.get("needs")
        if needs == "synthetic_dataset" and not config.get("synthetic_dataset"):
            found.append("Teacher distillation needs a prepared teacher dataset.")
        elif needs in ("grpo", "dpo") and not getattr(pack.capabilities, needs):
            found.append(f"{strategy['label']} is not available for the {pack.display.name} task.")
    return found


def listing():
    return [
        {"id": name, "label": s["label"], "backends": list(s["backends"]), "kinds": list(s["kinds"])}
        for name, s in STRATEGIES.items()
    ]
