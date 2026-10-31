"""Task packs Forge can run. GSM8K is built in; uploaded packs are registered under data/packs."""

import json
import random
import re

from services.core.packs import gsm8k
from services.core.packs.spec import TaskPack

BUILTIN = {gsm8k.PACK.id: gsm8k.PACK}
DEFAULT = gsm8k.PACK.id
PACK_ID = re.compile(r"^[a-z0-9][a-z0-9-]{1,63}$")
TRAIN_CAP = 2000


def folder(pack_id=None):
    from services.core import store

    root = store.DATA / "packs"
    return root / pack_id if pack_id else root


def get(pack_id=None) -> TaskPack:
    pack_id = pack_id or DEFAULT
    if pack_id in BUILTIN:
        return BUILTIN[pack_id]
    path = folder(pack_id) / "spec.json" if PACK_ID.match(pack_id) else None
    if path is None or not path.exists():
        raise KeyError(f"Unknown task pack {pack_id!r}")
    return TaskPack.model_validate_json(path.read_text())


def regression_pack(pack):
    return get(pack.regression_pack) if pack.regression_pack else pack


def read(pack_id, pool):
    path = folder(pack_id) / f"{pool}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def exposed_ids(pack_id):
    """Sealed ids any campaign on this pack has already queried. They are never reused."""
    from services.core import store

    campaigns = [s for s in store.campaigns() if s.get("config", {}).get("task_pack", DEFAULT) == pack_id]
    seen = {p["id"] for s in campaigns for p in s.get("final", {}).get("baseline", {}).get("predictions", [])}
    return seen | {i for s in campaigns for i in s.get("sealed_ids", [])}


def data(pack, config):
    """(train rows, development rows, sealed loader) for a campaign. Sealed rows load only later."""
    if pack.id == gsm8k.PACK.id:
        return gsm8k.benchmark_data(config)
    rng = random.Random(config["seed"])
    dev_pool, train = read(pack.id, "dev"), read(pack.id, "train")
    dev = rng.sample(dev_pool, min(config["dev_samples"], len(dev_pool)))
    rng.shuffle(train)

    def sealed():
        available = [r for r in read(pack.id, "sealed") if r["id"] not in exposed_ids(pack.id)]
        if len(available) < config["final_samples"]:
            raise ValueError("Insufficient unqueried final examples; refuse test-set reuse")
        return random.Random(config["seed"] + 1).sample(available, config["final_samples"])

    return train[:TRAIN_CAP], dev, sealed


def save(pack, pools, report):
    """Write an immutable pack. Registering identical content twice returns the existing pack."""
    target = folder(pack.id)
    if (target / "spec.json").exists():
        return get(pack.id)
    target.mkdir(parents=True)
    for name, rows in pools.items():
        (target / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (target / "report.json").write_text(json.dumps(report, indent=2))
    (target / "spec.json").write_text(json.dumps(pack.model_dump(mode="json"), indent=2))
    return pack


def summary(pack):
    counts = {}
    if pack.id not in BUILTIN:
        report = json.loads((folder(pack.id) / "report.json").read_text())
        counts = report["split"] | {
            "sealed_remaining": len(
                [r for r in read(pack.id, "sealed") if r["id"] not in exposed_ids(pack.id)]
            )
        }
    return {
        "id": pack.id,
        "kind": pack.kind,
        "builtin": pack.id in BUILTIN,
        "display": pack.display.model_dump(),
        "capabilities": pack.capabilities.model_dump(),
        "regression_title": pack.regression_title,
        "regression_label": pack.regression_label,
        "checker": pack.task.make_checker().checker_id,
        "sha256": pack.sha256,
        "counts": counts,
    }


def available():
    registered = (
        sorted(p.name for p in folder().glob("*/spec.json") for p in [p.parent]) if folder().exists() else []
    )
    return [get(pack_id) for pack_id in [*BUILTIN, *registered]]
