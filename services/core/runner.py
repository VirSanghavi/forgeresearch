"""Isolated worker: persisted assignments -> real training -> independent evaluation -> gated promotion."""

import json
import sys
import time
import traceback
from pathlib import Path
from dotenv import load_dotenv
from forge_training.config import Example
from forge_training.data import question_key, verify_separation
from services.core import (
    agents,
    compute,
    distill,
    hard_examples,
    packs,
    production,
    store,
    team as chain,
    training,
    workbench,
)
from services.core.director import Proposal, propose, build_input, repeats
from services.core.briefing import publish
from services.core.verdict import SUMMARY as VERDICT_SUMMARY, final_verdict
from services.core.engine import (
    actor,
    add_review,
    artifact,
    cancelled,
    checkpoint_zip,
    governance,
    react,
    send,
    transition,
)
from services.core.actor_client import get_actor

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def production_split(config, train_rows, dev_rows, pack=None):
    """Labeled production failures: half join training, half are held out to measure the fix."""
    if not config.get("production_scan"):
        return [], [], {"keys": set(), "report": None}
    report = production.load_scan(config["production_scan"])
    if not report:
        raise ValueError("Production scan not found")
    train, held_out = production.campaign_rows(
        report,
        production.load_labels(config["production_scan"]),
        None if pack is None or pack.id == "gsm8k" else pack,
    )
    # Development rows stay pure: production failures that duplicate them are dropped.
    dev_keys = {question_key(r["question"]) for r in dev_rows}
    train = [r for r in train if question_key(r["question"]) not in dev_keys]
    held_out = [r for r in held_out if question_key(r["question"]) not in dev_keys]
    keys = {question_key(r["question"]) for r in train + held_out}
    return train, held_out, {"keys": keys, "report": report}


def run(cid):
    initial = store.get(cid)
    config = initial["config"]
    started = time.monotonic()
    deadline = started + config["max_seconds"]
    folder = store.ARTIFACTS / cid

    def stop():
        return cancelled(cid) or time.monotonic() >= deadline

    def save_field(**values):
        store.patch(cid, lambda s: s.update(values))

    def progress(stage):
        def update(done, total, prediction):
            save_field(progress={"stage": stage, "done": done, "total": total})
            store.event(
                cid,
                "evaluation_sample",
                f"{stage}: {done}/{total} evaluated",
                "evaluator",
                correct=prediction["correct"],
            )

        return update

    def worlds(stage):
        def execute(row, answer, i):
            eid = f"{cid}-{stage}-{i}"
            world = get_actor("RLWorldActor", eid)
            world.reset(eid, row["question"], training.extract(row["answer"]), config["seed"] + i)
            result = json.loads(
                world.step(
                    json.dumps(
                        {
                            "episode_id": eid,
                            "action_id": eid + "-submit",
                            "expected_version": 0,
                            "answer": answer,
                        }
                    )
                )
            )
            result["id"] = eid
            store.patch(cid, lambda s: s.setdefault("environments", []).append(result))
            return result

        return execute

    def team(role, instructions, evidence, schema, expid=None, recipient="director"):
        """One model-backed decision by `role`, which may spin up subagents. Their findings are
        posted to the research room as a thread under the parent's report. Returns the answer, or
        None when no model is configured, Hephaestus has stood the role down, or the call fails
        (deterministic work continues)."""
        if not agents.configured():
            return None
        settings = chain.settings_for(cid, role)
        if settings.get("enabled") is False:
            return None
        orders = chain.read_orders(cid, role, expid or "campaign")
        if orders:
            instructions += agents.ORDERS
            evidence = {**evidence, "operator_directives": orders}
        try:
            answer, session = agents.consult(role, instructions, evidence, schema, settings)
        except Exception as error:
            store.event(cid, "error", f"{agents.ROLES[role]} model unavailable: {error}", role)
            return None
        record_agents(role, session.transcript(), session.subagents(), expid, recipient)
        return answer

    def record_agents(role, transcript, helpers, expid, recipient):
        name = f"{expid or 'campaign'}-{role}-agents.json"
        (folder / name).write_text(json.dumps(transcript, indent=2))
        artifact(cid, folder / name, "agent_transcript", f"{agents.ROLES[role]} model calls and subagents")
        store.patch(
            cid,
            lambda s: s.setdefault("agent_calls", []).append(
                {
                    "role": role,
                    "experiment_id": expid,
                    "model": transcript.get("model") or agents.model(role),
                    "calls": transcript["calls"],
                    "usage": transcript["usage"],
                    "subagents": [h["name"] for h in helpers],
                    "vm_runs": transcript.get("vm_runs", 0),
                }
            ),
        )
        runs = [r | {"agent": n["agent"]} for n in transcript["agents"] for r in n.get("vm_runs", [])]
        if runs:
            first = next((r for r in runs if r.get("stdout")), runs[0])
            printed = (first.get("stdout") or first.get("stderr") or first.get("error") or "").strip()
            send(
                cid,
                role,
                recipient,
                "VM_RUNS",
                f"{agents.ROLES[role]} ran {len(runs)} script{'s' if len(runs) != 1 else ''} in its Freestyle VM"
                + (f". {first['agent']} printed: {agents.clip(printed, 300)}" if printed else "."),
                expid,
                route="room",
            )
        if not helpers:
            return
        parent = send(
            cid,
            role,
            recipient,
            "SUBAGENTS_REPORTED",
            f"{agents.ROLES[role]} ran {len(helpers)} subagents in parallel"
            + (
                f", {sum(h['depth'] > 1 for h in helpers)} of them started by other subagents"
                if any(h["depth"] > 1 for h in helpers)
                else ""
            )
            + ": "
            + ", ".join(h["name"] for h in helpers if h["depth"] == 1)
            + ".",
            expid,
            route="room",
        )
        for h in helpers:
            send(
                cid,
                role,
                recipient,
                "SUBAGENT_FINDING",
                (f"{h['name']} (helping {h['parent']})" if h["depth"] > 1 else h["name"])
                + f": {h.get('finding') or 'failed: ' + h.get('error', 'no finding')}",
                expid,
                route="room",
                reply_to=parent["message_id"],
            )

    def hardware(device):
        save_field(hardware=device)
        if device.get("name"):
            where = "Modal" if device.get("backend") == "modal" else "this machine"
            store.event(cid, "stage", f"Model work runs on {device['name']} ({where}).", "trainer")

    backend = None
    # One Freestyle VM per research agent, made on first use and deleted when the campaign ends.
    bench = (
        workbench.Workbenches(
            cid,
            config["max_seconds"],
            on_event=lambda kind, text, role: store.event(cid, "stage", text, role or "director"),
        )
        if agents.configured() and workbench.configured()
        else None
    )
    bench_token = workbench.current.set(bench)
    try:
        backend = compute.open_backend(
            on_hardware=hardware, on_notice=lambda text: store.event(cid, "error", text, "trainer")
        )
        if backend.name == "local":
            from forge_training.locking import acquire_model_lock

            # One model job per machine: a local teacher or another worker would thrash memory.
            model_lock = acquire_model_lock(f"campaign {cid}", wait=False)  # noqa: F841 (held until exit)
        pack = packs.get(config.get("task_pack"))
        # Terse worlds reward numeric answers; other packs are scored by their checker alone.
        rewarded = worlds if pack.capabilities.terse_world else lambda stage: None
        train_rows, dev_rows, sealed = packs.data(pack, config)
        prod_train, prod_held_out, scan = production_split(config, train_rows, dev_rows, pack)
        train_rows = [{k: r[k] for k in ("id", "question", "answer")} for r in prod_train] + [
            r for r in train_rows if question_key(r["question"]) not in scan["keys"]
        ]

        def example(r):
            return Example(id=r["id"], question=r["question"], answer=r["answer"])

        verify_separation([example(r) for r in train_rows], [example(r) for r in dev_rows + prod_held_out])
        manifest = {
            "dataset": pack.source.repo or pack.id,
            "revision": pack.source.revision,
            "train_ids": [r["id"] for r in train_rows],
            "dev_ids": [r["id"] for r in dev_rows],
            "production_scan": config.get("production_scan"),
            "production_train_ids": [r["id"] for r in prod_train],
            "production_held_out_ids": [r["id"] for r in prod_held_out],
            "test_policy": "One held-out subset, queried only after immutable checkpoint selection",
            "seed": config["seed"],
        }
        (folder / "dataset-manifest.json").write_text(json.dumps(manifest, indent=2))
        artifact(cid, folder / "dataset-manifest.json", "dataset", "Dataset provenance")
        held_rows = [{k: r[k] for k in ("id", "question", "answer")} for r in prod_held_out]
        synthetic_rows, synthetic = [], None
        if config.get("synthetic_dataset"):
            protected = {question_key(r["question"]) for r in dev_rows + held_rows}
            synthetic_rows, synthetic = distill.campaign_rows(
                pack, config["synthetic_dataset"], train_rows, protected
            )
            save_field(synthetic=synthetic)
            send(
                cid,
                "data",
                "director",
                "TEACHER_DATA_VERIFIED",
                f"Teacher dataset {synthetic['dataset_id']} re-verified from bytes: {synthetic['rows']} solutions "
                f"by {synthetic['teacher']['model']} match this campaign's training questions; "
                f"{synthetic['overlap_dropped']} rows dropped for overlap.",
            )
        if scan["report"]:
            r = scan["report"]
            found = ", ".join(
                f"{c['count']} {c['kind'].replace('_', ' ')}" for c in r["categories"] if c["count"]
            )
            save_field(
                production={
                    "scan_id": r["id"],
                    "name": r["name"],
                    "traces": r["traces"],
                    "flagged": r["flagged"],
                    "categories": r["categories"],
                    "audit": r["audit"],
                    "train": [
                        {"id": x["id"], "trace_id": x["trace_id"], "kinds": x["kinds"]} for x in prod_train
                    ],
                    "held_out": [
                        {
                            "id": x["id"],
                            "trace_id": x["trace_id"],
                            "kinds": x["kinds"],
                            "question": x["question"],
                        }
                        for x in prod_held_out
                    ],
                }
            )
            send(
                cid,
                "data",
                "director",
                "PRODUCTION_FAILURES_TRIAGED",
                f"{r['flagged']} of {r['traces']} production answers flagged ({found or 'none'}). "
                f"{len(prod_train)} labeled failures join training; {len(prod_held_out)} are held out to measure the fix.",
            )
        send(
            cid,
            "director",
            "evaluator",
            "BASELINE_REQUESTED",
            f"Evaluate {config['dev_samples']} disjoint development {pack.display.unit}. Keep final test questions sealed.",
        )
        transition(cid, "baseline", "Measuring the original model on development data")
        base_started = time.monotonic()
        actor(cid).reserve(cid + "-baseline", max(1, int(deadline - base_started)))
        model, tokenizer = backend.load(model_id=config["model_id"])
        baseline = backend.evaluate(
            model, tokenizer, dev_rows, progress("Baseline"), stop, worlds=rewarded("baseline"), pack=pack
        )
        # The pack's small declared regression set. This is not a broad safety evaluation.
        regressions = [r.model_dump(exclude_none=True) for r in pack.regression]
        reg_pack = packs.regression_pack(pack)
        base_reg = backend.evaluate(
            model,
            tokenizer,
            regressions,
            progress("Baseline regression"),
            stop,
            split="regression",
            pack=reg_pack,
        )
        base_prod = (
            backend.evaluate(
                model,
                tokenizer,
                held_rows,
                progress("Baseline production"),
                stop,
                split="production held-out",
                pack=pack,
            )
            if held_rows
            else None
        )
        if base_prod:
            save_field(production_baseline=base_prod)
        backend.unload(model)
        actor(cid).settle(cid + "-baseline", time.monotonic() - base_started)
        (folder / "baseline.json").write_text(json.dumps(baseline, indent=2))
        artifact(cid, folder / "baseline.json", "evaluation", "Baseline evaluation")
        save_field(baseline=baseline, baseline_regression=base_reg)
        send(
            cid,
            "evaluator",
            "director",
            "BASELINE_COMPLETED",
            f"Baseline: {baseline['correct']}/{baseline['n']} correct ({baseline['score'] * 100:.1f}%). {baseline['n'] - baseline['correct']} failures available for diagnosis.",
        )
        diagnosis = team(
            "data",
            agents.DATA,
            {
                "task": pack.display.name,
                "objective": config["objective"],
                "baseline": {"correct": baseline["correct"], "n": baseline["n"]},
                "development_failures": agents.failures(baseline),
                "production_failure_kinds": {
                    c["kind"]: c["count"] for c in scan["report"]["categories"] if c["count"]
                }
                if scan["report"]
                else None,
            },
            agents.DATA_SCHEMA,
        )
        if diagnosis:
            save_field(diagnosis=diagnosis)
            send(cid, "data", "director", "FAILURES_DIAGNOSED", diagnosis["summary"])
        previous = []
        best = None
        mined = None

        def holding(held):
            send(
                cid, "director", "hephaestus", "HOLD_ACKNOWLEDGED", f"Holding: {held['reason']}", route="room"
            )

        for index in range(config["max_experiments"]):
            if stop():
                raise InterruptedError("Campaign budget reached or cancellation requested")
            if chain.wait_while_held(cid, stop, holding):
                if stop():
                    raise InterruptedError("Campaign budget reached or cancellation requested")
                send(cid, "director", "hephaestus", "HOLD_RELEASED", "Released. Carrying on.", route="room")
            finish = chain.snapshot(cid)["finish"]
            if finish:
                send(
                    cid,
                    "director",
                    "governance",
                    "RESEARCH_STOPPED",
                    f"Hephaestus: {finish['reason']}",
                    route="room",
                )
                break
            transition(
                cid, "diagnosing", "Research director is selecting the next experiment from measured evidence"
            )
            measured = baseline if not best else best["evaluation"]
            context = build_input(config, measured, previous, scan["report"], prod_train)
            if diagnosis:
                context["data_researcher"] = diagnosis
            expid = f"exp-{index + 1:03d}"
            # Orders committed in Terse before this point are in the evidence, and recorded as read.
            orders = chain.read_orders(cid, "director", expid)
            if orders:
                context["operator_directives"] = orders
            save_field(director_context=context)
            input_file = folder / f"{expid}-director-input.json"
            input_file.write_text(json.dumps(context, indent=2))
            artifact(cid, input_file, "director_input", f"{expid} evidence sent to the director")
            publish(cid)
            assigned = chain.control(cid).take_assignment(expid)
            proposal = json.loads(assigned)["proposal"] if assigned else None
            if proposal and repeats(proposal, previous):
                send(
                    cid,
                    "director",
                    "hephaestus",
                    "ASSIGNMENT_REFUSED",
                    f"The assigned experiment would repeat {repeats(proposal, previous)}, so I chose another.",
                    expid,
                    route="room",
                )
                proposal = None
            if proposal:
                # Hephaestus chose this one. Everything after this point is the same as for any other.
                proposal = Proposal.model_validate(proposal).model_dump() | {
                    "director": "Assigned by Hephaestus"
                }
            else:
                try:
                    proposal = propose(
                        config,
                        measured,
                        previous,
                        context=context,
                        settings=chain.settings_for(cid, "director"),
                    )
                except ValueError as error:
                    if "identical training" not in str(error):
                        raise
                    # No new experiment left to try: stop researching and evaluate what we have.
                    send(cid, "director", "governance", "RESEARCH_STOPPED", f"{error}.", route="room")
                    break
            save_field(director=proposal["director"])
            expid = f"exp-{index + 1:03d}"
            if "agent_session" in proposal:
                session = proposal.pop("agent_session")
                record_agents("director", session, proposal.get("subagents", []), expid, "data")
            key = f"{cid}-{expid}"
            spec = {
                "id": expid,
                "strategy": proposal["strategy"],
                "hypothesis": proposal["hypothesis"],
                "proposal": proposal,
                "status": "proposed",
                "created_at": store.now(),
                "max_steps": min(8, config["train_steps"])
                if proposal["strategy"] == "grpo"
                else config["train_steps"],
                "metrics": [],
            }
            experiment = get_actor("ExperimentActor", key)
            experiment.initialize(json.dumps(spec))
            store.patch(cid, lambda s, value=spec: s.setdefault("experiments", []).append(value))

            def update_exp(_expid=expid, **changes):
                def apply(s):
                    for e in s["experiments"]:
                        if e["id"] == _expid:
                            e.update(changes)

                store.patch(cid, apply)

            # The director's decision is announced to the whole research room.
            send(cid, "director", "data", "EXPERIMENT_PROPOSED", proposal["hypothesis"], expid, route="room")
            review = add_review(
                cid,
                {
                    "id": key + "-preflight",
                    "phase": "preflight",
                    **({"synthetic": synthetic or {}} if proposal["strategy"] == "distill_sft" else {}),
                    "reward_version": pack.reward_version,
                    "task_pack_sha256": pack.sha256,
                    "sealed_data_access": False,
                    "provenance_valid": True,
                },
            )
            if review["decision"] != "approved":
                experiment.update("rejected", json.dumps(review))
                update_exp(status="rejected", review=review)
                continue
            # Reserve a bounded execution window atomically. A single worker owns the M3.
            remaining = max(1, int(deadline - time.monotonic()))
            state = json.loads(actor(cid).snapshot())
            reservation = min(remaining, max(1, config["max_seconds"] - int(state["consumed_seconds"])))
            actor(cid).reserve(key, reservation)
            exp_started = time.monotonic()
            experiment.update("approved", "{}")
            send(
                cid,
                "data",
                "trainer",
                "EXPERIMENT_APPROVED",
                f"Provenance checked. {len(train_rows)} permitted training rows. LoRA rank {proposal['lora_rank']}, learning rate {proposal['learning_rate']}, {config['train_steps']} steps.",
                expid,
            )
            send(
                cid,
                "director",
                "trainer",
                "TRAINING_ASSIGNED",
                f"Run {expid} ({proposal['strategy'].replace('_', ' ')}) within {reservation} reserved seconds. "
                "Report the checkpoint to the evaluator, not to me.",
                expid,
            )
            transition(
                cid,
                "training",
                f"Training {expid}: {proposal['strategy']} with real weight updates",
            )
            experiment.update("training", "{}")
            update_exp(status="training")

            def metric(m, _experiment=experiment, _expid=expid, _update=update_exp, _steps=spec["max_steps"]):
                _experiment.metric(json.dumps(m))
                state = json.loads(_experiment.snapshot())
                _update(metrics=state["metrics"])
                save_field(progress={"stage": _expid, "done": m["step"], "total": _steps})
                store.event(
                    cid,
                    "metric",
                    f"Step {m['step']} · loss {m['loss']:.4f}",
                    "trainer",
                    experiment_id=_expid,
                    metric=m,
                )
                if m["step"] == 1 or m["step"] % max(1, _steps // 4) == 0:
                    publish(cid)

            adapter = folder / expid
            try:
                if proposal["strategy"] == "dpo":
                    if not pack.capabilities.dpo or backend.name != "local":
                        raise RuntimeError("DPO needs a preference task on the local backend")
                    from services.core.preference import train_dpo

                    result = train_dpo(
                        proposal,
                        config,
                        train_rows,
                        adapter,
                        metric,
                        stop,
                        pack,
                        warm_start=best["path"] if best else None,
                    )
                elif proposal["strategy"] == "grpo":
                    if not pack.capabilities.grpo:
                        raise RuntimeError(f"GRPO is not available for the {pack.display.name} task")
                    if backend.name != "local":
                        raise RuntimeError("GRPO runs only on the local backend; choose SFT strategies")
                    from services.core.rl import train_grpo

                    result = train_grpo(
                        cid,
                        expid,
                        proposal,
                        config,
                        train_rows,
                        adapter,
                        metric,
                        stop,
                        warm_start=best["path"] if best else None,
                    )
                    artifact(
                        cid, adapter / "trajectories.json", "trajectory", f"{expid} Terse reward trajectories"
                    )
                else:
                    rows = train_rows
                    if proposal["strategy"] == "distill_sft":
                        # Teacher solutions replace the references; labeled production failures stay first.
                        rows = [r for r in train_rows if r["id"].startswith("prod-")] + synthetic_rows
                    elif proposal["strategy"] == "hard_sft":
                        labeled = [r for r in train_rows if r["id"].startswith("prod-")]
                        if mined is None:
                            plain = [r for r in train_rows if not r["id"].startswith("prod-")]
                            answers = hard_examples.mine(
                                backend, config, plain, progress("Training questions"), stop, pack
                            )
                            mined = hard_examples.build(plain, answers, config["seed"])
                            summary = {k: v for k, v in mined.items() if k != "rows"}
                            (folder / "hard-examples.json").write_text(json.dumps(summary, indent=2))
                            artifact(cid, folder / "hard-examples.json", "dataset", "Hard training questions")
                            send(
                                cid,
                                "data",
                                "trainer",
                                "HARD_EXAMPLES_MINED",
                                f"The base model missed {mined['hard']} of {mined['mined']} training questions. They train on cleaned references; its {mined['easy']} verified answers hold the rest in place.",
                                expid,
                            )
                        rows = labeled + mined["rows"]
                    result = backend.train(proposal, config, rows, adapter, metric, stop, pack=pack)
            except InterruptedError:
                raise
            except Exception as error:
                experiment.update("failed", json.dumps({"error": str(error)}))
                actor(cid).settle(key, min(reservation, time.monotonic() - exp_started))
                update_exp(status="failed", error=str(error))
                send(
                    cid,
                    "trainer",
                    "director",
                    "TRAINING_FAILED",
                    f"{expid} failed: {error}. Partial evidence is preserved. Revise the strategy.",
                    expid,
                )
                previous.append({"strategy": proposal["strategy"], "error": str(error), "score": None})
                continue
            artifact(cid, adapter / "training.json", "training", f"{expid} training configuration")
            weight_artifact = artifact(
                cid, adapter / "adapter_model.safetensors", "checkpoint", f"{expid} LoRA weights"
            )
            artifact(cid, checkpoint_zip(adapter), "checkpoint_zip", f"{expid} checkpoint bundle")
            send(
                cid,
                "trainer",
                "evaluator",
                "TRAINING_COMPLETED",
                f"{expid} saved. Weights changed: {result['weights_changed']}. SHA-256 {result['sha256'][:16]}. Reload independently for evaluation.",
                expid,
            )
            training_review = team(
                "trainer",
                agents.TRAINER,
                {
                    "strategy": proposal["strategy"],
                    "learning_rate": proposal["learning_rate"],
                    "lora_rank": proposal["lora_rank"],
                    "steps": spec["max_steps"],
                    "weights_changed": result["weights_changed"],
                    "loss_by_step": [
                        {"step": m["step"], "loss": round(m["loss"], 4)}
                        for m in json.loads(experiment.snapshot())["metrics"]
                    ],
                },
                agents.TRAINER_SCHEMA,
                expid,
                recipient="evaluator",
            )
            if training_review:
                update_exp(training_review=training_review)
                send(cid, "trainer", "director", "TRAINING_REVIEWED", training_review["assessment"], expid)
            # Governance answers in this checkpoint's thread once the evidence is in.
            checkpoint_post = send(
                cid,
                "trainer",
                "governance",
                "CHECKPOINT_SAVED",
                f"{expid} checkpoint is ready for independent review. SHA-256 {result['sha256'][:16]}.",
                expid,
                route="room",
            )
            transition(cid, "evaluating", "Loading the saved adapter into a fresh model for evaluation")
            experiment.update("evaluating", "{}")
            update_exp(status="evaluating", training=result)
            model, tokenizer = backend.load(adapter, config["model_id"])
            evaluation = backend.evaluate(
                model,
                tokenizer,
                dev_rows,
                progress(expid + " evaluation"),
                stop,
                worlds=rewarded(expid),
                pack=pack,
            )
            reg = backend.evaluate(
                model,
                tokenizer,
                regressions,
                progress(expid + " regression"),
                stop,
                split="regression",
                pack=reg_pack,
            )
            prod_eval = (
                backend.evaluate(
                    model,
                    tokenizer,
                    held_rows,
                    progress(expid + " production"),
                    stop,
                    split="production held-out",
                    pack=pack,
                )
                if held_rows
                else None
            )
            backend.unload(model)
            (folder / f"{expid}-evaluation.json").write_text(json.dumps(evaluation, indent=2))
            evaluation_artifact = artifact(
                cid, folder / f"{expid}-evaluation.json", "evaluation", f"{expid} development evaluation"
            )
            transition(cid, "reviewing", "Independent integrity and regression checks before promotion")
            review = add_review(
                cid,
                {
                    "id": key,
                    "phase": "checkpoint",
                    **({"synthetic": synthetic or {}} if proposal["strategy"] == "distill_sft" else {}),
                    "reward_version": pack.reward_version,
                    "task_pack_sha256": pack.sha256,
                    "sealed_data_access": False,
                    "provenance_valid": True,
                    "weights_changed": result["weights_changed"],
                    "independent_reload": True,
                    "artifact_verified": training.sha(adapter / "adapter_model.safetensors")
                    == result["sha256"],
                    "sha256": result["sha256"],
                    "checkpoint_artifact_id": weight_artifact["id"],
                    "evaluation_artifact_id": evaluation_artifact["id"],
                    "evaluation_sha256": evaluation_artifact["sha256"],
                    "regression_drop": base_reg["score"] - reg["score"],
                    "dev_score": evaluation["score"],
                },
            )
            reading = team(
                "evaluator",
                agents.EVALUATOR,
                {
                    "baseline": {"correct": measured["correct"], "n": measured["n"]},
                    "checkpoint": {"correct": evaluation["correct"], "n": evaluation["n"]},
                    "compared_against": "baseline" if not best else best["id"],
                    **agents.changes(measured, evaluation),
                },
                agents.EVALUATOR_SCHEMA,
                expid,
            )
            opinion = team(
                "governance",
                agents.GOVERNANCE,
                {
                    "deterministic_decision": review["decision"],
                    "reason_codes": review.get("reason_codes", []),
                    "development": {"before": measured["score"], "after": evaluation["score"]},
                    "regression": {"before": base_reg["score"], "after": reg["score"]},
                    "production_held_out": {
                        "before": base_prod["correct"],
                        "after": prod_eval["correct"],
                        "n": prod_eval["n"],
                    }
                    if prod_eval
                    else None,
                    "training_review": training_review,
                    "evaluator_reading": reading,
                },
                agents.GOVERNANCE_SCHEMA,
                expid,
            )
            if reading or opinion:
                update_exp(evaluator_reading=reading, council_opinion=opinion)
            experiment.update("completed", json.dumps({"evaluation": evaluation, "review": review}))
            spent = min(reservation, time.monotonic() - exp_started)
            actor(cid).settle(key, spent)
            update_exp(
                status="completed" if review["decision"] == "approved" else "rejected",
                evaluation=evaluation,
                regression=reg,
                production=prod_eval,
                review=review,
            )
            send(
                cid,
                "evaluator",
                "director",
                "EVALUATION_COMPLETED",
                f"{expid}: {evaluation['correct']}/{evaluation['n']} development answers correct. Review {review['decision']}. Regression change {(reg['score'] - base_reg['score']) * 100:+.1f} pp."
                + (
                    f" Held-out production failures now correct: {prod_eval['correct']}/{prod_eval['n']} (baseline {base_prod['correct']}/{base_prod['n']})."
                    if prod_eval
                    else ""
                )
                + (f" {reading['summary']}" if reading else ""),
                expid,
            )
            previous.append(
                {
                    "strategy": proposal["strategy"],
                    "score": evaluation["score"],
                    "review": review["decision"],
                    "learning_rate": proposal["learning_rate"],
                    "lora_rank": proposal["lora_rank"],
                }
            )
            if (
                review["decision"] == "approved"
                and evaluation["score"] > baseline["score"]
                and (best is None or evaluation["score"] > best["evaluation"]["score"])
            ):
                governance(cid).promote(key, result["sha256"])
                best = {
                    "id": expid,
                    "key": key,
                    "evaluation": evaluation,
                    "sha256": result["sha256"],
                    "path": str(adapter),
                    "production": prod_eval,
                }
                save_field(best=best)
                send(
                    cid,
                    "governance",
                    "director",
                    "CHECKPOINT_APPROVED",
                    f"{expid} passed independent checks. Development change: {(evaluation['score'] - baseline['score']) * 100:+.1f} percentage points.",
                    expid,
                )
                verdict = (
                    f"Approved for promotion. Integrity, regression and provenance checks passed for {expid}."
                )
            else:
                send(
                    cid,
                    "governance",
                    "director",
                    "STRATEGY_REVISED",
                    f"{expid} will not be promoted. Choose the next experiment using these findings.",
                    expid,
                )
                reasons = "; ".join(code.replace("_", " ").capitalize() for code in review["reason_codes"])
                verdict = f"Not promoted. {reasons or 'No improvement over the current best'}."
            promoted = best is not None and best["id"] == expid
            send(
                cid,
                "governance",
                "trainer",
                "CHECKPOINT_VERDICT",
                verdict,
                expid,
                route="room",
                reply_to=checkpoint_post["message_id"],
            )
            if opinion:
                send(
                    cid,
                    "governance",
                    "director",
                    "COUNCIL_OPINION",
                    f"Model review ({opinion['recommendation']}, advisory): {opinion['rationale']} Risk to watch: {opinion['risk']}",
                    expid,
                    route="room",
                    reply_to=checkpoint_post["message_id"],
                )
            react(cid, "governance", checkpoint_post["message_id"], "white_check_mark" if promoted else "x")
        # One last boundary before the final: a hold still works here, and nothing sealed has been read.
        if chain.wait_while_held(cid, stop, holding) and stop():
            raise InterruptedError("Campaign budget reached or cancellation requested")
        # Selection freezes before protected data is read. No improvement is a valid outcome.
        selection = best["key"] if best else "baseline"
        actor(cid).freeze_selection(selection)
        if not actor(cid).claim_final():
            raise RuntimeError("Final evaluation already claimed; refuse repeated test access")
        transition(
            cid, "finalizing", "Selection frozen. Running the held-out test subset once on identical settings"
        )
        # Final evaluation is bounded by the campaign deadline. Selection is already frozen,
        # so no further research compute can be reserved or dispatched.
        final_rows = sealed()
        save_field(sealed_ids=[row["id"] for row in final_rows])
        if config.get("final_backend") == "lm_eval_azure":
            from services.core.azure_final import evaluate as independent_evaluate

            measured_final = independent_evaluate(
                cid,
                final_rows,
                config,
                best["path"] if best else None,
                best["sha256"] if best else None,
                stop,
            )
            final_base, final_best = measured_final["baseline"], measured_final["checkpoint"]
            artifact(
                cid,
                folder / "independent-final/baseline-harness.json",
                "evaluation",
                "Official harness baseline evidence",
            )
            if best:
                artifact(
                    cid,
                    folder / "independent-final/candidate-harness.json",
                    "evaluation",
                    "Official harness checkpoint evidence",
                )
        else:
            model, tokenizer = backend.load(model_id=config["model_id"])
            final_base = backend.evaluate(
                model,
                tokenizer,
                final_rows,
                progress("Final baseline"),
                stop,
                split="sealed final",
                pack=pack,
            )
            backend.unload(model)
            if best:
                model, tokenizer = backend.load(best["path"], config["model_id"])
                final_best = backend.evaluate(
                    model,
                    tokenizer,
                    final_rows,
                    progress("Final checkpoint"),
                    stop,
                    split="sealed final",
                    pack=pack,
                )
                backend.unload(model)
            else:
                final_best = final_base
        if not best:
            final_best = final_base
        final = {
            "baseline": final_base,
            "checkpoint": final_best,
            "checkpoint_id": selection,
            **final_verdict(final_base, final_best, config["seed"]),
            "selection_basis": "development only",
            "test_queries_per_model": 1,
        }
        if base_prod:
            after = best["production"] if best and best.get("production") else base_prod
            fixed = [
                a["id"]
                for b, a in zip(base_prod["predictions"], after["predictions"], strict=True)
                if a["correct"] and not b["correct"]
            ]
            final["production"] = {
                "baseline": {"correct": base_prod["correct"], "n": base_prod["n"]},
                "checkpoint": {"correct": after["correct"], "n": after["n"]},
                "fixed_ids": fixed,
                **final_verdict(base_prod, after, config["seed"]),
            }
        measured = json.loads(actor(cid).snapshot())["consumed_seconds"]
        save_field(
            final=final,
            elapsed_seconds=time.monotonic() - started,
            compute_seconds=measured + final_base["seconds"] + (final_best["seconds"] if best else 0),
            progress=None,
        )
        (folder / "final-evaluation.json").write_text(json.dumps(final, indent=2))
        artifact(cid, folder / "final-evaluation.json", "evaluation", "Sealed final evaluation")
        report = {
            "campaign_id": cid,
            "model_id": config["model_id"],
            "model_revision": training.model_spec(config["model_id"])["revision"],
            "hardware": backend.hardware,
            "config": config,
            "final": final,
            "limitations": [
                *pack.limitations,
                "Local sponsor runtimes; no claim of cloud durability"
                + ("; model work ran on a remote Modal GPU" if backend.name == "modal" else ""),
                "Only executed strategies are claimed. GRPO uses actor-returned deterministic rewards when selected.",
            ],
            "snapshot": store.get(cid),
        }
        (folder / "report.json").write_text(json.dumps(report, indent=2))
        artifact(cid, folder / "report.json", "report", "Research report")
        send(
            cid,
            "director",
            "governance",
            "CAMPAIGN_COMPLETED",
            f"Final evaluation complete. {VERDICT_SUMMARY[final['improvement_claim']]} All sample-level evidence preserved.",
            route="room",
        )
        transition(cid, "completed", "Research complete. Evidence package ready.")
        from services.core.cloud import mirror

        mirror(cid)
    except Exception as e:
        status = "cancelled" if isinstance(e, InterruptedError) else "failed"
        message = str(e)
        pending_ids = []

        def fail(s):
            s.update(status=status, error=message, elapsed_seconds=time.monotonic() - started, progress=None)
            for pending in s.get("experiments", []):
                if pending["status"] in {"proposed", "approved", "training", "evaluating"}:
                    pending_ids.append(pending["id"])
                    pending["status"] = status

        store.patch(cid, fail)
        for identity in pending_ids:
            try:
                get_actor("ExperimentActor", f"{cid}-{identity}").update(
                    status, json.dumps({"error": str(e)})
                )
            except Exception:
                pass
        store.event(cid, "error", str(e), "trainer")
        try:
            actor(cid).transition(status, str(e))
            state = json.loads(actor(cid).snapshot())
            for key, r in state["reservations"].items():
                if r["status"] == "reserved":
                    actual = time.time() - r.get("reserved_at", time.time() - (time.monotonic() - started))
                    actor(cid).settle(key, min(r["seconds"], max(0, actual)))
        except Exception:
            pass
        traceback.print_exc()
        raise
    finally:
        if backend is not None:
            backend.close()
        if bench is not None:
            used = bench.close()
            if used:
                save_field(workbenches=used)
        workbench.current.reset(bench_token)


if __name__ == "__main__":
    from services.core.actor_client import transport

    # Outlast a Terse runtime restart, including its owner-lease fence (about 30 s after SIGKILL).
    # Retries reuse one request ID per call, so a call that already committed is not run again.
    transport.retry_seconds = 180
    transport.role = "worker"
    run(sys.argv[1])
