import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import {
  clients,
  launch,
  localAuthority,
} from "../../../packages/terse-actors/scripts/local-runtime.js";
import { canonical, digest, document, scopedId } from "../../../packages/terse-actors/src/core.js";
import type { ActorName, Document, Role } from "../../../packages/terse-actors/src/contracts.js";
import type { WorkflowMessage } from "../../../packages/terse-actors/src/workflow.js";
import { reconcile, settleAtReservedBound } from "../src/reconcile.js";
import {
  RelayDispatcher,
  participantRoles,
  registerParticipants,
  type Participants,
  type ReceiveReport,
} from "../src/relay.js";
import { TrainingBridge, prepareJob, pythonJson, readJson } from "../src/worker.js";
import { openRelay, type RelayTarget } from "./relay-target.js";

/** Resolves true only for the expected failure code; unexpected errors propagate. */
async function rejects(action: () => Promise<unknown>, code: string): Promise<boolean> {
  try {
    await action();
  } catch (error) {
    if (String(error).includes(code)) return true;
    throw error;
  }
  return false;
}
function invariant(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(`Scenario invariant failed: ${message}`);
}

/**
 * Real Relaycast and Terse services and real tiny-model optimization, with labeled fault
 * injection. Role handlers are scripted: this verifies dispatch, not autonomous research.
 */
export async function runScenario(output: string, python: string): Promise<Document> {
  await mkdir(output, { recursive: true });
  const campaign = "relay-mechanics",
    experimentId = scopedId("exp", campaign, "tiny-lora");
  const fixture = join(output, "fixture");
  await pythonJson(python, [
    "-c",
    "import sys,json; from pathlib import Path; from forge_training.fixtures import make_smoke_fixture; print(json.dumps({'config':str(make_smoke_fixture(Path(sys.argv[1])))}))",
    fixture,
  ]);
  const configPath = join(fixture, "experiment.json"),
    config = await readJson(configPath);
  await writeFile(
    configPath,
    JSON.stringify({ ...config, campaign_id: campaign, experiment_id: experimentId }),
  );
  const job = await prepareJob(
    python,
    "tiny",
    configPath,
    join(output, "frozen-baseline.json"),
    join(output, "evaluator"),
  );
  const baseline = await readJson(job.baselinePath),
    model = job.config.model as Document;
  const proposal = {
    experimentId,
    strategy: "sft_lora",
    model: { modelId: model.model_id, revision: model.revision },
    configSha256: job.configSha256,
    trainingDatasetSha256: job.trainingDatasetSha256,
    developmentDatasetSha256: (baseline.dataset as Document).content_sha256,
    hypothesis: "Offline random tiny-model optimization verifies dispatch mechanics only",
    maxCostMicroUsd: 100000,
  };
  const spec = {
    requirePreflight: true,
    objective: "Verify synchronized dispatch and real optimization; no model-superiority claim",
    model: proposal.model,
    benchmark: "offline-fixture",
    policyVersion: "v1",
    allowedStrategies: ["sft_lora"],
    maxBudgetMicroUsd: 100000,
    maxConcurrent: 1,
    maxExperiments: 2,
    maxAttempts: 2,
    maxLeaseMs: 10000,
    maxRunMs: 420000,
    minImprovementBps: 100,
    baseline: {
      reportSha256: job.baselineSha256,
      datasetSha256: proposal.developmentDatasetSha256,
      protocolSha256: digest(baseline.protocol),
      numExamples: baseline.num_examples,
      numCorrect: baseline.num_correct,
    },
  };
  const authority = localAuthority([campaign]);
  let runtime = await launch(authority, join(output, "terse"));
  let relayTarget: RelayTarget | undefined;
  try {
    relayTarget = await openRelay(output);
    const relay = await relayTarget.createWorkspace("forge-relay-mechanics");
    let participants = await registerParticipants(relay);
    let actors = clients(authority, runtime);
    let dispatcher = new RelayDispatcher(campaign, actors, participants);
    const command = (
      role: Role,
      actor: ActorName,
      id: string,
      method: string,
      payload: Document,
      requestId?: string,
    ) => actors[role].command(actor, id, campaign, method, payload, requestId);
    const transport: ReceiveReport = {
      viaRelay: 0,
      viaReconciliation: 0,
      duplicateCopies: 0,
      rejectedCopies: 0,
      alreadyHandled: 0,
    };
    const receive = async (
      role: (typeof participantRoles)[number],
      handler: (message: WorkflowMessage) => Promise<void>,
      via = dispatcher,
    ) => {
      const report = await via.receive(role, handler);
      for (const key of Object.keys(transport) as (keyof ReceiveReport)[])
        transport[key] += report[key];
      return report;
    };

    await command("owner", "ResearchCampaignActor", campaign, "initialize", spec);
    await command("owner", "GovernanceActor", campaign, "initialize", {
      objectiveSha256: digest(spec),
      policyVersion: "v1",
    });
    await dispatcher.publish("director", "research", experimentId, "RESEARCH_ASSIGNMENT", {
      jobId: job.jobId,
      configSha256: job.configSha256,
      capability: "math",
    });
    // FAULT: a send succeeds but its response is lost before the Terse transport receipt.
    const lostSendResponse = await rejects(
      () =>
        dispatcher.flush(async () => {
          throw new Error("INJECTED_LOST_SEND_RESPONSE");
        }),
      "INJECTED_LOST_SEND_RESPONSE",
    );
    invariant(lostSendResponse, "lost-response injection executed");
    await dispatcher.flush();
    const researcherInbox = await participants.data.inbox.list({ limit: 100 });
    invariant(researcherInbox.items.length === 1, "transport retry did not duplicate the DM");

    await receive("data", async () => {
      await dispatcher.publish(
        "data",
        "proposal",
        experimentId,
        "EXPERIMENT_PROPOSAL",
        document(proposal),
      );
    });
    await dispatcher.flush();
    await receive("director", async (message) => {
      await command(
        "director",
        "ResearchCampaignActor",
        campaign,
        "propose",
        message.body,
        "proposal-recorded",
      );
      await dispatcher.publish("director", "preflight-request", experimentId, "REVIEW_REQUEST", {
        jobId: job.jobId,
        configSha256: job.configSha256,
      });
    });
    const reservation = await command(
      "coordinator",
      "ResearchCampaignActor",
      campaign,
      "reserve",
      { experimentId },
      "reserve-tiny",
    );
    await command(
      "coordinator",
      "ExperimentActor",
      experimentId,
      "initialize",
      reservation,
      "initialize-tiny",
    );
    const preflightBlocked = await rejects(
      () => command("worker", "ExperimentActor", experimentId, "acquireLease", { leaseMs: 10000 }),
      "PREFLIGHT_REQUIRED",
    );
    invariant(preflightBlocked, "worker cannot acquire compute before preflight");
    await dispatcher.flush();
    await receive("reviewer", async () => {
      const validation = await pythonJson(python, [
        "-m",
        "forge_training",
        "validate",
        "--config",
        job.configPath,
      ]);
      if (digest(validation.config) !== job.configSha256)
        throw new Error("Preflight config changed");
      await command(
        "reviewer",
        "ResearchWorkflowActor",
        campaign,
        "review",
        {
          messageId: "preflight",
          experimentId,
          configSha256: job.configSha256,
          approved: true,
          reason:
            "Pinned local fixture; strict worker validation passed; CPU-only mechanics run within limits",
        },
        "review-tiny",
      );
    });
    await dispatcher.flush();
    await receive("coordinator", async () => {
      await command(
        "coordinator",
        "ResearchWorkflowActor",
        campaign,
        "assign",
        { messageId: "train", jobId: job.jobId, ...reservation },
        "assign-tiny",
      );
    });
    await dispatcher.flush();

    // FAULTS persisted in Relay before restart: an at-least-once duplicate of the genuine
    // assignment, and a forged copy from the wrong sender naming a job outside the allowlist.
    const assignment = (await dispatcher.deliveries()).find(
      (d) => d.message.kind === "TRAINING_ASSIGNMENT",
    )!.message;
    await participants.coordinator.sendMessage({
      to: "@forge-worker",
      text: canonical(assignment),
    });
    await participants.director.sendMessage({
      to: "@forge-worker",
      text: canonical({ ...assignment, body: { ...assignment.body, jobId: "forged-job" } }),
    });

    // FAULT: SIGKILL both persistent services after delivery, before the worker subscribes.
    // A remote engine is not ours to kill; every Relay client is dropped and reconnected.
    const relayRestart = await relayTarget.restart();
    const recoveredRelay = relayTarget.connect(relay.workspaceKey!);
    const reconnected = {} as Participants;
    for (const role of participantRoles)
      reconnected[role] = await recoveredRelay.workspace.reconnect({
        apiToken: participants[role].token!,
      });
    participants = reconnected;
    const closed = runtime.closed.catch(() => undefined);
    process.kill(runtime.connection.pid, "SIGKILL");
    await closed;
    runtime = await launch(authority, join(output, "terse"));
    actors = clients(authority, runtime);
    dispatcher = new RelayDispatcher(campaign, actors, participants);
    const afterRestart = await reconcile(dispatcher);
    invariant(
      afterRestart.unacknowledged.includes("train") && !afterRestart.unsent.length,
      "pending assignment survived both restarts",
    );

    const bridge = new TrainingBridge(python, dispatcher);
    bridge.register(job);
    // FAULT: the worker crashes after the authoritative completion, before announcing it.
    const crashedAfterCompletion = await rejects(
      () =>
        dispatcher.receive("worker", (message) =>
          bridge.train(message, async () => {
            throw new Error("INJECTED_CRASH_AFTER_COMPLETION");
          }),
        ),
      "INJECTED_CRASH_AFTER_COMPLETION",
    );
    invariant(crashedAfterCompletion, "post-completion crash injection executed");
    const earlyCompletion = await reconcile(dispatcher);
    invariant(
      earlyCompletion.unannouncedCompletions.includes(experimentId) &&
        earlyCompletion.unacknowledged.includes("train"),
      "early completion is discoverable from actor state",
    );
    let trainedMessage: WorkflowMessage | undefined;
    const workerReceipt = await receive("worker", async (message) => {
      trainedMessage = message;
      await bridge.train(message);
    });
    invariant(trainedMessage, "training assignment recovered after restart and crash");
    // A duplicate work notification must reconcile the completed bundle, not optimize again.
    await bridge.train(trainedMessage);

    await dispatcher.flush();
    await receive("evaluator", (message) => bridge.evaluate(message));
    await dispatcher.flush();
    const rejection = ["MECHANICS_ONLY", "REGRESSIONS_NOT_RUN"];
    await receive("reviewer", async (message) => {
      // A mechanics fixture has no independent regression suite: preserve an actual veto.
      await command(
        "reviewer",
        "GovernanceActor",
        campaign,
        "recordDecision",
        {
          experimentId,
          checkpointSha256: message.body.checkpointSha256,
          evaluationReportSha256: message.body.reportSha256,
          objectiveSha256: digest(spec),
          policyVersion: "v1",
          decision: "rejected",
          reasonCodes: rejection,
          evidenceRefs: [message.body.reportSha256],
          checks: {
            checksumVerified: true,
            objectiveAligned: true,
            evaluationReproducible: true,
            regressionsPassed: false,
            dataAllowed: true,
          },
        },
        "final-review-tiny",
      );
      await dispatcher.publish(
        "reviewer",
        "checkpoint-review",
        experimentId,
        "CHECKPOINT_REVIEWED",
        {
          decision: "rejected",
          reasonCodes: rejection,
          reportSha256: message.body.reportSha256,
        },
      );
    });
    await dispatcher.flush();
    await receive("director", async () => {
      await dispatcher.publish("director", "correction", experimentId, "CORRECTION", {
        action: "require-real-development-and-regression-evidence",
        newTrainingAuthorized: false,
      });
    });
    // FAULT: earlier chatter fills a one-item inbox page (the Relay inbox is FIFO), so the
    // recipient must reconcile the correction from the outbox's recorded Relay ID.
    for (const text of ["status: correction noted", "status: awaiting new evidence"])
      await participants.director.sendMessage({ to: "@forge-data", text });
    await dispatcher.flush();
    const correctionReceipt = await receive(
      "data",
      async () => undefined,
      new RelayDispatcher(campaign, actors, participants, 1),
    );
    invariant(
      correctionReceipt.viaReconciliation === 1 && correctionReceipt.viaRelay === 0,
      "a message missing from the inbox page is reconciled from the outbox",
    );
    // The reconciled correction's Relay delivery is still queued: it is closed without
    // re-running the handler, the same path as a Relay ack lost after the Forge ack.
    const staleReceipt = await receive("data", async () => {
      throw new Error("handler must not re-run for acknowledged work");
    });
    invariant(staleReceipt.alreadyHandled === 1, "a stale Relay delivery is acked, not re-run");
    const inboxes: Record<string, number> = {};
    for (const role of participantRoles)
      inboxes[role] = (await participants[role].inbox.list({ limit: 200 })).items.length;
    invariant(
      Object.values(inboxes).every((n) => n === 0),
      "every Relay inbox is drained: each delivery was acked or failed",
    );
    await settleAtReservedBound(dispatcher, experimentId, join(output, "cost", "tiny-lora.json"));
    const vetoHonored = await rejects(
      () => command("director", "ResearchCampaignActor", campaign, "promote", { experimentId }),
      "GOVERNANCE_VETO",
    );
    invariant(vetoHonored, "promotion honors the governance veto");

    const final = await reconcile(dispatcher);
    const experiment = (await actors.owner.read("ExperimentActor", experimentId, "getSnapshot"))!;
    const campaignState = (await actors.owner.read(
      "ResearchCampaignActor",
      campaign,
      "getSnapshot",
    ))!;
    const deliveries = await dispatcher.deliveries();
    const runDir = String(job.config.output_dir);
    const result = await readJson(join(runDir, "result.json"));
    const runEvents = (await readFile(join(runDir, "events.jsonl"), "utf8"))
      .trim()
      .split("\n")
      .map((line) => (JSON.parse(line) as Document).type);
    const evaluation = experiment.evaluation as Document;
    invariant(experiment.status === "evaluated", "experiment evaluated");
    invariant(experiment.attempts === 1, "exactly one worker attempt");
    invariant(
      runEvents.filter((type) => type === "TRAINING_STARTED").length === 1,
      "exactly one optimization run",
    );
    invariant(
      deliveries.filter((d) => d.message.kind === "TRAINING_COMPLETED").length === 1,
      "one completion announcement",
    );
    invariant(
      workerReceipt.viaRelay === 1 &&
        workerReceipt.duplicateCopies === 1 &&
        workerReceipt.rejectedCopies === 1,
      "worker handled one genuine assignment and ignored duplicate and forged copies",
    );
    invariant(
      deliveries.every((d) => d.acknowledged) && Object.values(final).every((rows) => !rows.length),
      "final reconciliation is clean",
    );
    invariant(
      campaignState.best === null && campaignState.reservedMicroUsd === 0,
      "no promotion; reservation settled",
    );
    const report = document({
      phase: 3,
      mode: "scripted-role-handlers-with-real-Relay-and-Terse",
      relaySdk: "14.0.0",
      relayEngine:
        relayTarget.kind === "local" ? "@relaycast/engine 12.0.0 (loopback)" : relayTarget.url,
      relayWorkspaces: relayTarget.workspaces,
      terse: "0.14.0",
      faultsInjected: [
        "lost-send-response",
        "duplicate-relay-copy",
        "forged-relay-copy",
        relayRestart,
        "terse-sigkill",
        "worker-crash-after-completion",
        "duplicate-work-notification",
        "message-beyond-inbox-page",
        "relay-ack-after-reconciliation",
      ],
      preflightBlocked,
      transport,
      workerReceipt,
      correctionReceipt,
      staleReceipt,
      relayInboxesAfterRun: inboxes,
      reconciliation: { afterRestart, earlyCompletion, final },
      messages: deliveries.map((d) => ({
        sequence: d.sequence,
        kind: d.message.kind,
        messageId: d.message.messageId,
        relayMessageId: d.relayMessageId,
        acknowledged: d.acknowledged,
      })),
      experimentStatus: experiment.status,
      workerAttempts: experiment.attempts,
      optimizationRuns: runEvents.filter((type) => type === "TRAINING_STARTED").length,
      weightsChanged: result.weights_changed,
      evaluatorReproducedWorkerPredictions: true,
      development: {
        numExamples: evaluation.numExamples,
        baselineCorrect: evaluation.baselineCorrect,
        candidateCorrect: evaluation.candidateCorrect,
      },
      comparison: result.comparison,
      governanceDecision: "rejected",
      promotionBlocked: "GOVERNANCE_VETO",
      bestCheckpoint: campaignState.best,
      spentMicroUsd: campaignState.spentMicroUsd,
      reservedMicroUsd: campaignState.reservedMicroUsd,
      costBasis: "reserved-upper-bound; unmeasured local process",
      modelImprovementClaimed: false,
      autonomousResearchClaimed: false,
    });
    await writeFile(join(output, "verification.json"), JSON.stringify(report, null, 2));
    return report;
  } finally {
    await relayTarget?.stop();
    await runtime.stop();
  }
}
