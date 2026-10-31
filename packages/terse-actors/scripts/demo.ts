import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { clients, launch, localAuthority } from "./local-runtime.js";
import { digest, rewardCommitment, scopedId } from "../src/core.js";

// A coordination demo only. No optimizer, benchmark score, or model gain is simulated.
const campaign = "demo";
const authority = localAuthority([campaign]);
const directory = await mkdtemp(join(tmpdir(), "forge-terse-demo-"));
let runtime = await launch(authority, directory);
try {
  let c = clients(authority, runtime);
  const C = "ResearchCampaignActor",
    E = "ExperimentActor",
    W = "RLWorldActor";
  const model = { modelId: "fixture/no-training", revision: "a".repeat(40) };
  const spec = {
    objective: "Demonstrate durable coordination only",
    model,
    benchmark: "fixture",
    policyVersion: "v1",
    allowedStrategies: ["sft_lora"],
    maxBudgetMicroUsd: 1000000,
    maxConcurrent: 1,
    maxExperiments: 2,
    maxAttempts: 2,
    maxLeaseMs: 1000,
    maxRunMs: 10000,
    minImprovementBps: 100,
    baseline: {
      reportSha256: digest("fixture-baseline"),
      datasetSha256: digest("fixture-dev"),
      protocolSha256: digest("fixture-protocol"),
      numExamples: 1,
      numCorrect: 0,
    },
  };
  await c.owner.command(C, campaign, campaign, "initialize", spec);
  const experimentId = scopedId("exp", campaign, "one");
  await c.director.command(C, campaign, campaign, "propose", {
    experimentId,
    model,
    strategy: "sft_lora",
    configSha256: digest("fixture-config"),
    trainingDatasetSha256: digest("fixture-train"),
    developmentDatasetSha256: digest("fixture-dev"),
    hypothesis: "Exercise reservations without running training",
    maxCostMicroUsd: 100000,
  });
  const reservation = await c.coordinator.command(
    C,
    campaign,
    campaign,
    "reserve",
    { experimentId },
    "reserve-demo",
  );
  const worldId = scopedId("world", campaign, "math");
  const previous = process.env.FORGE_REWARD_SECRET;
  process.env.FORGE_REWARD_SECRET = authority.rewardSecret;
  let commitment: string;
  try {
    commitment = rewardCommitment(campaign, worldId, "sum", "v1", 1, "What is 6 + 7?", "13");
  } finally {
    if (previous === undefined) delete process.env.FORGE_REWARD_SECRET;
    else process.env.FORGE_REWARD_SECRET = previous;
  }
  await c.data.command(W, worldId, campaign, "reset", {
    episodeId: "demo-episode",
    task: {
      taskId: "sum",
      taskVersion: "v1",
      seed: 1,
      question: "What is 6 + 7?",
      answerCommitment: commitment,
      rewardVersion: "exact-numeric-v1",
      maxSteps: 2,
    },
  });
  const action = {
    episodeId: "demo-episode",
    actionId: "answer",
    expectedVersion: 0,
    action: { type: "submit_answer", answer: "13" },
  };
  await c.worker.command(W, worldId, campaign, "step", action, "answer-demo");
  const before = await c.owner.read(C, campaign, "getSnapshot");
  const closed = runtime.closed.catch(() => undefined);
  process.kill(runtime.connection.pid, "SIGKILL");
  await closed;
  runtime = await launch(authority, directory);
  c = clients(authority, runtime);
  const recovered = await c.owner.read(C, campaign, "getSnapshot");
  if (digest(before) !== digest(recovered)) throw new Error("Campaign changed after restart");
  const replay = await c.worker.command(W, worldId, campaign, "step", action, "answer-demo");
  await c.coordinator.command(E, experimentId, campaign, "abort", {
    ...reservation,
    reason: "Demo did not launch training",
  });
  await c.coordinator.command(C, campaign, campaign, "settle", {
    experimentId,
    actualMicroUsd: 0,
    costEvidenceSha256: digest("no-compute-launched"),
  });
  const final = await c.owner.read(C, campaign, "getSnapshot");
  const trajectory = await c.worker.command(W, worldId, campaign, "getTrajectory", {});
  if (replay.replayed !== true || trajectory.rewardTotal !== 1 || final!.reservedMicroUsd !== 0)
    throw new Error("Demo invariant failed");
  console.log(
    JSON.stringify(
      {
        scenario: "coordination-only fixture",
        runtime: "durable-actors@0.14.0",
        crashRecovery: true,
        duplicateActionReplayed: replay.replayed,
        trajectory,
        budget: {
          reservedMicroUsd: final!.reservedMicroUsd,
          spentMicroUsd: final!.spentMicroUsd,
          remainingMicroUsd: final!.remainingMicroUsd,
        },
        bestCheckpoint: final!.best,
        trainingExecuted: false,
        modelImprovementClaimed: false,
      },
      null,
      2,
    ),
  );
} finally {
  await runtime.stop();
  await rm(directory, { recursive: true, force: true });
}
