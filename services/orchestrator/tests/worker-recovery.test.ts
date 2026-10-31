import { afterAll, beforeAll, expect, test } from "bun:test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  clients,
  launch,
  localAuthority,
} from "../../../packages/terse-actors/scripts/local-runtime.js";
import type { Document } from "../../../packages/terse-actors/src/contracts.js";
import { digest, scopedId } from "../../../packages/terse-actors/src/core.js";
import type { WorkflowMessage } from "../../../packages/terse-actors/src/workflow.js";
import { RelayDispatcher, type Participants } from "../src/relay.js";
import { settleAtReservedBound } from "../src/reconcile.js";
import { trainWithRetries } from "../src/worker.js";

const authority = localAuthority(["attempts", "deadline", "expired", "no-progress", "completion"]);
let directory: string;
let runtime: Awaited<ReturnType<typeof launch>>;
let actors: ReturnType<typeof clients>;
beforeAll(async () => {
  directory = await mkdtemp(join(tmpdir(), "forge-worker-recovery-"));
  runtime = await launch(authority, join(directory, "terse"));
  actors = clients(authority, runtime);
}, 120000);
afterAll(async () => {
  await runtime?.stop();
  if (directory) await rm(directory, { recursive: true, force: true });
});

async function assignment(campaignId: string, maxAttempts = 2, maxRunMs = 60000) {
  const model = { modelId: "fixture/model", revision: "a".repeat(40) };
  const experimentId = scopedId("exp", campaignId, "job");
  await actors.owner.command("ResearchCampaignActor", campaignId, campaignId, "initialize", {
    objective: "Worker recovery fixture; no optimization or performance claim",
    model,
    benchmark: "fixture",
    policyVersion: "v1",
    allowedStrategies: ["sft_lora"],
    maxBudgetMicroUsd: 100,
    maxConcurrent: 1,
    maxExperiments: 1,
    maxAttempts,
    maxRunMs,
    maxLeaseMs: 10000,
    minImprovementBps: 100,
    baseline: {
      reportSha256: digest("baseline"),
      datasetSha256: digest("dev"),
      protocolSha256: digest("protocol"),
      numExamples: 1,
      numCorrect: 0,
    },
  });
  await actors.director.command("ResearchCampaignActor", campaignId, campaignId, "propose", {
    experimentId,
    strategy: "sft_lora",
    model,
    configSha256: digest("config"),
    trainingDatasetSha256: digest("train"),
    developmentDatasetSha256: digest("dev"),
    hypothesis: "Fault injection exercises existing recovery",
    maxCostMicroUsd: 100,
  });
  const grant = await actors.coordinator.command(
    "ResearchCampaignActor",
    campaignId,
    campaignId,
    "reserve",
    { experimentId },
  );
  await actors.coordinator.command(
    "ExperimentActor",
    experimentId,
    campaignId,
    "initialize",
    grant,
  );
  const dispatcher = new RelayDispatcher(campaignId, actors, {} as Participants);
  const message: WorkflowMessage = {
    schemaVersion: 1,
    campaignId,
    experimentId,
    messageId: "train-job",
    kind: "TRAINING_ASSIGNMENT",
    senderRole: "coordinator",
    senderKeyId: "coordinator-local",
    recipientRole: "worker",
    body: {
      jobId: "job",
      configSha256: digest("config"),
      grantId: ((grant.grant as Document).claims as Document).grantId,
    },
  };
  const command = (method: string, payload: Document) =>
    actors.worker.command("ExperimentActor", experimentId, campaignId, method, payload);
  const snapshot = async () =>
    (await actors.worker.read("ExperimentActor", experimentId, "getSnapshot"))!;
  return { dispatcher, message, command, snapshot };
}

async function expectSettledFailure(dispatcher: RelayDispatcher, message: WorkflowMessage) {
  const deliveries = await dispatcher.deliveries();
  expect(deliveries).toHaveLength(1);
  expect(deliveries[0].message.kind).toBe("TRAINING_FAILED");
  await settleAtReservedBound(
    dispatcher,
    message.experimentId,
    join(directory, `${message.campaignId}-cost.json`),
  );
  const campaign = await actors.coordinator.read(
    "ResearchCampaignActor",
    message.campaignId,
    "getSnapshot",
  );
  expect(campaign).toMatchObject({ reservedMicroUsd: 0, spentMicroUsd: 100 });
}

test("real actor attempt limits produce one terminal failure and settleable cost", async () => {
  const ctx = await assignment("attempts");
  let calls = 0;
  const bridge = {
    async train() {
      calls++;
      const lease = await ctx.command("acquireLease", { leaseMs: 10000 });
      await ctx.command("reportFailure", {
        generation: lease.generation,
        reason: "Injected worker failure",
      });
      throw new Error("Injected worker failure");
    },
  };
  await trainWithRetries(bridge, ctx.dispatcher, ctx.message);
  expect(calls).toBe(2);
  expect(await ctx.snapshot()).toMatchObject({ status: "failed", attempts: 2 });
  await expectSettledFailure(ctx.dispatcher, ctx.message);
});

test("runtime expiry while retryable closes the experiment instead of retrying forever", async () => {
  const ctx = await assignment("deadline", 2, 1000);
  let calls = 0;
  const bridge = {
    async train() {
      calls++;
      const lease = await ctx.command("acquireLease", { leaseMs: 10000 });
      await ctx.command("reportFailure", {
        generation: lease.generation,
        reason: "Injected early failure",
      });
      await Bun.sleep(1100);
      throw new Error("Injected early failure");
    },
  };
  await trainWithRetries(bridge, ctx.dispatcher, ctx.message);
  expect(calls).toBe(2);
  expect(await ctx.snapshot()).toMatchObject({ status: "failed", attempts: 1 });
  await expectSettledFailure(ctx.dispatcher, ctx.message);
});

test("a crashed final attempt with an expired lease is aborted and settled", async () => {
  const ctx = await assignment("expired", 1);
  let calls = 0;
  const bridge = {
    async train() {
      calls++;
      await ctx.command("acquireLease", { leaseMs: 50 });
      await Bun.sleep(80);
      throw new Error("Injected lease loss");
    },
  };
  await trainWithRetries(bridge, ctx.dispatcher, ctx.message);
  expect(calls).toBe(2);
  expect(await ctx.snapshot()).toMatchObject({ status: "failed", attempts: 1 });
  await expectSettledFailure(ctx.dispatcher, ctx.message);
});

test("errors before a new attempt stay pending without consuming retries or spinning", async () => {
  const ctx = await assignment("no-progress");
  const lease = await ctx.command("acquireLease", { leaseMs: 10000 });
  await ctx.command("reportFailure", {
    generation: lease.generation,
    reason: "First attempt failed",
  });
  let calls = 0;
  const bridge = {
    async train() {
      calls++;
      throw new Error("CONFIG_MISMATCH");
    },
  };
  await expect(trainWithRetries(bridge, ctx.dispatcher, ctx.message)).rejects.toThrow(
    "CONFIG_MISMATCH",
  );
  expect(calls).toBe(1);
  expect(await ctx.snapshot()).toMatchObject({ status: "retryable", attempts: 1 });
  expect(await ctx.dispatcher.deliveries()).toEqual([]);
});

test("a lost completion reply retries announcement without starting another attempt", async () => {
  const ctx = await assignment("completion");
  let calls = 0;
  const bridge = {
    async train() {
      calls++;
      if ((await ctx.snapshot()).status === "completed") {
        await ctx.dispatcher.publish(
          "worker",
          "trained-job",
          ctx.message.experimentId,
          "TRAINING_COMPLETED",
          { jobId: "job" },
        );
        return;
      }
      const lease = await ctx.command("acquireLease", { leaseMs: 10000 });
      await ctx.command("complete", {
        generation: lease.generation,
        result: {
          checkpointArtifactId: "checkpoint-job",
          checkpointSha256: digest("checkpoint"),
          manifestSha256: digest("manifest"),
          configSha256: digest("config"),
          weightsChanged: true,
        },
      });
      throw new Error("INJECTED_LOST_COMPLETION_REPLY");
    },
  };
  await trainWithRetries(bridge, ctx.dispatcher, ctx.message);
  expect(calls).toBe(2);
  expect(await ctx.snapshot()).toMatchObject({ status: "completed", attempts: 1 });
  expect((await ctx.dispatcher.deliveries()).map((d) => d.message.kind)).toEqual([
    "TRAINING_COMPLETED",
  ]);
});
