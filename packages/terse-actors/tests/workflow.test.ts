import { afterAll, beforeAll, expect, test } from "bun:test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { LocalActorRuntime } from "durable-actors/dev";
import { clients, launch, localAuthority } from "../scripts/local-runtime.js";
import { digest, scopedId } from "../src/core.js";
import type { ActorName, Document, Role } from "../src/contracts.js";

const campaign = "workflow";
const authority = localAuthority([campaign]);
let runtime: LocalActorRuntime;
let c: ReturnType<typeof clients>;
let dataDir: string;
const C: ActorName = "ResearchCampaignActor",
  E: ActorName = "ExperimentActor",
  F: ActorName = "ResearchWorkflowActor";
const hash = (name: string) => digest(name);
const exp = (label: string) => scopedId("exp", campaign, label);
const spec = {
  requirePreflight: true,
  objective: "Workflow outbox fixture; no model training or measured improvement",
  model: { modelId: "fixture/model", revision: "b".repeat(40) },
  benchmark: "fixture-math",
  policyVersion: "v1",
  allowedStrategies: ["sft_lora"],
  maxBudgetMicroUsd: 1000,
  maxConcurrent: 2,
  maxExperiments: 10,
  maxAttempts: 2,
  maxLeaseMs: 10000,
  maxRunMs: 120000,
  minImprovementBps: 100,
  baseline: {
    reportSha256: hash("baseline"),
    datasetSha256: hash("dev"),
    protocolSha256: hash("protocol"),
    numExamples: 10,
    numCorrect: 2,
  },
};
const proposal = (label: string) => ({
  experimentId: exp(label),
  strategy: "sft_lora",
  model: spec.model,
  configSha256: hash(label),
  trainingDatasetSha256: hash("train"),
  developmentDatasetSha256: hash("dev"),
  hypothesis: "Workflow fixture",
  maxCostMicroUsd: 100,
});
const command = (
  role: Role,
  actor: ActorName,
  actorId: string,
  method: string,
  payload: Document,
  requestId?: string,
) => c[role].command(actor, actorId, campaign, method, payload, requestId);
const workflow = (role: Role, method: string, payload: Document, requestId?: string) =>
  command(role, F, campaign, method, payload, requestId);
const snapshot = async () => (await c.owner.read(F, campaign, "getSnapshot"))!;
const deliveries = async () => (await snapshot()).deliveries as Record<string, Document>;
const grants: Record<string, Document> = {};

beforeAll(async () => {
  dataDir = await mkdtemp(join(tmpdir(), "forge-workflow-test-"));
  runtime = await launch(authority, dataDir);
  c = clients(authority, runtime);
  await command("owner", C, campaign, "initialize", spec);
}, 120000);
afterAll(async () => {
  if (runtime) await runtime.stop();
  if (dataDir) await rm(dataDir, { recursive: true, force: true });
});

test("messages follow fixed role routes and campaign-scoped experiment IDs", async () => {
  const body = { capability: "math" };
  await expect(
    workflow("data", "publish", {
      messageId: "m-wrong-sender",
      experimentId: exp("a"),
      kind: "RESEARCH_ASSIGNMENT",
      body,
    }),
  ).rejects.toThrow("MESSAGE_ROLE");
  await expect(
    workflow("owner", "publish", {
      messageId: "m-owner",
      experimentId: exp("a"),
      kind: "RESEARCH_ASSIGNMENT",
      body,
    }),
  ).rejects.toThrow("FORBIDDEN");
  await expect(
    workflow("director", "publish", {
      messageId: "m-unscoped",
      experimentId: "exp-unscoped",
      kind: "RESEARCH_ASSIGNMENT",
      body,
    }),
  ).rejects.toThrow("ACTOR_NAMESPACE");
  // Kinds produced by dedicated methods cannot be forged through publish.
  await expect(
    workflow("director", "publish", {
      messageId: "m-forged",
      experimentId: exp("a"),
      kind: "TRAINING_ASSIGNMENT",
      body,
    }),
  ).rejects.toThrow();
  const queued = await workflow(
    "director",
    "publish",
    { messageId: "m1", experimentId: exp("a"), kind: "RESEARCH_ASSIGNMENT", body },
    "queue-m1",
  );
  expect(queued.recipientRole).toBe("data");
  expect(queued.senderKeyId).toBe("director-local");
  expect(
    await workflow(
      "director",
      "publish",
      { messageId: "m1", experimentId: exp("a"), kind: "RESEARCH_ASSIGNMENT", body },
      "queue-m1",
    ),
  ).toEqual(queued);
  await expect(
    workflow("director", "publish", {
      messageId: "m1",
      experimentId: exp("a"),
      kind: "RESEARCH_ASSIGNMENT",
      body: { capability: "changed" },
    }),
  ).rejects.toThrow("MESSAGE_IMMUTABLE");
  expect((await snapshot()).version).toBe(1);
});

test("delivery receipts are coordinator-only, immutable and acknowledged by the recipient", async () => {
  await expect(workflow("data", "acknowledge", { messageId: "m1" })).rejects.toThrow(
    "NOT_DELIVERED",
  );
  await expect(
    workflow("director", "markSent", { messageId: "m1", relayMessageId: "relay-1" }),
  ).rejects.toThrow("FORBIDDEN");
  await expect(
    workflow("coordinator", "markSent", { messageId: "missing", relayMessageId: "relay-1" }),
  ).rejects.toThrow("UNKNOWN_MESSAGE");
  await workflow("coordinator", "markSent", { messageId: "m1", relayMessageId: "relay-1" });
  // A fresh request recording the same transport identity is harmless.
  await workflow("coordinator", "markSent", { messageId: "m1", relayMessageId: "relay-1" });
  await expect(
    workflow("coordinator", "markSent", { messageId: "m1", relayMessageId: "relay-2" }),
  ).rejects.toThrow("DELIVERY_CONFLICT");
  await expect(workflow("worker", "acknowledge", { messageId: "m1" })).rejects.toThrow(
    "MESSAGE_ROLE",
  );
  await workflow("data", "acknowledge", { messageId: "m1" });
  const entry = (await deliveries()).m1;
  expect(entry.relayMessageId).toBe("relay-1");
  expect(entry.acknowledged).toBe(true);
});

test("compute requires a matching, terminal preflight approval and one assignment", async () => {
  for (const label of ["rejected", "approved"]) {
    await command("director", C, campaign, "propose", proposal(label));
    grants[label] = (
      await command("coordinator", C, campaign, "reserve", { experimentId: exp(label) })
    ).grant as Document;
    expect((grants[label].claims as Document).requirePreflight).toBe(true);
    await command("coordinator", E, exp(label), "initialize", { grant: grants[label] });
    await expect(
      command("worker", E, exp(label), "acquireLease", { leaseMs: 1000 }),
    ).rejects.toThrow("PREFLIGHT_REQUIRED");
  }
  await expect(
    workflow("coordinator", "assign", {
      messageId: "assign-early",
      jobId: "job-approved",
      grant: grants.approved,
    }),
  ).rejects.toThrow("PREFLIGHT_REQUIRED");
  const review = (label: string, approved: boolean) => ({
    messageId: `preflight-${label}`,
    experimentId: exp(label),
    configSha256: hash(label),
    approved,
    reason: approved ? "Fixture limits checked" : "Fixture rejection",
  });
  await expect(workflow("director", "review", review("approved", true))).rejects.toThrow(
    "FORBIDDEN",
  );
  await workflow("reviewer", "review", review("rejected", false));
  await expect(workflow("reviewer", "review", review("rejected", true))).rejects.toThrow(
    "REVIEW_FINAL",
  );
  await expect(
    workflow("coordinator", "assign", {
      messageId: "assign-rejected",
      jobId: "job-rejected",
      grant: grants.rejected,
    }),
  ).rejects.toThrow("PREFLIGHT_REQUIRED");
  await expect(
    command("worker", E, exp("rejected"), "acquireLease", { leaseMs: 1000 }),
  ).rejects.toThrow("PREFLIGHT_REQUIRED");

  await workflow("reviewer", "review", review("approved", true));
  // Approval alone is not an assignment: the worker still cannot claim compute.
  await expect(
    command("worker", E, exp("approved"), "acquireLease", { leaseMs: 1000 }),
  ).rejects.toThrow("PREFLIGHT_REQUIRED");
  await expect(
    workflow("coordinator", "assign", {
      messageId: "assign-tampered",
      jobId: "job-approved",
      grant: { ...grants.approved, signature: "0".repeat(64) },
    }),
  ).rejects.toThrow("INVALID_GRANT");
  const assigned = await workflow("coordinator", "assign", {
    messageId: "assign-approved",
    jobId: "job-approved",
    grant: grants.approved,
  });
  expect(assigned.recipientRole).toBe("worker");
  expect((assigned.body as Document).configSha256).toBe(hash("approved"));
  await expect(
    workflow("coordinator", "assign", {
      messageId: "assign-again",
      jobId: "job-approved",
      grant: grants.approved,
    }),
  ).rejects.toThrow("ALREADY_ASSIGNED");
  const lease = await command("worker", E, exp("approved"), "acquireLease", { leaseMs: 10000 });
  expect(lease.generation).toBe(1);
  expect(await c.owner.read(F, campaign, "getAuthorization", [exp("rejected")])).toBeNull();
});

test("outbox order, receipts and reviews survive a native runtime SIGKILL", async () => {
  const before = await snapshot();
  const rows = Object.values(before.deliveries as Record<string, Document>);
  const order = rows
    .sort((a, b) => Number(a.sequence) - Number(b.sequence))
    .map((row) => (row.message as Document).messageId);
  expect(order).toEqual(["m1", "preflight-rejected", "preflight-approved", "assign-approved"]);
  const experiment = await c.owner.read(E, exp("approved"), "getSnapshot");
  const closed = runtime.closed.catch(() => undefined);
  process.kill(runtime.connection.pid, "SIGKILL");
  await closed;
  runtime = await launch(authority, dataDir);
  c = clients(authority, runtime);
  expect(await snapshot()).toEqual(before);
  // Compare stored state, not lease liveness: a slow restart can outlast the lease.
  expect(await c.owner.read(E, exp("approved"), "getSnapshot")).toEqual(experiment);
  expect(await c.owner.read(F, campaign, "getAuthorization", [exp("approved")])).toMatchObject({
    approved: true,
    configSha256: hash("approved"),
  });
  const events = (await c.owner.read(F, campaign, "getEvents", [0, 100]))!.events as Document[];
  expect(events.map((row) => row.sequence)).toEqual(events.map((_, i) => i + 1));
  expect(events.map((row) => row.type)).toEqual([
    "MESSAGE_QUEUED",
    "MESSAGE_SENT",
    "MESSAGE_SENT",
    "MESSAGE_ACKNOWLEDGED",
    "PREFLIGHT_REVIEWED",
    "PREFLIGHT_REVIEWED",
    "TRAINING_ASSIGNED",
  ]);
});
