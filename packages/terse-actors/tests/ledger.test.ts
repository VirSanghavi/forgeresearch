import { afterAll, beforeAll, expect, test } from "bun:test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { LocalActorRuntime } from "durable-actors/dev";
import { clients, launch, localAuthority } from "../scripts/local-runtime.js";
import { digest, scopedId } from "../src/core.js";
import type { ActorName, Document, Role } from "../src/contracts.js";

const campaign = "ledger";
const authority = localAuthority([campaign]);
let runtime: LocalActorRuntime;
let c: ReturnType<typeof clients>;
let dataDir: string;
const C: ActorName = "ResearchCampaignActor",
  E: ActorName = "ExperimentActor",
  L: ActorName = "ResearchLedgerActor";
const hash = (name: string) => digest(name);
const exp = (label: string) => scopedId("exp", campaign, label);
const spec = {
  objective: "Ledger fixture; no model training or measured improvement",
  model: { modelId: "fixture/model", revision: "c".repeat(40) },
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
const command = (
  role: Role,
  actor: ActorName,
  actorId: string,
  method: string,
  payload: Document,
  requestId?: string,
) => c[role].command(actor, actorId, campaign, method, payload, requestId);
const record = (payload: Document, requestId?: string, role: Role = "director") =>
  command(role, L, campaign, "record", payload, requestId);
const entry = (entryId: string, kind: string, extra: Document = {}): Document => ({
  entryId,
  kind,
  summary: `${kind} fixture`,
  parents: [],
  evidence: [],
  body: {},
  ...extra,
});
const baseline = { source: "baseline", reportSha256: hash("baseline") };
const evaluation = (label: string) => ({
  source: "evaluation",
  experimentId: exp(label),
  reportSha256: hash(`eval-${label}`),
});
const decision = (entryId: string, launch: string[], extra: Document = {}) =>
  entry(entryId, "decision", {
    evidence: [baseline],
    parents: ["diagnosis"],
    body: {
      rejected: [{ option: "do-nothing", reason: "Baseline fails the objective" }],
      launch: launch.map((label) => ({
        experimentId: exp(label),
        label,
        recipe: { data: "full", learningRate: 0.0001 },
      })),
    },
    ...extra,
  });

beforeAll(async () => {
  dataDir = await mkdtemp(join(tmpdir(), "forge-ledger-test-"));
  runtime = await launch(authority, dataDir);
  c = clients(authority, runtime);
  await command("owner", C, campaign, "initialize", spec);
}, 120000);
afterAll(async () => {
  if (runtime) await runtime.stop();
  if (dataDir) await rm(dataDir, { recursive: true, force: true });
});

test("only the director appends; entries need verified evidence or parents", async () => {
  await expect(
    record(entry("diagnosis", "diagnosis", { evidence: [baseline] }), undefined, "reviewer"),
  ).rejects.toThrow("FORBIDDEN");
  await expect(record(entry("diagnosis", "diagnosis"))).rejects.toThrow("EVIDENCE_REQUIRED");
  await expect(
    record(
      entry("diagnosis", "diagnosis", {
        evidence: [{ source: "baseline", reportSha256: hash("other") }],
      }),
    ),
  ).rejects.toThrow("EVIDENCE_MISMATCH");
  const first = await record(
    entry("diagnosis", "diagnosis", { evidence: [baseline] }),
    "diagnosis-request",
  );
  expect(
    await record(entry("diagnosis", "diagnosis", { evidence: [baseline] }), "diagnosis-request"),
  ).toEqual(first);
  await expect(record(entry("diagnosis", "diagnosis", { evidence: [baseline] }))).rejects.toThrow(
    "ENTRY_IMMUTABLE",
  );
  await expect(record(entry("h1", "hypothesis"))).rejects.toThrow("EVIDENCE_REQUIRED");
  await expect(record(entry("h1", "hypothesis", { parents: ["missing"] }))).rejects.toThrow(
    "UNKNOWN_PARENT",
  );
  await record(entry("h1", "hypothesis", { parents: ["diagnosis"] }));
  expect((await c.owner.read(L, campaign, "getSnapshot"))!.version).toBe(2);
});

test("decisions name rejected alternatives and launch each experiment once", async () => {
  await expect(
    record(entry("d0", "decision", { evidence: [baseline], parents: ["diagnosis"] })),
  ).rejects.toThrow("DECISION_SHAPE");
  await expect(
    record({
      ...decision("d0", []),
      body: { rejected: [], launch: [{ experimentId: "exp-unscoped", label: "x", recipe: {} }] },
    }),
  ).rejects.toThrow("ACTOR_NAMESPACE");
  await record(decision("d1", ["a", "b"]));
  const launch = (await c.owner.read(L, campaign, "getLaunch", [exp("a")]))!;
  expect(launch.entryId).toBe("d1");
  expect(launch.authorKeyId).toBe("director-local");
  expect((launch.launch as Document).label).toBe("a");
  expect(await c.owner.read(L, campaign, "getLaunch", [exp("never")])).toBeNull();
  await expect(record(decision("d2", ["a"]))).rejects.toThrow("LAUNCH_EXISTS");
});

test("experiment evidence must match recorded evaluations or terminal failures", async () => {
  for (const label of ["a", "b"]) {
    await command("director", C, campaign, "propose", {
      experimentId: exp(label),
      strategy: "sft_lora",
      model: spec.model,
      configSha256: hash(label),
      trainingDatasetSha256: hash("train"),
      developmentDatasetSha256: hash("dev"),
      hypothesis: "Ledger fixture",
      maxCostMicroUsd: 100,
    });
    const reserved = await command("coordinator", C, campaign, "reserve", {
      experimentId: exp(label),
    });
    await command("coordinator", E, exp(label), "initialize", reserved);
    if (label === "b") {
      await command("coordinator", E, exp(label), "abort", { ...reserved, reason: "Fixture" });
      continue;
    }
    const lease = await command("worker", E, exp(label), "acquireLease", { leaseMs: 10000 });
    await expect(record(decision("d-early", [], { evidence: [evaluation("a")] }))).rejects.toThrow(
      "EVIDENCE_MISMATCH",
    );
    await command("worker", E, exp(label), "complete", {
      generation: lease.generation,
      result: {
        checkpointArtifactId: `checkpoint-${label}`,
        checkpointSha256: hash(`weights-${label}`),
        manifestSha256: hash(`manifest-${label}`),
        configSha256: hash(label),
        weightsChanged: true,
      },
    });
    await command("evaluator", E, exp(label), "recordEvaluation", {
      reportArtifactId: `eval-${label}`,
      reportSha256: hash(`eval-${label}`),
      baselineReportSha256: hash("baseline"),
      checkpointSha256: hash(`weights-${label}`),
      datasetSha256: hash("dev"),
      protocolSha256: hash("protocol"),
      evaluationRole: "development",
      numExamples: 10,
      baselineCorrect: 2,
      candidateCorrect: 3,
    });
  }
  await expect(
    record(
      decision("d-wrong", [], {
        evidence: [{ ...evaluation("a"), reportSha256: hash("forged-report") }],
      }),
    ),
  ).rejects.toThrow("EVIDENCE_MISMATCH");
  await expect(
    record(decision("d-wrong", [], { evidence: [{ source: "failure", experimentId: exp("a") }] })),
  ).rejects.toThrow("EVIDENCE_MISMATCH");
  await record(
    decision("d3", ["c"], {
      parents: ["d1"],
      evidence: [baseline, evaluation("a"), { source: "failure", experimentId: exp("b") }],
    }),
  );
});

test("a stop rule closes the ledger, durably across a native runtime SIGKILL", async () => {
  await expect(
    record(entry("stop", "stop", { evidence: [evaluation("a")], parents: ["d3"] })),
  ).rejects.toThrow("STOP_RULE_REQUIRED");
  await record(
    entry("stop", "stop", {
      evidence: [evaluation("a")],
      parents: ["d3"],
      body: { rule: "NO_FURTHER_IMPROVEMENT", selected: exp("a") },
    }),
  );
  const before = await c.owner.read(L, campaign, "getSnapshot");
  const closed = runtime.closed.catch(() => undefined);
  process.kill(runtime.connection.pid, "SIGKILL");
  await closed;
  runtime = await launch(authority, dataDir);
  c = clients(authority, runtime);
  expect(await c.owner.read(L, campaign, "getSnapshot")).toEqual(before);
  expect(before!.closedBy).toBe("stop");
  await expect(record(entry("late", "hypothesis", { parents: ["d3"] }))).rejects.toThrow(
    "LEDGER_CLOSED",
  );
  const events = (await c.owner.read(L, campaign, "getEvents", [0, 100]))!.events as Document[];
  expect(events.map((row) => row.sequence)).toEqual([1, 2, 3, 4, 5]);
  const entries = Object.values(before!.entries as Record<string, Document>);
  expect(entries.map((row) => row.sequence).sort()).toEqual([1, 2, 3, 4, 5]);
});
