import { afterAll, beforeAll, expect, test } from "bun:test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { LocalActorRuntime } from "durable-actors/dev";
import { clients, launch, localAuthority } from "../scripts/local-runtime.js";
import { digest, document, rewardCommitment, scopedId } from "../src/core.js";
import type { ActorName, Document, Role } from "../src/contracts.js";

const campaign = "integration";
const authority = localAuthority([campaign, "limits"]);
let runtime: LocalActorRuntime;
let c: ReturnType<typeof clients>;
let dataDir: string;
const C: ActorName = "ResearchCampaignActor",
  E: ActorName = "ExperimentActor";
const G: ActorName = "GovernanceActor",
  A: ActorName = "ArtifactRegistryActor",
  W: ActorName = "RLWorldActor";
const hash = (name: string) => digest(name);
const spec = {
  objective: "Controlled coordination fixture; no model training or measured improvement",
  model: { modelId: "fixture/model", revision: "a".repeat(40) },
  benchmark: "fixture-math",
  policyVersion: "v1",
  allowedStrategies: ["sft_lora"],
  maxBudgetMicroUsd: 200,
  maxConcurrent: 2,
  maxExperiments: 20,
  maxAttempts: 3,
  maxLeaseMs: 10000,
  maxRunMs: 120000,
  minImprovementBps: 500,
  baseline: {
    reportSha256: hash("baseline"),
    datasetSha256: hash("dev"),
    protocolSha256: hash("protocol"),
    numExamples: 10,
    numCorrect: 2,
  },
};
const exp = (label: string) => scopedId("exp", campaign, label);
const world = scopedId("world", campaign, "math");
const proposal = (label: string) => ({
  experimentId: exp(label),
  strategy: "sft_lora",
  model: spec.model,
  configSha256: hash(label),
  trainingDatasetSha256: hash("train"),
  developmentDatasetSha256: hash("dev"),
  hypothesis: "Coordination fixture",
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
const snapshot = async (actor: ActorName, actorId: string) =>
  (await c.owner.read(actor, actorId, "getSnapshot"))!;
let grant1: Document, grant2: Document;
let acceptedLabels: string[];

beforeAll(async () => {
  dataDir = await mkdtemp(join(tmpdir(), "forge-terse-test-"));
  runtime = await launch(authority, dataDir);
  c = clients(authority, runtime);
}, 120000);
afterAll(async () => {
  if (runtime) await runtime.stop();
  if (dataDir) await rm(dataDir, { recursive: true, force: true });
});

test("signed roles, scope, expiry, immutable objective and idempotent requests", async () => {
  await expect(command("worker", C, campaign, "initialize", spec)).rejects.toThrow("FORBIDDEN");
  const signed = c.owner.signed(C, campaign, campaign, "initialize", spec, "initialize");
  await expect(
    c.owner.send({ ...signed, payload: { ...spec, objective: "tampered" } }),
  ).rejects.toThrow("INVALID_SIGNATURE");
  await expect(
    c.owner.send(c.owner.signed(C, campaign, campaign, "initialize", spec, "expired", -1)),
  ).rejects.toThrow("EXPIRED_REQUEST");
  await expect(c.owner.command(C, campaign, "other", "initialize", spec)).rejects.toThrow(
    "FORBIDDEN",
  );
  const result = await c.owner.send(signed);
  expect(await command("owner", C, campaign, "initialize", spec, "initialize")).toEqual(result);
  await expect(
    command("owner", C, campaign, "initialize", { ...spec, objective: "changed" }, "initialize"),
  ).rejects.toThrow("IDEMPOTENCY_CONFLICT");
  await expect(command("owner", C, campaign, "initialize", spec)).rejects.toThrow(
    "OBJECTIVE_IMMUTABLE",
  );
  expect((await snapshot(C, campaign)).version).toBe(1);
  await command("owner", G, campaign, "initialize", {
    objectiveSha256: digest(spec),
    policyVersion: "v1",
  });
});

test("ten concurrent reservations cannot exceed budget; rejected transitions roll back", async () => {
  for (let i = 0; i < 10; i++) await command("director", C, campaign, "propose", proposal(`p${i}`));
  const results = await Promise.allSettled(
    Array.from({ length: 10 }, (_, i) =>
      command(
        "coordinator",
        C,
        campaign,
        "reserve",
        { experimentId: exp(`p${i}`) },
        `reserve-${i}`,
      ),
    ),
  );
  acceptedLabels = results.flatMap((r, i) => (r.status === "fulfilled" ? [`p${i}`] : []));
  expect(acceptedLabels).toHaveLength(2);
  const state = await snapshot(C, campaign);
  expect(state.reservedMicroUsd).toBe(200);
  expect(state.remainingMicroUsd).toBe(0);
  expect(state.version).toBe(13);
  const reservations = state.reservations as Record<string, Document>;
  grant1 = reservations[exp(acceptedLabels[0])].grant as Document;
  grant2 = reservations[exp(acceptedLabels[1])].grant as Document;
  const i = Number(acceptedLabels[0].slice(1));
  expect(
    await command(
      "coordinator",
      C,
      campaign,
      "reserve",
      { experimentId: exp(acceptedLabels[0]) },
      `reserve-${i}`,
    ),
  ).toEqual({ grant: grant1 });
  await expect(
    command("coordinator", C, campaign, "settle", {
      experimentId: exp(acceptedLabels[0]),
      actualMicroUsd: 0,
      costEvidenceSha256: hash("cost"),
    }),
  ).rejects.toThrow("EXPERIMENT_NOT_TERMINAL");
  expect((await snapshot(C, campaign)).version).toBe(13);
});

test("reservation grants, lease expiry and generations fence stale workers", async () => {
  const id = exp(acceptedLabels[0]);
  await expect(command("worker", E, id, "acquireLease", { leaseMs: 100 })).rejects.toThrow(
    "CAMPAIGN_SCOPE",
  );
  await expect(
    command("coordinator", E, id, "initialize", {
      grant: { ...grant1, signature: "0".repeat(64) },
    }),
  ).rejects.toThrow("INVALID_GRANT");
  await command("coordinator", E, id, "initialize", { grant: grant1 });
  const lease = await command("worker", E, id, "acquireLease", { leaseMs: 50 });
  await Bun.sleep(80);
  await expect(
    command("worker", E, id, "renewLease", { generation: lease.generation, leaseMs: 1000 }),
  ).rejects.toThrow("STALE_LEASE");
  const newer = await command("worker", E, id, "acquireLease", { leaseMs: 10000 });
  expect(newer.generation).toBe(2);
  await expect(
    command("worker", E, id, "renewLease", { generation: 1, leaseMs: 1000 }),
  ).rejects.toThrow("STALE_LEASE");
  await expect(command("worker", E, id, "acquireLease", { leaseMs: 1000 })).rejects.toThrow(
    "LEASE_HELD",
  );
  await command("worker", E, id, "renewLease", { generation: 2, leaseMs: 10000 });
  await expect(
    command("worker", E, id, "complete", { generation: 2, result: training("wrong") }),
  ).rejects.toThrow("CONFIG_MISMATCH");
});

function training(label: string): Document {
  return {
    checkpointArtifactId: `checkpoint-${label}`,
    checkpointSha256: hash(`weights-${label}`),
    manifestSha256: hash(`manifest-${label}`),
    configSha256: hash(label),
    weightsChanged: true,
  };
}
function evaluation(label: string): Document {
  return {
    reportArtifactId: `eval-${label}`,
    reportSha256: hash(`eval-${label}`),
    baselineReportSha256: hash("baseline"),
    checkpointSha256: hash(`weights-${label}`),
    datasetSha256: hash("dev"),
    protocolSha256: hash("protocol"),
    evaluationRole: "development",
    numExamples: 10,
    baselineCorrect: 2,
    candidateCorrect: 4,
  };
}
function review(label: string, approved: boolean): Document {
  return {
    experimentId: exp(label),
    checkpointSha256: hash(`weights-${label}`),
    evaluationReportSha256: hash(`eval-${label}`),
    objectiveSha256: digest(spec),
    policyVersion: "v1",
    decision: approved ? "approved" : "rejected",
    reasonCodes: approved ? [] : ["REGRESSION"],
    evidenceRefs: [hash("review-fixture")],
    checks: {
      checksumVerified: true,
      objectiveAligned: true,
      evaluationReproducible: true,
      regressionsPassed: approved,
      dataAllowed: true,
    },
  };
}
async function registerArtifacts(label: string) {
  for (const [kind, artifactId, sha] of [
    ["checkpoint", `checkpoint-${label}`, `weights-${label}`],
    ["evaluation", `eval-${label}`, `eval-${label}`],
  ]) {
    await command(kind === "checkpoint" ? "worker" : "evaluator", A, campaign, "register", {
      artifactId,
      experimentId: exp(label),
      kind,
      sha256: hash(sha),
      uri: `fixture://${artifactId}`,
      sizeBytes: 100,
      provenanceSha256: hash("fixture-provenance"),
    });
  }
}

test("evaluation and promotion require terminal work, settled cost, approval and artifacts", async () => {
  const label = acceptedLabels[0],
    id = exp(label);
  await expect(command("evaluator", E, id, "recordEvaluation", evaluation(label))).rejects.toThrow(
    "TRAINING_NOT_COMPLETED",
  );
  await command("worker", E, id, "complete", { generation: 2, result: training(label) });
  await expect(command("worker", E, id, "acquireLease", { leaseMs: 1000 })).rejects.toThrow(
    "TERMINAL_EXPERIMENT",
  );
  await expect(
    command("evaluator", E, id, "recordEvaluation", {
      ...evaluation(label),
      checkpointSha256: hash("wrong"),
    }),
  ).rejects.toThrow("EVALUATION_MISMATCH");
  await command("evaluator", E, id, "recordEvaluation", evaluation(label));
  await expect(command("director", C, campaign, "promote", { experimentId: id })).rejects.toThrow(
    "UNSETTLED_EXPERIMENT",
  );
  await expect(
    command("coordinator", C, campaign, "settle", {
      experimentId: id,
      actualMicroUsd: 101,
      costEvidenceSha256: hash("cost"),
    }),
  ).rejects.toThrow("BUDGET_EXCEEDED");
  const settled = await command(
    "coordinator",
    C,
    campaign,
    "settle",
    { experimentId: id, actualMicroUsd: 40, costEvidenceSha256: hash("cost") },
    "settled-first",
  );
  expect(
    await command(
      "coordinator",
      C,
      campaign,
      "settle",
      { experimentId: id, actualMicroUsd: 40, costEvidenceSha256: hash("cost") },
      "settled-first",
    ),
  ).toEqual(settled);
  await expect(command("director", C, campaign, "promote", { experimentId: id })).rejects.toThrow(
    "GOVERNANCE_VETO",
  );
  await expect(
    command("director", G, campaign, "recordDecision", review(label, true)),
  ).rejects.toThrow("FORBIDDEN");
  await command("reviewer", G, campaign, "recordDecision", review(label, true));
  await expect(command("director", C, campaign, "promote", { experimentId: id })).rejects.toThrow(
    "ARTIFACT_MISMATCH",
  );
  await registerArtifacts(label);
  const promoted = await command(
    "director",
    C,
    campaign,
    "promote",
    { experimentId: id },
    "promote-first",
  );
  expect(promoted.candidateCorrect).toBe(4);
  await expect(command("owner", "FinalEvaluationActor", campaign, "freeze", {})).rejects.toThrow(
    "CAMPAIGN_OPEN",
  );
  expect(
    await command("director", C, campaign, "promote", { experimentId: id }, "promote-first"),
  ).toEqual(promoted);
  await expect(command("director", C, campaign, "promote", { experimentId: id })).rejects.toThrow(
    "NO_VALID_IMPROVEMENT",
  );
});

test("early artifact registration and terminal veto remain authoritative", async () => {
  const label = acceptedLabels[1],
    id = exp(label);
  await registerArtifacts(label);
  await command("reviewer", G, campaign, "recordDecision", review(label, false));
  await expect(
    command("reviewer", G, campaign, "recordDecision", review(label, true)),
  ).rejects.toThrow("DECISION_FINAL");
  await command("coordinator", E, id, "initialize", { grant: grant2 });
  const lease = await command("worker", E, id, "acquireLease", { leaseMs: 10000 });
  await command("worker", E, id, "complete", {
    generation: lease.generation,
    result: training(label),
  });
  await command("evaluator", E, id, "recordEvaluation", evaluation(label));
  await command("coordinator", C, campaign, "settle", {
    experimentId: id,
    actualMicroUsd: 20,
    costEvidenceSha256: hash("cost2"),
  });
  await expect(command("director", C, campaign, "promote", { experimentId: id })).rejects.toThrow(
    "GOVERNANCE_VETO",
  );
  expect((await snapshot(C, campaign)).spentMicroUsd).toBe(60);
});

let worldAction: Document;
test("math worlds score exact answers once and enforce version, episode and role boundaries", async () => {
  const previous = process.env.FORGE_REWARD_SECRET;
  process.env.FORGE_REWARD_SECRET = authority.rewardSecret;
  let commitment: string;
  try {
    commitment = rewardCommitment(campaign, world, "sum", "v1", 7, "What is 20 + 22?", "42");
  } finally {
    if (previous === undefined) delete process.env.FORGE_REWARD_SECRET;
    else process.env.FORGE_REWARD_SECRET = previous;
  }
  const reset = {
    episodeId: "episode1",
    task: {
      taskId: "sum",
      taskVersion: "v1",
      seed: 7,
      question: "What is 20 + 22?",
      answerCommitment: commitment,
      rewardVersion: "exact-numeric-v1",
      maxSteps: 3,
    },
  };
  await expect(command("worker", W, world, "reset", reset)).rejects.toThrow("FORBIDDEN");
  const observation = await command("data", W, world, "reset", reset);
  expect(JSON.stringify(observation)).not.toContain(commitment);
  await command("data", W, world, "saveSnapshot", { snapshotId: "initial" });
  const calculation = {
    episodeId: "episode1",
    actionId: "calculate",
    expectedVersion: 0,
    action: { type: "calculate", operation: "add", left: "20", right: "22" },
  };
  const concurrent = await Promise.allSettled([
    command("worker", W, world, "step", calculation),
    command("worker", W, world, "step", { ...calculation, actionId: "competing" }),
  ]);
  expect(concurrent.filter((r) => r.status === "fulfilled")).toHaveLength(1);
  await expect(
    command("worker", W, world, "step", { ...calculation, actionId: "stale" }),
  ).rejects.toThrow("VERSION_CONFLICT");
  worldAction = {
    episodeId: "episode1",
    actionId: "answer",
    expectedVersion: 1,
    action: { type: "submit_answer", answer: "+0042.000" },
  };
  const answer = await command("worker", W, world, "step", worldAction, "answer-request");
  expect((answer.transition as Document).reward).toBe(1);
  expect((await command("worker", W, world, "step", worldAction)).replayed).toBe(true);
  expect((await command("worker", W, world, "step", worldAction, "answer-request")).replayed).toBe(
    true,
  );
  await expect(
    command("worker", W, world, "step", {
      ...worldAction,
      action: { type: "submit_answer", answer: "43" },
    }),
  ).rejects.toThrow("ACTION_CONFLICT");
  const trajectory = await command("worker", W, world, "getTrajectory", {});
  expect(trajectory.rewardTotal).toBe(1);
  expect(trajectory.transitions).toHaveLength(2);
  await expect(
    command("worker", W, scopedId("world", campaign, "other"), "observe", {}),
  ).rejects.toThrow("WORLD_SCOPE");
});

test("SIGKILL and native runtime restart preserve budgets, leases, veto, worlds and receipts", async () => {
  const label = "recover",
    id = exp(label);
  await command("director", C, campaign, "propose", proposal(label));
  const reserved = await command(
    "coordinator",
    C,
    campaign,
    "reserve",
    { experimentId: id },
    "reserve-recover",
  );
  await command("coordinator", E, id, "initialize", reserved);
  await command("worker", E, id, "acquireLease", { leaseMs: 50 });
  const before = await snapshot(C, campaign);
  const closed = runtime.closed.catch(() => undefined);
  process.kill(runtime.connection.pid, "SIGKILL");
  await closed;
  runtime = await launch(authority, dataDir);
  c = clients(authority, runtime);
  expect(await snapshot(C, campaign)).toEqual(before);
  expect(
    await command("coordinator", C, campaign, "reserve", { experimentId: id }, "reserve-recover"),
  ).toEqual(reserved);
  const lease = await command("worker", E, id, "acquireLease", { leaseMs: 10000 });
  expect(lease.generation).toBe(2);
  await expect(
    command("worker", E, id, "complete", { generation: 1, result: training(label) }),
  ).rejects.toThrow("STALE_LEASE");
  expect((await command("worker", W, world, "step", worldAction, "answer-request")).replayed).toBe(
    true,
  );
  expect((await command("worker", W, world, "getTrajectory", {})).rewardTotal).toBe(1);
  await expect(
    command("director", C, campaign, "promote", { experimentId: exp(acceptedLabels[1]) }),
  ).rejects.toThrow("GOVERNANCE_VETO");
  await command("coordinator", E, id, "abort", {
    ...reserved,
    reason: "Recovery fixture complete",
  });
  await command("coordinator", C, campaign, "settle", {
    experimentId: id,
    actualMicroUsd: 0,
    costEvidenceSha256: hash("abort-cost"),
  });
  expect((await snapshot(C, campaign)).reservedMicroUsd).toBe(0);
});

test("restoration forks history; wrong answers, truncation and stale episodes are explicit", async () => {
  const restored = await command("data", W, world, "restore", {
    snapshotId: "initial",
    newEpisodeId: "fork",
  });
  expect(restored.parentEpisodeId).toBe("episode1");
  await expect(
    command("worker", W, world, "step", { ...worldAction, actionId: "old" }),
  ).rejects.toThrow("EPISODE_CONFLICT");
  const wrong = await command("worker", W, world, "step", {
    episodeId: "fork",
    actionId: "wrong",
    expectedVersion: restored.stateVersion,
    action: { type: "submit_answer", answer: "1e100" },
  });
  expect((wrong.transition as Document).reward).toBe(0);
  const fork = await command("data", W, world, "restore", {
    snapshotId: "initial",
    newEpisodeId: "truncated",
  });
  for (let i = 0; i < 3; i++) {
    const result = await command("worker", W, world, "step", {
      episodeId: "truncated",
      actionId: `calc-${i}`,
      expectedVersion: Number(fork.stateVersion) + i,
      action: { type: "calculate", operation: "multiply", left: "2", right: "3" },
    });
    expect((result.transition as Document).truncated).toBe(i === 2);
  }
  await expect(
    command("worker", W, world, "step", {
      episodeId: "truncated",
      actionId: "too-late",
      expectedVersion: Number(fork.stateVersion) + 3,
      action: { type: "submit_answer", answer: "42" },
    }),
  ).rejects.toThrow("EPISODE_FINISHED");
});

test("unused grants abort safely, terminal states cannot restart, stop prevents new work", async () => {
  const label = "unused",
    id = exp(label);
  await command("director", C, campaign, "propose", proposal(label));
  const reservation = await command("coordinator", C, campaign, "reserve", { experimentId: id });
  await command("coordinator", E, id, "abort", { ...reservation, reason: "Never started" });
  await command("coordinator", C, campaign, "settle", {
    experimentId: id,
    actualMicroUsd: 0,
    costEvidenceSha256: hash("unused-cost"),
  });
  await expect(command("coordinator", E, id, "initialize", reservation)).rejects.toThrow(
    "EXPERIMENT_EXISTS",
  );
  await expect(command("worker", E, id, "acquireLease", { leaseMs: 1000 })).rejects.toThrow(
    "TERMINAL_EXPERIMENT",
  );
  await command("owner", C, campaign, "stop", { reason: "Tests complete" });
  await expect(command("director", C, campaign, "propose", proposal("stopped"))).rejects.toThrow(
    "CAMPAIGN_STOPPED",
  );
  const events = await c.owner.read(C, campaign, "getEvents", [0, 100]);
  const rows = events!.events as Document[];
  expect(rows.map((row) => row.sequence)).toEqual(rows.map((_, i) => i + 1));
  expect(rows.filter((row) => row.type === "CHECKPOINT_PROMOTED")).toHaveLength(1);
  expect(document(await snapshot(C, campaign)).spentMicroUsd).toBe(60);
});

test("final evaluation freezes once after stop, binds one release and records once", async () => {
  const F: ActorName = "FinalEvaluationActor";
  const best = (await snapshot(C, campaign)).best as Document;
  const release = {
    releaseSha256: hash("release"),
    checkpointSha256: best.checkpointSha256,
    developmentHarnessSha256: hash("development-harness"),
  };
  const final = {
    releaseSha256: release.releaseSha256,
    checkpointSha256: best.checkpointSha256,
    datasetSha256: hash("gsm8k-test"),
    harness: { name: "lm-evaluation-harness", version: "0.4.9.2", task: "gsm8k" },
    protocolSha256: hash("protocol"),
    numExamples: 1319,
    strict: { baselineCorrect: 400, candidateCorrect: 450 },
    flexible: { baselineCorrect: 500, candidateCorrect: 510 },
    finalImprovementSupported: true,
    resultBundleSha256: hash("result"),
  };
  await expect(command("director", F, campaign, "freeze", {})).rejects.toThrow("FORBIDDEN");
  await expect(command("owner", F, campaign, "bindRelease", release)).rejects.toThrow("NOT_FROZEN");
  const frozen = await command("owner", F, campaign, "freeze", {}, "freeze");
  expect(frozen.checkpointSha256).toBe(best.checkpointSha256);
  expect(await command("owner", F, campaign, "freeze", {}, "freeze")).toEqual(frozen);
  await expect(command("owner", F, campaign, "freeze", {})).rejects.toThrow("FREEZE_FINAL");
  await expect(command("final_evaluator", F, campaign, "recordFinal", final)).rejects.toThrow(
    "NOT_RELEASED",
  );
  await expect(
    command("owner", F, campaign, "bindRelease", { ...release, checkpointSha256: hash("other") }),
  ).rejects.toThrow("NOT_FROZEN_CHECKPOINT");
  await expect(command("director", F, campaign, "bindRelease", release)).rejects.toThrow(
    "FORBIDDEN",
  );
  await command("owner", F, campaign, "bindRelease", release);
  await expect(
    command("owner", F, campaign, "bindRelease", { ...release, releaseSha256: hash("swap") }),
  ).rejects.toThrow("RELEASE_FINAL");
  await expect(command("owner", F, campaign, "recordFinal", final)).rejects.toThrow("FORBIDDEN");
  await expect(
    command("final_evaluator", F, campaign, "recordFinal", {
      ...final,
      releaseSha256: hash("swap"),
    }),
  ).rejects.toThrow("NOT_FROZEN_RELEASE");
  await expect(
    command("final_evaluator", F, campaign, "recordFinal", { ...final, numExamples: 10 }),
  ).rejects.toThrow();
  const recorded = await command("final_evaluator", F, campaign, "recordFinal", final);
  expect(recorded.evaluatorKeyId).toBe("final_evaluator-local");
  await expect(
    command("final_evaluator", F, campaign, "recordFinal", {
      ...final,
      strict: { baselineCorrect: 400, candidateCorrect: 0 },
    }),
  ).rejects.toThrow("FINAL_ONCE");
  expect(((await snapshot(F, campaign)).final as Document).strict).toEqual(final.strict);
  const events = (await c.owner.read(F, campaign, "getEvents", [0, 10]))!.events as Document[];
  expect(events.map((event) => event.type)).toEqual([
    "CHECKPOINT_FROZEN",
    "RELEASE_BOUND",
    "FINAL_RECORDED",
  ]);
});

test("concurrency cap, bounded retries, protocol mismatch and strict RPC payloads", async () => {
  const scope = "limits";
  const call = (role: Role, actor: ActorName, id: string, method: string, payload: Document) =>
    c[role].command(actor, id, scope, method, payload);
  const policy = { ...spec, maxBudgetMicroUsd: 1000, maxConcurrent: 1, maxAttempts: 2 };
  await call("owner", C, scope, "initialize", policy);
  await call("owner", G, scope, "initialize", {
    objectiveSha256: digest(policy),
    policyVersion: "v1",
  });
  const first = scopedId("exp", scope, "retry"),
    second = scopedId("exp", scope, "protocol");
  const p = { ...proposal("retry"), experimentId: first };
  await expect(
    call("director", C, scope, "propose", { ...p, maxCostMicroUsd: 0.1 }),
  ).rejects.toThrow();
  await expect(call("director", C, scope, "propose", { ...p, unknown: true })).rejects.toThrow();
  await call("director", C, scope, "propose", p);
  await call("director", C, scope, "propose", { ...proposal("protocol"), experimentId: second });
  const reservation = await call("coordinator", C, scope, "reserve", { experimentId: first });
  await expect(call("coordinator", C, scope, "reserve", { experimentId: second })).rejects.toThrow(
    "CONCURRENCY_LIMIT",
  );
  await call("coordinator", E, first, "initialize", reservation);
  for (let i = 1; i <= 2; i++) {
    await call("worker", E, first, "acquireLease", { leaseMs: 10000 });
    const failed = await call("worker", E, first, "reportFailure", {
      generation: i,
      reason: "Injected worker failure",
    });
    expect(failed.status).toBe(i === 1 ? "retryable" : "failed");
  }
  await expect(call("worker", E, first, "acquireLease", { leaseMs: 10000 })).rejects.toThrow(
    "TERMINAL_EXPERIMENT",
  );
  await call("coordinator", C, scope, "settle", {
    experimentId: first,
    actualMicroUsd: 1,
    costEvidenceSha256: hash("failed-cost"),
  });
  const next = await call("coordinator", C, scope, "reserve", { experimentId: second });
  await call("coordinator", E, second, "initialize", next);
  await call("worker", E, second, "acquireLease", { leaseMs: 10000 });
  await call("worker", E, second, "complete", { generation: 1, result: training("protocol") });
  await call("evaluator", E, second, "recordEvaluation", {
    ...evaluation("protocol"),
    protocolSha256: hash("wrong-protocol"),
  });
  await call("coordinator", C, scope, "settle", {
    experimentId: second,
    actualMicroUsd: 1,
    costEvidenceSha256: hash("protocol-cost"),
  });
  for (const [kind, artifactId, sha] of [
    ["checkpoint", "checkpoint-protocol", "weights-protocol"],
    ["evaluation", "eval-protocol", "eval-protocol"],
  ]) {
    await call(kind === "checkpoint" ? "worker" : "evaluator", A, scope, "register", {
      artifactId,
      experimentId: second,
      kind,
      sha256: hash(sha),
      uri: `fixture://${artifactId}`,
      sizeBytes: 1,
      provenanceSha256: hash("fixture"),
    });
  }
  await call("reviewer", G, scope, "recordDecision", {
    ...review("protocol", true),
    experimentId: second,
    objectiveSha256: digest(policy),
  });
  // Even a trusted review cannot override the campaign's frozen evaluation protocol.
  await expect(call("director", C, scope, "promote", { experimentId: second })).rejects.toThrow(
    "BASELINE_MISMATCH",
  );
  expect((await snapshot(C, scope)).best).toBeNull();
});
