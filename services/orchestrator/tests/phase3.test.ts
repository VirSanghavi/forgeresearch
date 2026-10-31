import { afterAll, beforeAll, expect, test } from "bun:test";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Document } from "../../../packages/terse-actors/src/contracts.js";
import { runScenario } from "../scripts/scenario.js";

// Real Relaycast, Terse and Python optimization; requires the training-core environment.
const python = process.env.FORGE_PYTHON;
let output: string;
let report: Document;

beforeAll(async () => {
  if (!python) throw new Error("Set FORGE_PYTHON to the training-core virtualenv Python");
  output = await mkdtemp(join(tmpdir(), "forge-phase3-test-"));
  report = await runScenario(output, python);
}, 600000);
afterAll(async () => {
  if (output) await rm(output, { recursive: true, force: true });
});

test("a Relay-delivered assignment causes exactly one authorized optimization run", () => {
  expect(report.preflightBlocked).toBe(true);
  expect(report.workerReceipt).toEqual({
    viaRelay: 1,
    viaReconciliation: 0,
    duplicateCopies: 1,
    rejectedCopies: 1,
    alreadyHandled: 0,
  });
  expect(report.optimizationRuns).toBe(1);
  expect(report.workerAttempts).toBe(1);
  expect(report.weightsChanged).toBe(true);
  const kinds = (report.messages as Document[]).map((m) => m.kind);
  expect(kinds.filter((kind) => kind === "TRAINING_COMPLETED")).toHaveLength(1);
});

test("messages dispatch in causal order and every one is sent and acknowledged", () => {
  const messages = report.messages as Document[];
  expect(messages.map((m) => m.kind)).toEqual([
    "RESEARCH_ASSIGNMENT",
    "EXPERIMENT_PROPOSAL",
    "REVIEW_REQUEST",
    "PREFLIGHT_REVIEWED",
    "TRAINING_ASSIGNMENT",
    "TRAINING_COMPLETED",
    "EVALUATION_COMPLETED",
    "CHECKPOINT_REVIEWED",
    "CORRECTION",
  ]);
  const sequences = messages.map((m) => Number(m.sequence));
  expect(sequences).toEqual([...sequences].sort((a, b) => a - b));
  expect(messages.every((m) => m.acknowledged && m.relayMessageId)).toBe(true);
});

test("restarts, a post-completion crash and a missed page leave pending work discoverable", () => {
  expect(report.correctionReceipt).toEqual({
    viaRelay: 0,
    viaReconciliation: 1,
    duplicateCopies: 0,
    rejectedCopies: 0,
    alreadyHandled: 0,
  });
  expect((report.staleReceipt as Document).alreadyHandled).toBe(1);
  expect(Object.values(report.relayInboxesAfterRun as Document)).toEqual([0, 0, 0, 0, 0, 0]);
  const { afterRestart, earlyCompletion, final } = report.reconciliation as Record<
    string,
    Document
  >;
  expect(afterRestart.unacknowledged).toEqual(["train"]);
  expect(earlyCompletion.unannouncedCompletions).toHaveLength(1);
  expect(earlyCompletion.unacknowledged).toEqual(["train"]);
  expect(final).toEqual({
    unsent: [],
    unacknowledged: [],
    unannouncedCompletions: [],
    expiredLeases: [],
  });
});

test("evaluation is reproduced, governance veto holds and budget is settled", async () => {
  expect(report.experimentStatus).toBe("evaluated");
  expect(report.evaluatorReproducedWorkerPredictions).toBe(true);
  expect(report.governanceDecision).toBe("rejected");
  expect(report.promotionBlocked).toBe("GOVERNANCE_VETO");
  expect(report.bestCheckpoint).toBeNull();
  expect(report.reservedMicroUsd).toBe(0);
  expect(report.spentMicroUsd).toBe(100000);
  expect(report.modelImprovementClaimed).toBe(false);
  expect(report.autonomousResearchClaimed).toBe(false);
  // Stored evidence and the reported development counts agree.
  const evaluator = JSON.parse(await readFile(join(output, "evaluator/candidate.json"), "utf8"));
  const development = report.development as Document;
  expect(development.candidateCorrect).toBe(evaluator.num_correct);
  expect(development.numExamples).toBe(evaluator.num_examples);
});
