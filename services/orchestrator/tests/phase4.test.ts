import { afterAll, beforeAll, expect, test } from "bun:test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Document } from "../../../packages/terse-actors/src/contracts.js";
import { runResearch } from "../scripts/research.js";

// Real Relaycast, Terse and Python optimization on the offline tiny fixture. The fixture
// cannot learn arithmetic, so this checks decision mechanics, not model improvement.
const python = process.env.FORGE_PYTHON;
let output: string;
let report: Document;
let ledger: Document[];
const entry = (entryId: string) => ledger.find((e) => e.entryId === entryId)!;

beforeAll(async () => {
  if (!python) throw new Error("Set FORGE_PYTHON to the training-core virtualenv Python");
  output = await mkdtemp(join(tmpdir(), "forge-phase4-test-"));
  report = await runResearch(output, python, "fixture");
  ledger = report.ledger as Document[];
}, 600000);
afterAll(async () => {
  if (output) await rm(output, { recursive: true, force: true });
});

test("the director diagnoses the baseline and explores two recipes", () => {
  expect(ledger[0]).toMatchObject({
    entryId: "diagnosis",
    kind: "diagnosis",
    evidence: [{ source: "baseline", reportSha256: (report.baseline as Document).reportSha256 }],
  });
  const first = entry("decision-1");
  const launched = first.launch as Document[];
  expect(launched.map((l) => (l.recipe as Document).data).sort()).toEqual(["concise", "full"]);
  expect(launched.every((l) => l.operator === "explore")).toBe(true);
  for (const l of launched) expect(first.parents).toContain(`hypothesis-${l.label}`);
});

test("a later launch is chosen from recorded measurements of earlier experiments", () => {
  const second = entry("decision-2");
  const round1 = (entry("decision-1").launch as Document[]).map((l) => l.experimentId);
  const cited = (second.evidence as Document[])
    .filter((e) => e.source === "evaluation")
    .map((e) => e.experimentId);
  expect(cited.sort()).toEqual([...round1].sort());
  const [revision] = second.launch as Document[];
  const rejected = second.rejected as Document[];
  // The losing exploration is a recorded rejected alternative, with a measured reason.
  const winner = String(revision.parent);
  const loser = (entry("decision-1").launch as Document[]).find((l) => l.label !== winner)!;
  expect(rejected.find((r) => r.option === loser.label)?.reason).toBeTruthy();
  // The chosen operator is the one the measurements indicate, and others were possible.
  const counterfactuals = second.counterfactuals as Document;
  expect(Object.values(counterfactuals)).toContain(revision.operator);
  expect(new Set(Object.values(counterfactuals).filter(Boolean)).size).toBeGreaterThan(1);
  expect(rejected.filter((r) => String(r.option).endsWith(`:${winner}`))).toHaveLength(2);
});

test("every launch passed independent preflight and reached a recorded evaluation", () => {
  const experiments = report.experiments as Document[];
  expect(experiments.length).toBeGreaterThanOrEqual(3);
  for (const experiment of experiments) {
    expect(experiment.status).toBe("evaluated");
    expect(experiment.attempts).toBe(1);
    expect(experiment.preflight).toMatchObject({ approved: true });
    expect(String((experiment.preflight as Document).reason)).toContain("recomputed recipe");
  }
});

test("a declared stop rule closes the campaign with budget settled", () => {
  const stop = ledger.at(-1)!;
  expect(stop.kind).toBe("stop");
  expect([
    "NO_FURTHER_IMPROVEMENT",
    "REVISION_LIMIT",
    "EXPERIMENT_LIMIT",
    "BUDGET_EXHAUSTED",
    "SEARCH_SPACE_EXHAUSTED",
  ]).toContain(String((report.stop as Document).rule));
  expect(report.campaignStatus).toBe("stopped");
  expect(report.reservedMicroUsd).toBe(0);
  expect(report.spentMicroUsd).toBe(100000 * (report.experiments as Document[]).length);
  expect(report.bestCheckpointPromoted).toBeNull();
  expect(report.reconciliation).toEqual({
    unsent: [],
    unacknowledged: [],
    unannouncedCompletions: [],
    expiredLeases: [],
  });
  expect(report.humanSelectedExperiments).toBe(false);
  expect(report.modelImprovementClaimed).toBe(false);
});
