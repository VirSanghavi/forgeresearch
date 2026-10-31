import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { mkdtemp, readdir, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Document } from "../../../packages/terse-actors/src/contracts.js";
import { runResearch } from "../scripts/research.js";
import { readJson } from "../src/worker.js";

// Autonomous GRPO campaign: real Relaycast, Terse and Python processes on the warm-started
// offline tiny fixture. Rewards come from RLWorldActor transitions; no human selection.
const python = process.env.FORGE_PYTHON;
let output: string;
let report: Document;
let experiments: Document[];

beforeAll(async () => {
  if (!python) throw new Error("Set FORGE_PYTHON to the training-core virtualenv Python");
  output = await mkdtemp(join(tmpdir(), "forge-phase6-test-"));
  report = await runResearch(output, python, "fixture-grpo");
  experiments = report.experiments as Document[];
}, 600000);
afterAll(async () => {
  if (output) await rm(output, { recursive: true, force: true });
});

test("every experiment is reward-driven, preflight-approved and evaluated", () => {
  expect(report.method).toBe("grpo");
  expect(experiments.length).toBeGreaterThanOrEqual(2);
  for (const experiment of experiments) {
    expect(experiment.strategy).toBe("grpo");
    expect((experiment.preflight as Document).approved).toBe(true);
    expect(experiment.status).toBe("evaluated");
  }
});

test("each run's rewards are exactly its world transitions, read from the journals", () => {
  let total = 0;
  for (const experiment of experiments) {
    const rewards = experiment.rewards as Document;
    expect(rewards.transitionsInJournals).toBe(rewards.rollouts);
    expect(rewards.rollouts).toBe(Number(rewards.updates) * 8);
    expect(Number(rewards.rewarded)).toBeGreaterThan(0);
    expect(Number(rewards.rewarded)).toBeLessThan(Number(rewards.rollouts));
    // Lineage: the initial adapter and one snapshot per update, ending at the checkpoint.
    const lineage = rewards.lineage as Document;
    expect(Object.keys(lineage.snapshots as Document)).toHaveLength(Number(rewards.updates) + 1);
    expect((lineage.snapshots as Document)[String(rewards.updates)]).toBe(
      lineage.final_policy_sha256,
    );
    total += Number(rewards.rollouts);
  }
  // Every gateway transition belongs to a reported rollout; nothing was replayed or retried.
  const gateways = report.worldGateways as Document[];
  expect(gateways).toHaveLength(experiments.length);
  expect(gateways.reduce((sum, g) => sum + Number(g.transitions), 0)).toBe(total);
  expect(gateways.every((g) => g.replayed === 0 && g.retries === 0)).toBe(true);
});

test("the trainer only ever received answer-free tasks", async () => {
  for (const experiment of experiments) {
    const staged = await readFile(
      join(output, "experiments", String(experiment.label), "run/inputs/train.jsonl"),
      "utf8",
    );
    for (const line of staged.trimEnd().split("\n"))
      expect(Object.keys(JSON.parse(line)).sort()).toEqual(["id", "question"]);
  }
});

test("governance re-derives reward provenance before its terminal decision", async () => {
  const governance = report.governance as Document[];
  expect(governance.length).toBeGreaterThanOrEqual(1);
  for (const record of governance) {
    const evidence = JSON.parse(
      await readFile(join(output, "governance", `${record.experimentId}.json`), "utf8"),
    );
    const world = evidence.facts.worldRewards;
    expect(world.transitionsInJournals).toBe(world.rollouts);
    expect(record.reasonCodes).not.toContain("REWARD_PROVENANCE");
    expect(evidence.checks.dataAllowed).toBe(true);
  }
  for (const veto of report.vetoEnforcement as Document[])
    expect([veto.director, veto.coordinator]).toEqual(["GOVERNANCE_VETO", "GOVERNANCE_VETO"]);
  expect(report.campaignStatus).toBe("stopped");
  expect(report.reconciliation).toEqual({
    expiredLeases: [],
    unacknowledged: [],
    unannouncedCompletions: [],
    unsent: [],
  });
});

test("later decisions cite measured evidence of earlier GRPO runs", () => {
  const ledger = report.ledger as Document[];
  const decisions = ledger.filter((e) => e.kind === "decision" && e.launch);
  expect(String(decisions[0].summary)).toContain("world-rewarded task recipes");
  const later = decisions.find((e) => (e.launch as Document[]).length && e !== decisions[0]);
  if (later) {
    const cited = (later.evidence as Document[]).filter((e) => e.source === "evaluation");
    expect(cited.length).toBeGreaterThanOrEqual(2);
  }
  expect(ledger.at(-1)!.kind).toBe("stop");
});

describe("SFT then GRPO: a campaign that continues an adapter", () => {
  let continued: string;
  let result: Document;
  beforeAll(async () => {
    continued = await mkdtemp(join(tmpdir(), "forge-phase6-continued-"));
    result = await runResearch(continued, python!, "fixture-grpo", { fromSft: true });
  }, 600000);
  afterAll(async () => {
    if (continued) await rm(continued, { recursive: true, force: true });
  });

  test("baselines are the starting adapter, so gains are GRPO's own", async () => {
    const start = result.startingAdapter as Document;
    expect(String(start.sha256)).toMatch(/^[a-f0-9]{64}$/);
    const baselines = (await readdir(continued)).filter((f) => f.startsWith("frozen-baseline"));
    for (const name of baselines)
      expect((await readJson(join(continued, name))).checkpoint_sha256).toBe(start.sha256);
    const regression = await readJson(join(continued, "governance", "regression-baseline.json"));
    expect(regression.checkpoint_sha256).toBe(start.sha256);
    const best = result.bestCheckpointPromoted as Document | null;
    if (best)
      expect(Number(best.candidateCorrect)).toBeGreaterThan(
        Number((result.baseline as Document).numCorrect),
      );
  });

  test("every run starts from the adapter's weights and is rewarded only by the worlds", () => {
    const experiments = result.experiments as Document[];
    const starts = new Set(
      experiments.map(
        (e) => ((e.rewards as Document).lineage as Document).initial_policy_sha256 as string,
      ),
    );
    expect(starts.size).toBe(1);
    for (const experiment of experiments) {
      const rewards = experiment.rewards as Document;
      expect(experiment.status).toBe("evaluated");
      expect(rewards.transitionsInJournals).toBe(rewards.rollouts);
    }
  });
});
