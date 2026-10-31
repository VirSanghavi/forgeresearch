import { afterAll, beforeAll, expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Document } from "../../../packages/terse-actors/src/contracts.js";
import { runResearch } from "../scripts/research.js";

// CONTROLLED FAILURE INJECTION: the first checkpoint sent to governance is altered after
// registration. Real Relaycast, Terse and Python processes; offline tiny fixture model.
const python = process.env.FORGE_PYTHON;
let output: string;
let report: Document;
let governance: Document[];

beforeAll(async () => {
  if (!python) throw new Error("Set FORGE_PYTHON to the training-core virtualenv Python");
  output = await mkdtemp(join(tmpdir(), "forge-phase5-test-"));
  report = await runResearch(output, python, "fixture", { tamperFirstReviewedCheckpoint: true });
  governance = report.governance as Document[];
}, 600000);
afterAll(async () => {
  if (output) await rm(output, { recursive: true, force: true });
});

test("an invalid candidate is vetoed with preserved, hash-bound evidence", async () => {
  const [vetoed] = governance;
  expect(report.faultsInjected).toEqual([`INJECTED_CHECKPOINT_TAMPER:${vetoed.label}`]);
  expect(vetoed.decision).toBe("rejected");
  expect(vetoed.reasonCodes).toContain("ARTIFACT_IDENTITY");
  const path = join(output, "governance", `${vetoed.experimentId}.json`);
  const bytes = await readFile(path);
  expect(createHash("sha256").update(bytes).digest("hex")).toBe(String(vetoed.evidenceSha256));
  const evidence = JSON.parse(bytes.toString());
  expect(evidence.checks.checksumVerified).toBe(false);
  const checkpoint = evidence.facts.checkpoint;
  expect(checkpoint.recomputed).not.toBe(checkpoint.recordedByExperiment);
  expect(checkpoint.recordedByRegistry).toBe(checkpoint.recordedByExperiment);
});

test("every promotion path honors the veto", () => {
  expect(report.vetoEnforcement).toEqual([
    { label: governance[0].label, director: "GOVERNANCE_VETO", coordinator: "GOVERNANCE_VETO" },
  ]);
});

test("the director takes a permitted corrective action recorded in the ledger", () => {
  const ledger = report.ledger as Document[];
  const corrective = ledger.find((e) => e.entryId === `governance-veto-${governance[0].label}`)!;
  expect(corrective.kind).toBe("decision");
  expect(String(corrective.summary)).toContain(
    `review the next-ranked candidate ${governance[1].label}`,
  );
  expect((corrective.rejected as Document[])[0].option).toBe(governance[0].label);
  expect((corrective.evidence as Document[]).some((e) => e.source === "evaluation")).toBe(true);
  expect(ledger.indexOf(corrective)).toBeLessThan(ledger.findIndex((e) => e.kind === "stop"));
});

test("the fallback is independently approved; the improvement bar still applies", async () => {
  const approved = governance[1];
  expect(approved.decision).toBe("approved");
  const evidence = JSON.parse(
    await readFile(join(output, "governance", `${approved.experimentId}.json`), "utf8"),
  );
  expect(Object.values(evidence.checks).every(Boolean)).toBe(true);
  expect(evidence.facts.regressionSuite.numExamples).toBe(2);
  // The tiny fixture cannot beat its baseline, so an approved candidate is still refused.
  expect(approved.promotion).toBe("NO_VALID_IMPROVEMENT");
  expect(report.bestCheckpointPromoted).toBeNull();
  expect(report.campaignStatus).toBe("stopped");
});
