import { afterAll, beforeAll, expect, test } from "bun:test";
import { spawn } from "node:child_process";
import { appendFile, cp, mkdtemp, readFile, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Document } from "../../../packages/terse-actors/src/contracts.js";
import { runResearch } from "../scripts/research.js";
import { bundleCampaign, reproduceBundle, verifyBundle } from "../src/bundle.js";
import { fileHash, pythonJson, readJson } from "../src/worker.js";

// Phase 8 on the offline constant fixture, through real Relaycast, Terse and Python: freeze
// after the campaign stopped, training-core's pinned harness (development comparison,
// governance-checked release, rehearsal), the evidence bundle and a clean-copy reproduction.
// Fixture releases are refused final test access, so no sealed result exists here.
const python = process.env.FORGE_PYTHON;
const harnessPython = process.env.FORGE_HARNESS_PYTHON ?? python;
let output: string;
let report: Document;
let outcome: Document;
let live: Document;

beforeAll(async () => {
  if (!python) throw new Error("Set FORGE_PYTHON to the training-core virtualenv Python");
  output = await mkdtemp(join(tmpdir(), "forge-phase8-test-"));
  report = await runResearch(output, python, "fixture-final", {
    finalize: { harnessPython: harnessPython! },
  });
  outcome = await readJson(join(output, "final/outcome.json"));
  live = await readJson(join(output, "live/state.json"));
}, 600000);
afterAll(async () => {
  if (output) await rm(output, { recursive: true, force: true });
});

test("the promoted checkpoint is frozen after the stop and exactly its release is bound", async () => {
  expect(report.phase).toBe(8);
  expect(report.campaignStatus).toBe("stopped");
  expect(outcome.status).toBe("rehearsed");
  const final = live.final as Document;
  const frozen = final.frozen as Document,
    bound = final.release as Document;
  expect(frozen.checkpointSha256).toBe(
    (report.bestCheckpointPromoted as Document).checkpointSha256,
  );
  const release = await readJson(join(output, "final/release/release.json"));
  expect((release.candidate as Document).sha256).toBe(frozen.checkpointSha256);
  expect(release.official_test_loaded).toBe(false);
  expect(bound.releaseSha256).toBe(await fileHash(join(output, "final/release/bundle.json")));
  expect(Number(bound.boundAtMs)).toBeGreaterThanOrEqual(Number(frozen.frozenAtMs));
  // The release copied the report as it stood before finalization.
  expect((await readJson(join(output, "final/release/campaign.json"))).final).toBeNull();
});

test("research roles cannot freeze, swap the release or record; fixtures never reach final data", () => {
  const probes = outcome.probes as Record<string, string>;
  expect(probes.directorFreeze).toBe("FORBIDDEN");
  expect(probes.secondFreeze).toBe("FREEZE_FINAL");
  expect(probes.releaseSwap).toBe("RELEASE_FINAL");
  expect(probes.researchFinalRecord).toBe("FORBIDDEN");
  expect(probes.rehearsalOverwrite).toStartWith("FileExistsError");
  expect(probes.fixtureFinalAccess).toContain("Fixture models may rehearse but cannot");
  expect((outcome.sealed as Document).status).toBe("not-run");
});

test("the pinned harness compares both policies under one protocol", async () => {
  for (const name of ["development", "rehearsal"]) {
    const [baseline, candidate] = await Promise.all(
      ["baseline", "candidate"].map((side) =>
        readJson(join(output, "final", name, `${side}.json`)),
      ),
    );
    expect(baseline.protocol).toEqual(candidate.protocol);
    expect(baseline.runtime).toEqual(candidate.runtime);
    expect(((baseline.runtime as Document).lm_eval as Document).version).toBe("0.4.9.2");
    expect(candidate.checkpoint_sha256).toBe(
      ((live.final as Document).frozen as Document).checkpointSha256,
    );
    const comparison = await readJson(join(output, "final", name, "comparison.json"));
    const strict = (comparison.metrics as Document).strict as Document;
    expect(strict.candidate_correct).toBeGreaterThan(Number(strict.baseline_correct));
    expect(comparison.final_improvement_supported).toBe(false);
  }
});

test("bundle claims and gate checks hold, and a clean copy reproduces the rehearsal", async () => {
  const built = await bundleCampaign(output, harnessPython!);
  const checks = built.checks as Record<string, Document>;
  expect(Object.fromEntries(Object.entries(checks).map(([k, v]) => [k, v.status]))).toEqual({
    sealedResultsDidNotInfluenceSelection: "passed",
    harnessEvidenceVerified: "passed",
    finalEvaluationRunsOnce: "passed",
    sealedFinalEvaluation: "not-run",
    allClaimsMapToArtifacts: "passed",
    recordingsMatchTheRun: "not-recorded",
    currentEventRules: "unverified",
  });
  expect(checks.currentEventRules.record).toBeUndefined();
  const copy = await mkdtemp(join(tmpdir(), "forge-phase8-copy-"));
  try {
    await cp(join(output, "final/bundle"), join(copy, "bundle"), { recursive: true });
    const intact = await verifyBundle(join(copy, "bundle"));
    expect([intact.changed, intact.unlisted, (intact.claims as Document).failed]).toEqual([
      [],
      [],
      [],
    ]);
    expect(
      await reproduceBundle(join(copy, "bundle"), harnessPython!, join(copy, "again")),
    ).toEqual({
      reproduced: true,
      identicalGenerations: { baseline: true, candidate: true },
      identicalComparison: true,
    });
    await appendFile(join(copy, "bundle/final/release/checkpoint/adapter_config.json"), " ");
    expect((await verifyBundle(join(copy, "bundle"))).changed).toEqual([
      "final/release/checkpoint/adapter_config.json",
    ]);
  } finally {
    await rm(copy, { recursive: true, force: true });
  }
}, 300000);

test("the separate final evaluator refuses a fixture release and records nothing", async () => {
  const scratch = await mkdtemp(join(tmpdir(), "forge-phase8-authority-"));
  try {
    const authority = join(scratch, "authority");
    await pythonJson(harnessPython!, [
      "-m",
      "forge_training.final_cli",
      "init-authority",
      "--authority",
      authority,
    ]);
    const child = spawn(
      join(import.meta.dir, "../node_modules/.bin/bun"),
      ["scripts/final-evaluate.ts", output, "--authority", authority],
      {
        cwd: join(import.meta.dir, ".."),
        env: { ...process.env, FORGE_HARNESS_PYTHON: harnessPython },
        stdio: ["ignore", "pipe", "pipe"],
      },
    );
    let stderr = "";
    child.stderr!.on("data", (chunk) => (stderr += chunk));
    const code = await new Promise<number>((done) => child.once("close", (c) => done(c ?? 1)));
    expect(code).toBe(1);
    expect(stderr).toContain("Fixture models may rehearse but cannot");
    expect((await readJson(join(output, "final/outcome.json"))).status).toBe("rehearsed");
    expect(await stat(join(output, "final/final-record.json")).catch(() => null)).toBeNull();
    expect(await stat(join(output, "final/evaluation")).catch(() => null)).toBeNull();
  } finally {
    await rm(scratch, { recursive: true, force: true });
  }
}, 300000);

test("the report never claims improvement from fixture mechanics", async () => {
  expect(report.modelImprovementClaimed).toBe(false);
  expect((await readFile(join(output, "final/bundle/README.md"), "utf8")).toString()).toContain(
    "Fixture model",
  );
});
