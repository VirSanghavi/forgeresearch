import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { cp, mkdir, readdir, readFile, stat, writeFile } from "node:fs/promises";
import { dirname, join, relative, sep } from "node:path";
import type { Document } from "../../../packages/terse-actors/src/contracts.js";
import { pythonJson } from "./worker.js";

/**
 * Phase 8 evidence bundle, built from stored files only. Every claim names the artifact,
 * the JSON pointer and the artifact's SHA-256; `verifyBundle` re-reads each one. The gate
 * checks are recorded with their observations, including what could not be verified.
 * Harness bundles (development comparison, release, rehearsal, final) are re-verified with
 * training-core's own `forge-harness verify`.
 */
const sha256 = (bytes: Buffer | string) => createHash("sha256").update(bytes).digest("hex");
const fileSha = async (path: string) => sha256(await readFile(path));
const readJson = async (path: string) => JSON.parse(await readFile(path, "utf8")) as Document;
async function exists(path: string) {
  return stat(path).then(
    () => true,
    () => false,
  );
}
function pointer(document: unknown, path: string): unknown {
  return path
    .split("/")
    .slice(1)
    .reduce<unknown>(
      (value, key) => (value as Record<string, unknown> | undefined)?.[key],
      document,
    );
}

export interface Claim {
  id: string;
  statement: string;
  value: unknown;
  artifact: string;
  pointer: string;
  sha256: string;
}

/** Re-reads every claimed value from its artifact; any difference fails the claim. */
export async function verifyClaims(
  root: string,
  claims: Claim[],
): Promise<{ verified: number; failed: string[] }> {
  const failed: string[] = [];
  for (const c of claims) {
    const path = join(root, c.artifact);
    if (!(await exists(path)) || (await fileSha(path)) !== c.sha256)
      failed.push(`${c.id}: artifact changed`);
    else if (JSON.stringify(pointer(await readJson(path), c.pointer)) !== JSON.stringify(c.value))
      failed.push(`${c.id}: value differs`);
  }
  return { verified: claims.length - failed.length, failed };
}

async function walk(directory: string): Promise<string[]> {
  const files: string[] = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) files.push(...(await walk(path)));
    else if (entry.isFile()) files.push(path);
  }
  return files;
}
async function tar(archive: string, directory: string, name: string): Promise<void> {
  const code = await new Promise<number>((resolve) =>
    spawn("tar", ["-czf", archive, "-C", directory, name], { stdio: "ignore" }).once("close", (c) =>
      resolve(c ?? 1),
    ),
  );
  if (code !== 0) throw new Error(`BUNDLE_ARCHIVE_FAILED: ${archive}`);
}
/** training-core's verification of a sealed harness bundle; returns its bundle hash. */
async function verifyHarness(python: string, directory: string, expected?: string) {
  const result = await pythonJson(python, [
    "-m",
    "forge_training.harness_cli",
    "verify",
    "--bundle",
    directory,
    ...(expected ? ["--expected-hash", expected] : []),
  ]);
  return String(result.bundle_sha256);
}

/** The probe outcomes `finalize` must observe: every attempt refused for the right reason. */
const expectedProbes: Record<string, (value: string) => boolean> = {
  directorFreeze: (v) => v === "FORBIDDEN",
  secondFreeze: (v) => v === "FREEZE_FINAL",
  releaseSwap: (v) => v === "RELEASE_FINAL",
  researchFinalRecord: (v) => v === "FORBIDDEN",
  rehearsalOverwrite: (v) => v.startsWith("FileExistsError"),
  fixtureFinalAccess: (v) => v.includes("Fixture models may rehearse but cannot"),
};

export async function bundleCampaign(root: string, python: string): Promise<Document> {
  const live = await readJson(join(root, "live/state.json"));
  const outcome = await readJson(join(root, "final/outcome.json"));
  if (outcome.status !== "rehearsed" && outcome.status !== "evaluated")
    throw new Error(`NOTHING_TO_BUNDLE: ${String(outcome.reason ?? outcome.status)}`);
  const actorFinal = live.final as Document;
  const frozen = actorFinal.frozen as Document;
  const bound = actorFinal.release as Document;
  const campaign = live.campaign as Document;
  const best = campaign.best as Document;
  const evaluated = outcome.status === "evaluated";
  const label = String(frozen.experimentId).replace(/^exp-[a-f0-9]{32}-/, "");
  const governance = `governance/${String(frozen.experimentId)}.json`;
  // The development baseline the campaign used is the report whose hash the live state names.
  let baselineFile = "";
  for (const name of (await readdir(root)).filter((n) => n.startsWith("frozen-baseline")))
    if ((await fileSha(join(root, name))) === (live.baseline as Document).reportSha256)
      baselineFile = name;
  if (!baselineFile) throw new Error("BUNDLE_EVIDENCE: the campaign's baseline report is missing");

  // Claims: every number a report or demo states, tied to its artifact.
  const statements: [string, string, string, string][] = [
    [
      "baseline-development",
      "Starting policy, development, custom scorer: correct answers",
      baselineFile,
      "/num_correct",
    ],
    [
      "promoted-checkpoint",
      "Checkpoint the campaign promoted",
      "live/state.json",
      "/campaign/best/checkpointSha256",
    ],
    [
      "promoted-development",
      "Promoted checkpoint, development, custom scorer: correct answers",
      "live/state.json",
      "/campaign/best/candidateCorrect",
    ],
    [
      "governance-approval",
      "Governance decision on the promoted checkpoint",
      governance,
      "/decision",
    ],
    [
      "frozen-checkpoint",
      "Checkpoint frozen after the campaign stopped",
      "live/state.json",
      "/final/frozen/checkpointSha256",
    ],
    [
      "release-bundle",
      "Release bundle the campaign bound",
      "live/state.json",
      "/final/release/releaseSha256",
    ],
    [
      "harness-version",
      "Pinned EleutherAI lm-evaluation-harness",
      "final/release/release.json",
      "/runtime/lm_eval/version",
    ],
    [
      "protocol",
      "Frozen evaluation protocol, identical for both policies",
      "final/release/release.json",
      "/protocol",
    ],
    [
      "development-strict",
      "Development harness, GSM8K strict match: starting policy and candidate",
      "final/development/comparison.json",
      "/metrics/strict",
    ],
    [
      "development-flexible",
      "Development harness, GSM8K flexible extract",
      "final/development/comparison.json",
      "/metrics/flexible",
    ],
    [
      "rehearsal-strict",
      "Release rehearsal on development data, strict match (not a final claim)",
      "final/rehearsal/comparison.json",
      "/metrics/strict",
    ],
    [
      "rehearsal-flexible",
      "Release rehearsal on development data, flexible extract",
      "final/rehearsal/comparison.json",
      "/metrics/flexible",
    ],
    [
      "cost",
      "Campaign spend in micro-USD (local cost unmeasured; charged at reservation bounds)",
      "live/state.json",
      "/campaign/spentMicroUsd",
    ],
  ];
  if (evaluated)
    statements.push(
      [
        "final-strict",
        "Sealed GSM8K test (1,319), strict match: starting policy and frozen checkpoint",
        "final/evaluation/comparison.json",
        "/metrics/strict",
      ],
      [
        "final-flexible",
        "Sealed GSM8K test, flexible extract",
        "final/evaluation/comparison.json",
        "/metrics/flexible",
      ],
      [
        "final-supported",
        "Predeclared final improvement gate",
        "final/evaluation/comparison.json",
        "/final_improvement_supported",
      ],
      [
        "final-record",
        "Strict-match counts in the final evaluator's actor record",
        "final/final-record.json",
        "/strict",
      ],
    );
  const recordingPath = join(root, "recording/manifest.json");
  const recording = (await exists(recordingPath)) ? await readJson(recordingPath) : null;
  if (recording)
    statements.push(
      [
        "recording-source",
        "Source recording duration in seconds",
        "recording/manifest.json",
        "/videos/0/durationSeconds",
      ],
      [
        "recording-100x",
        "100x export duration in seconds",
        "recording/manifest.json",
        "/videos/1/durationSeconds",
      ],
    );
  const claims: Claim[] = [];
  for (const [id, statement, artifact, at] of statements) {
    const path = join(root, artifact);
    const value = pointer(await readJson(path), at);
    claims.push({ id, statement, value, artifact, pointer: at, sha256: await fileSha(path) });
  }

  // Gate checks, each with what was observed.
  const harnessDirs = ["development", "release", "rehearsal", ...(evaluated ? ["evaluation"] : [])];
  const harnessBundles: Record<string, string> = {};
  let harnessFailure: string | null = null;
  for (const name of harnessDirs)
    try {
      harnessBundles[name] = await verifyHarness(python, join(root, "final", name));
    } catch (error) {
      harnessFailure = `${name}: ${String(error instanceof Error ? error.message : error).slice(0, 300)}`;
    }
  const releaseDoc = await readJson(join(root, "final/release/release.json"));
  const testUntouched: Record<string, boolean> = {
    release: releaseDoc.official_test_loaded === false,
    development:
      (await readJson(join(root, "final/development/result.json"))).official_test_loaded === false,
    rehearsal:
      (await readJson(join(root, "final/rehearsal/result.json"))).official_test_loaded === false,
  };
  const order = {
    campaignStoppedBeforeFreeze: campaign.status === "stopped",
    frozenIsPromoted: frozen.checkpointSha256 === best.checkpointSha256,
    releasePackagesFrozen: (releaseDoc.candidate as Document).sha256 === frozen.checkpointSha256,
    releaseBoundAfterFreeze: Number(bound.boundAtMs) >= Number(frozen.frozenAtMs),
    ...(evaluated
      ? {
          finalRecordedAfterRelease:
            Number((outcome.final as Document).recordedAtMs) > Number(bound.boundAtMs),
        }
      : {}),
  };
  const probes = outcome.probes as Record<string, string>;
  const probeFailures = Object.entries(probes)
    .filter(([name, value]) => !expectedProbes[name]?.(value))
    .map(([name]) => name);
  // The orchestrator profile's recorder (apps/web) is retired; the command center records
  // only its own profile. A manifest from that recorder is still checked if present.
  let recordingCheck: Document = {
    status: "not-recorded",
    reason: "No recording of this campaign; the orchestrator profile has no recorder",
  };
  if (recording) {
    const about = recording.campaign as Document | undefined;
    const ok =
      about?.campaignId === live.campaignId &&
      about?.finalStateSha256 === (await fileSha(join(root, "live/state.json"))) &&
      Date.parse(String(recording.startedAt)) <= Date.parse(String(live.startedAt)) &&
      Date.parse(String(recording.finishedAt)) >= Date.parse(String(live.updatedAt));
    recordingCheck = {
      status: ok ? "passed" : "failed",
      recordingStartedAt: recording.startedAt,
      campaignStartedAt: live.startedAt,
      campaignFinishedAt: live.updatedAt,
      recordingFinishedAt: recording.finishedAt,
    };
  }
  const eventRules: Document = {
    status: "unverified",
    reason: "No event rules record is included; eligibility is not claimed",
  };
  const claimCheck = await verifyClaims(root, claims);
  const checks = {
    sealedResultsDidNotInfluenceSelection: {
      status:
        Object.values(order).every(Boolean) && Object.values(testUntouched).every(Boolean)
          ? "passed"
          : "failed",
      ...order,
      officialTestNotLoadedBeforeFinal: testUntouched,
    },
    harnessEvidenceVerified: {
      status:
        !harnessFailure &&
        harnessBundles.release === bound.releaseSha256 &&
        harnessBundles.development === bound.developmentHarnessSha256
          ? "passed"
          : "failed",
      bundles: harnessBundles,
      ...(harnessFailure ? { failure: harnessFailure } : {}),
    },
    finalEvaluationRunsOnce: {
      status: probeFailures.length ? "failed" : "passed",
      probes,
      ...(probeFailures.length ? { unexpected: probeFailures } : {}),
    },
    sealedFinalEvaluation: {
      ...(outcome.sealed as Document),
      ...(evaluated ? { resultBundleSha256: (outcome.final as Document).resultBundleSha256 } : {}),
    },
    allClaimsMapToArtifacts: {
      status: claimCheck.failed.length ? "failed" : "passed",
      ...claimCheck,
    },
    recordingsMatchTheRun: recordingCheck,
    currentEventRules: eventRules,
  };

  // The bundle: harness evidence, the release (frozen adapter, also as an archive) and the
  // campaign evidence behind each claim.
  const bundle = join(root, "final/bundle");
  await mkdir(bundle, { recursive: false });
  const copy = async (path: string) => {
    if (!(await exists(join(root, path)))) return;
    await mkdir(dirname(join(bundle, path)), { recursive: true });
    await cp(join(root, path), join(bundle, path), { recursive: true });
  };
  for (const path of [
    ...harnessDirs.map((name) => `final/${name}`),
    "final/outcome.json",
    "final/protocol.json",
    "final/limits.json",
    "final/final-record.json",
    "research-report.json",
    "ledger.json",
    "live/state.json",
    baselineFile,
    governance,
    `experiments/${label}/run/manifest.json`,
    `experiments/${label}/run/result.json`,
    `experiments/${label}/run/training.json`,
    `experiments/${label}/evaluator/candidate.json`,
    "recording/manifest.json",
    ...((recording?.videos as Document[] | undefined) ?? []).map((v) => String(v.path)),
  ])
    await copy(path);
  await tar(join(bundle, "checkpoint.tar.gz"), join(bundle, "final/release"), "checkpoint");
  await writeFile(join(bundle, "claims.json"), JSON.stringify(claims, null, 2));
  await writeFile(join(bundle, "checks.json"), JSON.stringify(checks, null, 2));
  const counts = (file: string) =>
    readJson(join(root, file)).then((c) => {
      const m = c.metrics as Record<string, Document>;
      const cell = (k: string) =>
        `${String(m[k].baseline_correct)} → ${String(m[k].candidate_correct)} of ${String(m[k].num_examples)}`;
      return `${cell("strict")} | ${cell("flexible")}`;
    });
  const fixture = (live.model as Document | undefined)?.status === "fixture";
  await writeFile(
    join(bundle, "README.md"),
    `# Forge evidence bundle: ${String(live.campaignId)}

Frozen checkpoint \`${String(frozen.checkpointSha256)}\` (${label}), promoted by the campaign after
governance approval, packaged as release \`${String(bound.releaseSha256)}\` and evaluated with
EleutherAI lm-evaluation-harness ${String(((releaseDoc.runtime as Document).lm_eval as Document).version)} (GSM8K task, unmodified) under one frozen protocol.

| Evaluation | Strict match (start → frozen) | Flexible extract |
| --- | --- | --- |
| Development harness | ${await counts("final/development/comparison.json")} |
| Release rehearsal (development data) | ${await counts("final/rehearsal/comparison.json")} |
| Sealed GSM8K test | ${evaluated ? await counts("final/evaluation/comparison.json") : "not run | not run"} |
${evaluated ? "" : `\nSealed test not run: ${String((outcome.sealed as Document).reason)}.\n`}${fixture ? "\n**Fixture model:** these numbers exercise the mechanics and are not evidence about a useful model.\n" : ""}
- \`claims.json\`: every stated number with its artifact, JSON pointer and SHA-256.
- \`checks.json\`: the Phase 8 gate checks and what each observed. Event eligibility
  is unverified and is not claimed by this bundle.
- \`final/release/\`: the frozen release (adapter, evidence, protocol); \`checkpoint.tar.gz\`
  holds its adapter alone.
- \`SHA256SUMS\`: every file in this bundle.

Verify a copy, then reproduce the rehearsal from it (training-core's harness environment,
the same pinned runtime, and the base model the release names):

\`\`\`sh
bun services/orchestrator/scripts/bundle.ts --verify <bundle>
bun services/orchestrator/scripts/bundle.ts --reproduce <bundle> --output <new-directory>
\`\`\`
`,
  );
  const sums = (await walk(bundle))
    .filter((f) => !f.endsWith("SHA256SUMS"))
    .sort()
    .map(async (f) => `${await fileSha(f)}  ${relative(bundle, f).split(sep).join("/")}`);
  await writeFile(join(bundle, "SHA256SUMS"), (await Promise.all(sums)).join("\n") + "\n");
  return { bundle, checks, claims: claims.length };
}

/** Checks a bundle copy anywhere: every file against SHA256SUMS, every claim against its file. */
export async function verifyBundle(bundle: string): Promise<Document> {
  const lines = (await readFile(join(bundle, "SHA256SUMS"), "utf8")).trimEnd().split("\n");
  const changed: string[] = [];
  for (const line of lines) {
    const [digest, path] = [line.slice(0, 64), line.slice(66)];
    if (!(await exists(join(bundle, path))) || (await fileSha(join(bundle, path))) !== digest)
      changed.push(path);
  }
  const listed = new Set(lines.map((l) => l.slice(66)));
  const unlisted = (await walk(bundle))
    .map((f) => relative(bundle, f).split(sep).join("/"))
    .filter((p) => p !== "SHA256SUMS" && !listed.has(p));
  const claims = await verifyClaims(
    bundle,
    JSON.parse(await readFile(join(bundle, "claims.json"), "utf8")) as Claim[],
  );
  return { files: lines.length, changed, unlisted, claims };
}

/**
 * Clean-copy reproduction: an intact bundle's release is rehearsed again into a new directory
 * and every generation, score and the comparison must equal the recorded rehearsal.
 */
export async function reproduceBundle(
  bundle: string,
  python: string,
  output: string,
): Promise<Document> {
  const intact = await verifyBundle(bundle);
  if (
    (intact.changed as string[]).length ||
    (intact.unlisted as string[]).length ||
    (intact.claims as { failed: string[] }).failed.length
  )
    throw new Error("BUNDLE_CHANGED: verify the bundle before reproducing it");
  const claims = JSON.parse(await readFile(join(bundle, "claims.json"), "utf8")) as Claim[];
  const release = join(bundle, "final/release");
  await verifyHarness(
    python,
    release,
    String(claims.find((c) => c.id === "release-bundle")!.value),
  );
  await pythonJson(
    python,
    ["-m", "forge_training.final_cli", "rehearse", "--release", release, "--output", output],
    { timeoutMs: 3_600_000 },
  );
  const key = (s: Document) => JSON.stringify([s.id, s.prediction, s.strict, s.flexible]);
  const identical: Record<string, boolean> = {};
  for (const side of ["baseline", "candidate"]) {
    const before = (await readJson(join(bundle, "final/rehearsal", `${side}.json`))).samples;
    const after = (await readJson(join(output, `${side}.json`))).samples;
    identical[side] =
      JSON.stringify((before as Document[]).map(key)) ===
      JSON.stringify((after as Document[]).map(key));
  }
  const comparison =
    JSON.stringify(await readJson(join(bundle, "final/rehearsal/comparison.json"))) ===
    JSON.stringify(await readJson(join(output, "comparison.json")));
  return {
    reproduced: identical.baseline && identical.candidate && comparison,
    identicalGenerations: identical,
    identicalComparison: comparison,
  };
}
