import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { Document } from "../../../packages/terse-actors/src/contracts.js";
import type { RelayDispatcher } from "./relay.js";
import { pythonJson } from "./worker.js";

/** Training-core harness settings used by release verification. */
export interface HarnessSettings {
  protocol: Document;
  limits: Document;
}

export interface FinalizeOptions {
  dispatcher: RelayDispatcher;
  output: string;
  /** Python with training-core's `harness` extra (EleutherAI lm-evaluation-harness). */
  harnessPython: string;
  harness: HarnessSettings;
  /** Whether the campaign's model is an offline fixture (final test access is refused). */
  fixture: boolean;
}

/** Error code of a refused command: an actor code, else the Python error type and message. */
export function refusal(error: unknown): string {
  const message = String(error instanceof Error ? error.message : error);
  const python = /"error_type": "(\w+)", "message": "([^"]*)"/.exec(message);
  if (python) return `${python[1]}: ${python[2].replace(/\/[^\s'"]+/g, "<path>")}`.slice(0, 160);
  return /\b([A-Z][A-Z_]{3,}):/.exec(message)?.[1] ?? message.slice(0, 160);
}

/**
 * Phase 8 on the research side, after the campaign stopped. The owner freezes the promoted
 * checkpoint in the actor; training-core's pinned harness then compares it with the starting
 * policy on development data, packages the release (`forge-harness freeze`, which re-checks
 * governance and held-out evidence), and rehearses that release on development data. The
 * owner binds the release hash in the actor. The sealed test is never read here: the
 * separately owned evaluator runs it once (`scripts/final-evaluate.ts`).
 */
export async function finalize(options: FinalizeOptions): Promise<Document> {
  const { dispatcher, output, harnessPython } = options;
  const { actors, campaignId } = dispatcher;
  const directory = join(output, "final");
  const outcome: Document = { status: "started" };
  const save = async () =>
    await writeFile(join(directory, "outcome.json"), JSON.stringify(outcome, null, 2));
  const harness = (command: string, args: string[], timeoutMs = 3_600_000) =>
    pythonJson(harnessPython, ["-m", `forge_training.${command}`, ...args], { timeoutMs });
  const stage = async <T>(name: string, run: () => Promise<T>): Promise<T | null> => {
    try {
      return await run();
    } catch (error) {
      Object.assign(outcome, { status: `failed-${name}`, reason: refusal(error) });
      await save();
      return null;
    }
  };
  await mkdir(directory, { recursive: true });

  const frozen = await stage("freeze", () =>
    actors.owner.command("FinalEvaluationActor", campaignId, campaignId, "freeze", {}, "freeze"),
  );
  if (!frozen) return outcome;
  outcome.frozen = frozen;
  const experiment = (await actors.owner.read(
    "ExperimentActor",
    String(frozen.experimentId),
    "getSnapshot",
  ))!;
  const artifact = await actors.owner.read("ArtifactRegistryActor", campaignId, "getArtifact", [
    (experiment.training as Document).checkpointArtifactId,
  ]);
  const checkpoint = fileURLToPath(String(artifact?.uri));
  const protocol = join(directory, "protocol.json"),
    limits = join(directory, "limits.json");
  await writeFile(protocol, JSON.stringify(options.harness.protocol, null, 2));
  await writeFile(limits, JSON.stringify(options.harness.limits, null, 2));
  const settings = ["--protocol", protocol, "--limits", limits];

  // Paired development comparison under the release protocol, then the release itself.
  const development = await stage("development-harness", () =>
    harness("harness_cli", [
      "development",
      "--config",
      join(dirname(checkpoint), "resolved-config.json"),
      "--checkpoint",
      checkpoint,
      "--expected-hash",
      String(frozen.checkpointSha256),
      ...settings,
      "--output",
      join(directory, "development"),
    ]),
  );
  if (!development) return outcome;
  outcome.development = { bundleSha256: development.bundle_sha256 };
  const release = await stage("release", () =>
    harness("harness_cli", [
      "freeze",
      "--campaign",
      output,
      "--harness-evidence",
      join(directory, "development"),
      ...settings,
      "--output",
      join(directory, "release"),
    ]),
  );
  if (!release) return outcome;
  const binding = {
    releaseSha256: String(release.bundle_sha256),
    checkpointSha256: String(frozen.checkpointSha256),
    developmentHarnessSha256: String(development.bundle_sha256),
  };
  const bound = await actors.owner.command(
    "FinalEvaluationActor",
    campaignId,
    campaignId,
    "bindRelease",
    binding,
    "bind-release",
  );
  outcome.release = bound;

  // Rehearsal: the release path end to end on development data; never a final claim.
  const rehearsal = await stage("rehearsal", () =>
    harness("final_cli", [
      "rehearse",
      "--release",
      join(directory, "release"),
      "--output",
      join(directory, "rehearsal"),
    ]),
  );
  if (!rehearsal) return outcome;
  outcome.rehearsal = { bundleSha256: rehearsal.bundle_sha256 };

  // Enforcement probes, recorded as evidence: each must be refused.
  const refused = async (attempt: () => Promise<unknown>) => {
    try {
      await attempt();
      return "ACCEPTED";
    } catch (error) {
      return refusal(error);
    }
  };
  const final = (role: "director" | "owner", method: string, payload: Document) =>
    actors[role].command("FinalEvaluationActor", campaignId, campaignId, method, payload);
  const zero = "0".repeat(64);
  const forgedResult = {
    releaseSha256: binding.releaseSha256,
    checkpointSha256: binding.checkpointSha256,
    datasetSha256: zero,
    harness: { name: "lm-evaluation-harness", version: "0.4.9.2", task: "gsm8k" },
    protocolSha256: zero,
    numExamples: 1,
    strict: { baselineCorrect: 0, candidateCorrect: 1 },
    flexible: { baselineCorrect: 0, candidateCorrect: 1 },
    finalImprovementSupported: true,
    resultBundleSha256: zero,
  };
  const probes: Document = {
    directorFreeze: await refused(() => final("director", "freeze", {})),
    secondFreeze: await refused(() => final("owner", "freeze", {})),
    releaseSwap: await refused(() =>
      final("owner", "bindRelease", { ...binding, releaseSha256: zero }),
    ),
    researchFinalRecord: await refused(() => final("director", "recordFinal", forgedResult)),
    rehearsalOverwrite: await refused(() =>
      harness("final_cli", [
        "rehearse",
        "--release",
        join(directory, "release"),
        "--output",
        join(directory, "rehearsal"),
      ]),
    ),
  };
  if (options.fixture) {
    // A fixture release must not be able to obtain final test access at all.
    const scratch = await mkdtemp(join(tmpdir(), "forge-final-probe-"));
    try {
      const authority = join(scratch, "authority");
      await harness("final_cli", ["init-authority", "--authority", authority]);
      probes.fixtureFinalAccess = await refused(() =>
        harness("final_cli", [
          "authorize",
          "--authority",
          authority,
          "--release",
          join(directory, "release"),
        ]),
      );
    } finally {
      await rm(scratch, { recursive: true, force: true });
    }
  }
  Object.assign(outcome, {
    status: "rehearsed",
    probes,
    sealed: options.fixture
      ? {
          status: "not-run",
          reason: "Fixture models may rehearse but cannot access final test data",
        }
      : {
          status: "pending",
          reason: "Awaiting the separately owned final evaluator (scripts/final-evaluate.ts)",
        },
  });
  await save();
  return JSON.parse(JSON.stringify(outcome)) as Document;
}
