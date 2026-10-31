import { readFile, rename, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import {
  clients,
  launch,
  localAuthority,
} from "../../../packages/terse-actors/scripts/local-runtime.js";
import type { Document } from "../../../packages/terse-actors/src/contracts.js";
import { digest } from "../../../packages/terse-actors/src/core.js";
import { bundleCampaign } from "../src/bundle.js";
import { refusal } from "../src/finalize.js";
import { exists, pythonJson, readJson } from "../src/worker.js";

/**
 * Phase 8, evaluator side: the sealed final evaluation of a finalized campaign, once, through
 * training-core's one-attempt evaluator (`forge-final`), recorded with the final evaluator's
 * key against the release the campaign bound. Usage:
 *   bun scripts/final-evaluate.ts <campaign-dir> --authority <dir> [--grant <file>]
 * The authority is the evaluator's (`forge-final init-authority`); research never holds it.
 * Locally the actor runtime restarts on the campaign's journal with fresh keys, so same-UID
 * local mode rehearses the boundary rather than isolating it.
 */
const args = process.argv.slice(2);
const option = (name: string) => {
  const index = args.indexOf(name);
  return index >= 0 ? resolve(args[index + 1]) : null;
};
const campaign = args[0] && !args[0].startsWith("--") ? resolve(args[0]) : null;
const authorityDir = option("--authority");
const python = process.env.FORGE_HARNESS_PYTHON ?? process.env.FORGE_PYTHON;
if (!campaign || !authorityDir || !python)
  throw new Error(
    "Usage: FORGE_HARNESS_PYTHON=<python> bun scripts/final-evaluate.ts <campaign-dir> --authority <dir> [--grant <file>]",
  );
const live = await readJson(join(campaign, "live/state.json"));
const campaignId = String(live.campaignId);
const release = join(campaign, "final/release");
const keys = localAuthority([campaignId]);
const runtime = await launch(keys, join(campaign, "terse"));
try {
  const actors = clients(keys, runtime);
  const state = (await actors.owner.read("FinalEvaluationActor", campaignId, "getSnapshot"))!;
  const frozen = state.frozen as Document | null,
    bound = state.release as Document | null;
  if (!frozen || !bound) throw new Error("NOT_RELEASED: finalize the campaign first");
  if (state.final) throw new Error("FINAL_ONCE: this campaign's final result is recorded");
  const harness = (command: string, extra: string[], timeoutMs = 600_000) =>
    pythonJson(python, ["-m", `forge_training.${command}`, ...extra], { timeoutMs });
  // The release on disk must be the one the campaign bound.
  await harness("harness_cli", [
    "verify",
    "--bundle",
    release,
    "--expected-hash",
    String(bound.releaseSha256),
  ]);
  const grant =
    option("--grant") ??
    String(
      (await harness("final_cli", ["authorize", "--authority", authorityDir, "--release", release]))
        .grant,
    );
  const limits = (await readJson(join(release, "release.json"))).limits as Document;
  const result = await harness(
    "final_cli",
    [
      "evaluate",
      "--authority",
      authorityDir,
      "--grant",
      grant,
      "--release",
      release,
      "--output",
      join(campaign, "final/evaluation"),
    ],
    (2 * Number(limits.max_seconds_per_model) + 600) * 1000,
  );
  const evaluation = join(campaign, "final/evaluation");
  const plan = await readJson(join(evaluation, "plan.json"));
  const comparison = await readJson(join(evaluation, "comparison.json"));
  const metrics = comparison.metrics as Record<string, Document>;
  const pair = (name: string) => ({
    baselineCorrect: Number(metrics[name].baseline_correct),
    candidateCorrect: Number(metrics[name].candidate_correct),
  });
  const protocol = plan.protocol as Document;
  const recorded = await actors.final_evaluator.command(
    "FinalEvaluationActor",
    campaignId,
    campaignId,
    "recordFinal",
    {
      releaseSha256: String(bound.releaseSha256),
      checkpointSha256: String(frozen.checkpointSha256),
      datasetSha256: String((plan.dataset as Document).parquet_sha256),
      harness: {
        name: "lm-evaluation-harness",
        version: String(((plan.runtime as Document).lm_eval as Document).version),
        task: String(protocol.task),
      },
      protocolSha256: digest(protocol),
      numExamples: Number(metrics.strict.num_examples),
      strict: pair("strict"),
      flexible: pair("flexible"),
      finalImprovementSupported: comparison.final_improvement_supported === true,
      resultBundleSha256: String(result.bundle_sha256),
    },
    "record-final",
  );
  // Beside the sealed evaluation bundle, never inside it.
  await writeFile(join(campaign, "final/final-record.json"), JSON.stringify(recorded, null, 2));
  const outcome = JSON.parse(await readFile(join(campaign, "final/outcome.json"), "utf8"));
  await writeFile(
    join(campaign, "final/outcome.json"),
    JSON.stringify(
      { ...outcome, status: "evaluated", sealed: { status: "recorded" }, final: recorded },
      null,
      2,
    ),
  );
  // The evidence bundle now includes the sealed result; the rehearsal bundle is kept beside it.
  if (await exists(join(campaign, "final/bundle")))
    await rename(join(campaign, "final/bundle"), join(campaign, "final/bundle-rehearsal"));
  const bundle = await bundleCampaign(campaign, python);
  console.log(JSON.stringify({ status: "recorded", final: recorded, bundle }, null, 2));
} catch (error) {
  console.error(JSON.stringify({ status: "refused", reason: refusal(error) }));
  process.exitCode = 1;
} finally {
  await runtime.stop();
}
