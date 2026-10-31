import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import {
  clients,
  launch,
  localAuthority,
} from "../../../packages/terse-actors/scripts/local-runtime.js";
import { digest, document } from "../../../packages/terse-actors/src/core.js";
import type { Document } from "../../../packages/terse-actors/src/contracts.js";
import { POLICY_VERSION } from "../src/director.js";
import { chooseEvaluationBatch } from "../src/evaluations.js";
import { finalize, type HarnessSettings } from "../src/finalize.js";
import { GovernanceReviewer, type RegressionSuite } from "../src/governance.js";
import { reconcile } from "../src/reconcile.js";
import { RelayDispatcher, registerParticipants } from "../src/relay.js";
import { ResearchCampaign, type ResearchProfile } from "../src/research.js";
import { TrainingBridge, exists, pythonJson, readJson } from "../src/worker.js";
import { verifyWorldRollouts, type WorldGateway } from "../src/worlds.js";
import { openRelay, type RelayTarget } from "./relay-target.js";

const limits = { maxRevisions: 2, maxFailures: 2 };
const GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866";
const QWEN = {
  model_id: "Qwen/Qwen2.5-0.5B-Instruct",
  revision: "7ae557604adf67be50417f59c2c2f167def9a775",
};

/** Builds the campaign template and allowed training partition for a profile. */
/** Writes JSONL records and a training-core manifest whose hash covers the exact bytes. */
async function writeDataset(directory: string, name: string, lines: string[], manifest: Document) {
  await mkdir(directory, { recursive: true });
  const records = `${lines.join("\n")}\n`;
  await writeFile(join(directory, `${name}.jsonl`), records, { flag: "wx" });
  await writeFile(
    join(directory, `${name}.manifest.json`),
    JSON.stringify({
      ...manifest,
      content_sha256: createHash("sha256").update(records).digest("hex"),
      num_examples: lines.length,
    }),
    { flag: "wx" },
  );
  return {
    records: join(directory, `${name}.jsonl`),
    manifest: join(directory, `${name}.manifest.json`),
  };
}
const regressionPolicy = { maxDropBps: 0, maxDevRegressions: 0 };

/** Real-model device: Apple GPU by default; FORGE_DEVICE=cuda or cpu elsewhere. */
function modelDevice(): "mps" | "cuda" | "cpu" {
  const device = process.env.FORGE_DEVICE ?? "mps";
  if (device !== "mps" && device !== "cuda" && device !== "cpu")
    throw new Error(`FORGE_DEVICE must be mps, cuda or cpu, not ${device}`);
  return device;
}
/**
 * The dashboard's only view of actor state: snapshots read with public actor reads and
 * written as files, so the dashboard never holds actor credentials. Atomic replace.
 */
async function exportLive(
  output: string,
  actors: ReturnType<typeof clients>,
  campaignId: string,
  status: "running" | "finalizing" | "finished" | "failed",
  about: Document,
): Promise<void> {
  const read = (actor: "ResearchLedgerActor" | "ResearchCampaignActor" | "ResearchWorkflowActor") =>
    actors.owner.read(actor, campaignId, "getSnapshot");
  const [campaign, ledger, workflow] = await Promise.all([
    read("ResearchCampaignActor"),
    read("ResearchLedgerActor"),
    read("ResearchWorkflowActor"),
  ]);
  const final = await actors.owner.read("FinalEvaluationActor", campaignId, "getSnapshot");
  const experiments: Document = {};
  for (const experimentId of Object.keys((campaign?.proposals as Document | undefined) ?? {}))
    experiments[experimentId] = await actors.owner.read(
      "ExperimentActor",
      experimentId,
      "getSnapshot",
    );
  const directory = join(output, "live");
  await mkdir(directory, { recursive: true });
  const partial = join(directory, `state.json.${process.pid}.partial`);
  await writeFile(
    partial,
    JSON.stringify({
      schemaVersion: 1,
      ...about,
      status,
      updatedAt: new Date().toISOString(),
      campaign,
      ledger,
      workflow,
      experiments,
      final,
    }),
  );
  await rename(partial, join(directory, "state.json"));
}

/** Every GRPO setting is stated, so the reviewer pins all of them (unset defaults are not). */
function grpoSettings(overrides: Document): Document {
  return {
    num_generations: 4,
    max_completion_tokens: 256,
    max_prompt_tokens: 1024,
    temperature: 1.0,
    top_p: 1.0,
    beta: 0.0,
    epsilon: 0.2,
    loss_type: "dapo",
    scale_rewards: "group",
    reward_version: "exact-numeric-v1",
    action_parser: "final_answer_v1",
    max_rollouts: 4096,
    snapshot_every: 1,
    gradient_checkpointing: false,
    request_timeout_seconds: 120,
    ...overrides,
  };
}

/** The adapter a GRPO campaign continues from: a directory and its content hash. */
async function adapterAt(python: string, path: string): Promise<Document> {
  // The training core resolves symlinks in config paths; record the path it will see.
  const adapter = await pythonJson(python, [
    "-c",
    "import sys,json; from pathlib import Path; from forge_training.artifacts import directory_sha256; p = Path(sys.argv[1]).resolve(); print(json.dumps({'path': str(p), 'sha256': directory_sha256(p)}))",
    path,
  ]);
  return { path: String(adapter.path), sha256: String(adapter.sha256) };
}

async function provision(
  name: string,
  output: string,
  python: string,
  options: { startAdapter?: string; fromSft?: boolean } = {},
): Promise<{ profile: ResearchProfile; regression: RegressionSuite }> {
  if (name === "fixture") {
    const fixture = join(output, "fixture");
    await pythonJson(python, [
      "-c",
      "import sys,json; from pathlib import Path; from forge_training.fixtures import make_smoke_fixture; print(json.dumps({'config':str(make_smoke_fixture(Path(sys.argv[1])))}))",
      fixture,
    ]);
    const templatePath = join(fixture, "experiment.json");
    const template = await readJson(templatePath);
    await writeFile(
      templatePath,
      JSON.stringify({
        ...template,
        campaign_id: "research-fixture",
        experiment_id: "baseline",
        // Exercises the batched evaluation path and its batch-size-1 validation.
        evaluation: { ...(template.evaluation as Document), batch_size: 2 },
      }),
    );
    // Held-out synthetic problems for the regression suite; never in any training recipe.
    const regression = await writeDataset(
      join(fixture, "regression"),
      "regression",
      [7, 8].map((i) =>
        JSON.stringify({
          id: `regression-${i}`,
          question: `What is ${i} plus 1?`,
          answer: `#### ${i + 1}`,
        }),
      ),
      {
        schema_version: 1,
        dataset_id: "regression",
        purpose: "development",
        source: "forge-offline-fixture",
        source_revision: "generator-v1",
        source_split: "synthetic",
        license: "CC0-1.0",
      },
    );
    const profile: ResearchProfile = {
      name,
      campaignId: "research-fixture",
      templatePath,
      source: {
        records: join(fixture, "data/train.jsonl"),
        manifest: join(fixture, "data/train.manifest.json"),
      },
      base: { data: "full", learningRate: 0.0001, maxSteps: 3 },
      limits: { ...limits, minLearningRate: 0.00001, maxLearningRate: 0.01, maxSteps: 12 },
      costMicroUsd: 100000,
      maxExperiments: 4,
      timeouts: { runMs: 420000, evaluationMs: 300000 },
    };
    return { profile, regression: { ...regression, ...regressionPolicy } };
  }
  if (name === "fixture-final") {
    // Offline tiny model on a task SFT reliably learns (every answer is 9), so a campaign
    // promotes a checkpoint for the Phase 8 freeze, release and rehearsal. Mechanics only.
    const fixture = join(output, "fixture");
    await pythonJson(python, [
      "-c",
      "import sys,json; from pathlib import Path; from forge_training.fixtures import make_constant_fixture; print(json.dumps({'config':str(make_constant_fixture(Path(sys.argv[1])))}))",
      fixture,
    ]);
    const config = await readJson(join(fixture, "experiment.json"));
    const templatePath = join(fixture, "template.json");
    await writeFile(
      templatePath,
      JSON.stringify({
        ...config,
        campaign_id: "research-fixture-final",
        experiment_id: "baseline",
      }),
    );
    const profile: ResearchProfile = {
      name,
      campaignId: "research-fixture-final",
      templatePath,
      source: {
        records: join(fixture, "data/train.jsonl"),
        manifest: join(fixture, "data/train.manifest.json"),
      },
      base: { data: "full", learningRate: 0.01, maxSteps: 48 },
      limits: { ...limits, minLearningRate: 0.001, maxLearningRate: 0.01, maxSteps: 96 },
      costMicroUsd: 100000,
      maxExperiments: 3,
      timeouts: { runMs: 420000, evaluationMs: 300000 },
    };
    return {
      profile,
      regression: {
        records: join(fixture, "data/regression.jsonl"),
        manifest: join(fixture, "data/regression.manifest.json"),
        ...regressionPolicy,
      },
    };
  }
  if (name === "fixture-grpo") {
    // Offline tiny model warm-started on the answer format, so world rewards vary within
    // groups (services/training-core fixtures.py). Mechanics only, never benchmark evidence.
    const fixture = join(output, "fixture");
    await pythonJson(python, [
      "-c",
      "import sys,json; from pathlib import Path; from forge_training.fixtures import make_grpo_fixture; print(json.dumps({'config':str(make_grpo_fixture(Path(sys.argv[1])))}))",
      fixture,
    ]);
    const { grpo, ...config } = await readJson(join(fixture, "experiment.json"));
    // The template is the baseline-evaluation config: supervised method, allowed partition
    // with answers. Each GRPO experiment trains on that partition's answer-free projection.
    const templatePath = join(fixture, "template.json");
    await writeFile(
      templatePath,
      JSON.stringify({
        ...config,
        campaign_id: "research-fixture-grpo",
        experiment_id: "baseline",
        train: { records: "data/train.jsonl", manifest: "data/train.manifest.json" },
        training: { ...(config.training as Document), method: "sft_lora" },
      }),
    );
    let start: Document | null = null;
    if (options.fromSft) {
      // SFT then GRPO, in miniature: a short supervised run on this campaign's own base
      // model teaches the answer format; GRPO then continues that adapter.
      const sftPath = join(fixture, "parent-sft.json");
      await writeFile(
        sftPath,
        JSON.stringify({
          ...config,
          campaign_id: "research-fixture-grpo-parent",
          experiment_id: "parent-sft",
          train: { records: "data/train.jsonl", manifest: "data/train.manifest.json" },
          training: { ...(config.training as Document), method: "sft_lora", max_steps: 4 },
          output_dir: "parent-sft",
        }),
      );
      await pythonJson(python, ["-m", "forge_training", "run", "--config", sftPath], {
        timeoutMs: 300000,
      });
      start = await adapterAt(python, join(fixture, "parent-sft", "checkpoint"));
    }
    const profile: ResearchProfile = {
      name,
      campaignId: "research-fixture-grpo",
      templatePath,
      source: {
        records: join(fixture, "data/train.jsonl"),
        manifest: join(fixture, "data/train.manifest.json"),
      },
      base: { data: "full", learningRate: 0.01, maxSteps: 6 },
      limits: { ...limits, minLearningRate: 0.001, maxLearningRate: 0.01, maxSteps: 24 },
      costMicroUsd: 100000,
      maxExperiments: 4,
      timeouts: { runMs: 420000, evaluationMs: 300000 },
      method: {
        method: "grpo",
        grpo: grpoSettings({ ...(grpo as Document), ...(start ? { init_adapter: start } : {}) }),
      },
    };
    return {
      profile,
      regression: {
        records: join(fixture, "data/regression.jsonl"),
        manifest: join(fixture, "data/regression.manifest.json"),
        ...regressionPolicy,
      },
    };
  }
  if (name !== "qwen" && name !== "qwen-grpo")
    throw new Error(
      "Profile must be 'fixture', 'fixture-grpo', 'fixture-final', 'qwen' or 'qwen-grpo'",
    );
  const grpo = name === "qwen-grpo";
  // GRPO from the raw instruct model sees almost no reward (about 1 in 32 sampled answers
  // has a correct final line), so the profile continues an adapter that writes the format.
  const start = grpo && options.startAdapter ? await adapterAt(python, options.startAdapter) : null;
  if (grpo && !start)
    throw new Error("qwen-grpo needs --start-adapter <approved SFT checkpoint directory>");
  const raw = join(output, "data", "gsm8k-raw"),
    data = join(output, "data", "gsm8k");
  // Official GSM8K train file only, from the pinned revision; the test split is never read.
  await pythonJson(
    python,
    [
      "-m",
      "forge_training",
      "prepare-gsm8k",
      "--output",
      raw,
      "--revision",
      GSM8K_REVISION,
      "--train-size",
      "128",
      "--dev-size",
      "32",
    ],
    { timeoutMs: 600000 },
  );
  // Same seed-42 shuffle as Phase 4: the first 16 held-out rows remain the development
  // set; the next 16 become the regression suite. Both are disjoint from training.
  const read = async (file: string) =>
    (await readFile(join(raw, file), "utf8")).trimEnd().split("\n");
  const devManifest = JSON.parse(await readFile(join(raw, "development.manifest.json"), "utf8"));
  const heldOut = await read("development.jsonl");
  await writeDataset(
    data,
    "train",
    await read("train.jsonl"),
    JSON.parse(await readFile(join(raw, "train.manifest.json"), "utf8")),
  );
  await writeDataset(data, "development", heldOut.slice(0, 16), devManifest);
  const regression = await writeDataset(data, "regression", heldOut.slice(16), {
    ...devManifest,
    dataset_id: "gsm8k-regression",
  });
  const templatePath = join(output, "template.json");
  await writeFile(
    templatePath,
    JSON.stringify(
      {
        schema_version: 1,
        campaign_id: grpo ? "research-qwen-grpo" : "research-qwen-gsm8k",
        experiment_id: "baseline",
        model: { source: "hub", model_status: "instruct", ...QWEN },
        train: { records: "data/gsm8k/train.jsonl", manifest: "data/gsm8k/train.manifest.json" },
        development: {
          records: "data/gsm8k/development.jsonl",
          manifest: "data/gsm8k/development.manifest.json",
        },
        training: grpo
          ? {
              // The template itself is the supervised baseline-evaluation config.
              method: "sft_lora",
              // Continuing an adapter that already scores: a moderate step size protects it.
              learning_rate: 0.00005,
              max_steps: 16,
              // Low memory: one completion per loss micro-batch keeps the vocabulary-sized
              // logits small; eight micro-batches make an update of 2 prompts x 4 samples.
              batch_size: 1,
              gradient_accumulation_steps: 8,
              max_sequence_length: 1024,
              lora_rank: 8,
              lora_alpha: 16,
              target_modules: ["q_proj", "v_proj"],
              max_seconds: 3000,
              max_training_tokens: 1000000,
            }
          : {
              method: "sft_lora",
              learning_rate: 0.0002,
              max_steps: 32,
              batch_size: 1,
              gradient_accumulation_steps: 1,
              max_sequence_length: 1024,
              lora_rank: 8,
              lora_alpha: 16,
              target_modules: ["q_proj", "v_proj"],
              max_seconds: 1500,
              max_training_tokens: 200000,
            },
        evaluation: {
          scorer: "forge_numeric_v1",
          max_new_tokens: 256,
          max_prompt_tokens: 1024,
          max_seconds: 1500,
          batch_size: 8,
        },
        seed: 42,
        device: modelDevice(),
        precision: "float32",
        // The meta-device preflight counts tied embeddings twice (~630M for 494M loaded).
        max_model_parameters: 700000000,
        output_dir: "unused-template-output",
      },
      null,
      2,
    ),
  );
  const source = {
    records: join(data, "train.jsonl"),
    manifest: join(data, "train.manifest.json"),
  };
  const timeouts = { runMs: 3600000, evaluationMs: 1800000 };
  const profile: ResearchProfile = grpo
    ? {
        name,
        campaignId: "research-qwen-grpo",
        templatePath,
        source,
        base: { data: "full", learningRate: 0.00005, maxSteps: 16 },
        limits: { ...limits, minLearningRate: 0.00001, maxLearningRate: 0.001, maxSteps: 64 },
        costMicroUsd: 100000,
        maxExperiments: 3,
        timeouts,
        method: {
          method: "grpo",
          grpo: grpoSettings({
            max_rollouts: 1024,
            // Fewer adapter files; gradient checkpointing trades compute for memory.
            snapshot_every: 4,
            gradient_checkpointing: true,
            request_timeout_seconds: 300,
            init_adapter: start,
          }),
        },
      }
    : {
        name,
        campaignId: "research-qwen-gsm8k",
        templatePath,
        source,
        base: { data: "full", learningRate: 0.0002, maxSteps: 32 },
        limits: { ...limits, minLearningRate: 0.00002, maxLearningRate: 0.002, maxSteps: 128 },
        costMicroUsd: 100000,
        maxExperiments: 4,
        timeouts,
      };
  return { profile, regression: { ...regression, ...regressionPolicy } };
}

/**
 * Release harness settings: a tiny-model protocol for
 * offline fixtures, and training-core's zero-shot chat protocol and local limits for Qwen.
 */
function harnessSettings(fixture: boolean): HarnessSettings {
  if (fixture)
    return {
      protocol: {
        num_fewshot: 0,
        apply_chat_template: false,
        max_new_tokens: 8,
        max_prompt_tokens: 128,
        device: "cpu",
        precision: "float32",
        seed: 42,
      },
      limits: { max_seconds_per_model: 120, min_available_gib: 1.0, min_disk_free_gib: 4.0 },
    };
  const examples = join(import.meta.dir, "../../training-core/examples");
  return {
    protocol: JSON.parse(readFileSync(join(examples, "harness-qwen-development.json"), "utf8")),
    limits: JSON.parse(readFileSync(join(examples, "harness-local-limits.json"), "utf8")),
  };
}

/**
 * Real Relaycast, Terse and Python optimization under the deterministic director policy.
 * No human selects experiments after start; all roles run in this one process.
 */
export async function runResearch(
  output: string,
  python: string,
  name: string,
  options: {
    tamperFirstReviewedCheckpoint?: boolean;
    cacheDir?: string | null;
    startAdapter?: string;
    fromSft?: boolean;
    /** Phase 8: freeze, package and rehearse the release with this harness Python. */
    finalize?: { harnessPython: string };
  } = {},
): Promise<Document> {
  await mkdir(output, { recursive: true });
  const provisioned = await provision(name, output, python, options);
  // A GRPO campaign continuing an adapter is measured against that starting policy.
  const startAdapter = (provisioned.profile.method?.grpo?.init_adapter ?? null) as Document | null;
  const start = startAdapter
    ? { path: String(startAdapter.path), sha256: String(startAdapter.sha256) }
    : null;
  const { regression } = provisioned;
  const cacheDir = options.cacheDir ?? null;
  // Frozen untrained baseline, at the requested batch size only if it matches batch size 1.
  const batch = await chooseEvaluationBatch({
    python,
    templatePath: provisioned.profile.templatePath,
    directory: output,
    cacheDir,
    timeoutMs: provisioned.profile.timeouts.evaluationMs,
    start,
  });
  const profile: ResearchProfile = { ...provisioned.profile, templatePath: batch.templatePath };
  const baselinePath = batch.baseline.path,
    baselineSha256 = batch.baseline.sha256;
  const method = profile.method?.method ?? "sft_lora";
  const baseline = await readJson(baselinePath);
  // Resolved model identity: local fixture paths become absolute, exactly as workers see them.
  const template = (
    await pythonJson(python, ["-m", "forge_training", "validate", "--config", profile.templatePath])
  ).config as Document;
  const model = template.model as Document;
  const spec = {
    requirePreflight: true,
    objective: `Improve development accuracy of ${model.model_id}${start ? ` from adapter ${start.sha256}` : ""} on ${name.startsWith("qwen") ? "GSM8K-train-derived development math" : "an offline synthetic fixture"} within budget; a valid campaign may find no improvement`,
    model: { modelId: model.model_id, revision: model.revision },
    benchmark: name.startsWith("qwen")
      ? "gsm8k-train-development-forge_numeric_v1"
      : "offline-fixture",
    policyVersion: "v1",
    allowedStrategies: [method],
    maxBudgetMicroUsd: profile.costMicroUsd * profile.maxExperiments,
    maxConcurrent: 1,
    maxExperiments: profile.maxExperiments,
    maxAttempts: 2,
    maxLeaseMs: 60000,
    maxRunMs: profile.timeouts.runMs,
    minImprovementBps: 100,
    baseline: {
      reportSha256: baselineSha256,
      datasetSha256: (baseline.dataset as Document).content_sha256,
      protocolSha256: digest(baseline.protocol),
      numExamples: baseline.num_examples,
      numCorrect: baseline.num_correct,
    },
  };
  const authority = localAuthority([profile.campaignId]);
  const runtime = await launch(authority, join(output, "terse"));
  let relayTarget: RelayTarget | undefined;
  try {
    relayTarget = await openRelay(output);
    const relay = await relayTarget.createWorkspace(`forge-${name}-research`);
    const actors = clients(authority, runtime);
    const dispatcher = new RelayDispatcher(
      profile.campaignId,
      actors,
      await registerParticipants(relay),
    );
    const owner = (actor: "ResearchCampaignActor" | "GovernanceActor", payload: Document) =>
      actors.owner.command(actor, profile.campaignId, profile.campaignId, "initialize", payload);
    await owner("ResearchCampaignActor", spec);
    await owner("GovernanceActor", { objectiveSha256: digest(spec), policyVersion: "v1" });
    // Reward-driven jobs: this process is also the trusted task service (holds the reward
    // secret and the data role's key) that scores rollouts against the world actors.
    const gateways: WorldGateway[] = [];
    const bridge = new TrainingBridge(
      python,
      dispatcher,
      { ...profile.timeouts, leaseMs: 60000 },
      method === "grpo"
        ? { rewardSecret: authority.rewardSecret, onGateway: (gateway) => gateways.push(gateway) }
        : undefined,
    );
    const research = new ResearchCampaign({
      output,
      python,
      profile,
      dispatcher,
      bridge,
      baselinePath,
      baselineSha256,
      directorKeyId: authority.identities.director.keyId,
      governance: new GovernanceReviewer(
        python,
        dispatcher,
        {
          objectiveSha256: digest(spec),
          policyVersion: "v1",
          regression,
          limits: profile.limits,
          source: profile.source,
          template: await readJson(profile.templatePath),
          method: profile.method,
        },
        join(output, "governance"),
        profile.timeouts.evaluationMs,
        cacheDir,
      ),
      maxReviews: 2,
      faults: options,
    });
    // Verification probe after the campaign: every promotion path must refuse a veto.
    const vetoProbe = async () => {
      const results = [];
      for (const record of research.vetoed()) {
        const attempt = async (role: "director" | "coordinator") => {
          try {
            await actors[role].command(
              "ResearchCampaignActor",
              profile.campaignId,
              profile.campaignId,
              "promote",
              { experimentId: record.experimentId },
            );
            return "PROMOTED";
          } catch (error) {
            return String(error instanceof Error ? error.message : error).split(":")[0];
          }
        };
        results.push({
          label: record.label,
          director: await attempt("director"),
          coordinator: await attempt("coordinator"),
        });
      }
      return results;
    };
    const about = {
      startedAt: new Date().toISOString(),
      profile: name,
      method,
      campaignId: profile.campaignId,
      model: { modelId: model.model_id, revision: model.revision, status: model.model_status },
      startingAdapter: start,
      baseline: {
        reportSha256: baselineSha256,
        numExamples: baseline.num_examples,
        numCorrect: baseline.num_correct,
      },
    };
    // Snapshots every 2 seconds; a failed read (for example during a restart) skips a tick.
    let exporting = Promise.resolve();
    const exporter = setInterval(() => {
      exporting = exporting
        .then(() => exportLive(output, actors, profile.campaignId, "running", about))
        .catch(() => undefined);
    }, 2000);
    const started = Date.now();
    let outcome: Awaited<ReturnType<typeof research.run>>;
    try {
      outcome = await research.run();
    } catch (error) {
      clearInterval(exporter);
      await exporting;
      await exportLive(output, actors, profile.campaignId, "failed", about).catch(() => undefined);
      throw error;
    }
    clearInterval(exporter);
    await exporting;
    const read = (
      actor: "ResearchLedgerActor" | "ResearchCampaignActor" | "ResearchWorkflowActor",
    ) => actors.owner.read(actor, profile.campaignId, "getSnapshot");
    const [ledger, campaign, workflow] = await Promise.all([
      read("ResearchLedgerActor"),
      read("ResearchCampaignActor"),
      read("ResearchWorkflowActor"),
    ]);
    const experiments = [];
    for (const experimentId of Object.keys(campaign!.proposals as Document)) {
      const state = (await actors.owner.read("ExperimentActor", experimentId, "getSnapshot"))!;
      const launchRecord = (await actors.owner.read(
        "ResearchLedgerActor",
        profile.campaignId,
        "getLaunch",
        [experimentId],
      ))!.launch as Document;
      const evaluation = state.evaluation as Document | null;
      const job = bridge.get(String(launchRecord.label));
      const runDir = job && String(job.config.output_dir);
      let rewards: Document | null = null;
      if (job?.world && runDir && (await exists(join(runDir, "manifest.json")))) {
        // Re-read from the world journals, not from any message.
        const world = await verifyWorldRollouts({
          reader: actors.owner,
          campaignId: profile.campaignId,
          jobId: job.jobId,
          runDir,
        });
        const training = await readJson(join(runDir, "training.json"));
        rewards = document({
          ...world,
          updates: training.optimizer_steps,
          firstUpdateRewardMean: training.first_update_reward_mean,
          lastUpdateRewardMean: training.last_update_reward_mean,
          lineage: training.lineage,
        });
      }
      experiments.push({
        label: launchRecord.label,
        experimentId,
        strategy: (state.proposal as Document | null)?.strategy ?? null,
        rewards,
        operator: launchRecord.operator,
        parent: launchRecord.parent,
        recipe: launchRecord.recipe,
        status: state.status,
        attempts: state.attempts,
        preflight: (workflow!.reviews as Record<string, Document>)[experimentId] ?? null,
        candidateCorrect: evaluation?.candidateCorrect ?? null,
        numExamples: evaluation?.numExamples ?? null,
        reportSha256: evaluation?.reportSha256 ?? null,
      });
    }
    const entries = Object.values(ledger!.entries as Record<string, Document>).sort(
      (a, b) => Number(a.sequence) - Number(b.sequence),
    );
    let report = document({
      phase: options.finalize ? 8 : method === "grpo" ? 6 : 5,
      method,
      startingAdapter: start,
      profile: name,
      policyVersion: POLICY_VERSION,
      campaignId: profile.campaignId,
      relay: { kind: relayTarget.kind, url: relayTarget.url, workspaces: relayTarget.workspaces },
      elapsedSeconds: Math.round((Date.now() - started) / 1000),
      baseline: {
        reportSha256: baselineSha256,
        numExamples: baseline.num_examples,
        numCorrect: baseline.num_correct,
      },
      experiments,
      ledger: entries.map((e) => ({
        sequence: e.sequence,
        entryId: e.entryId,
        kind: e.kind,
        summary: e.summary,
        parents: e.parents,
        evidence: e.evidence,
        rejected: (e.body as Document).rejected ?? null,
        counterfactuals: (e.body as Document).counterfactuals ?? null,
        launch: (e.body as Document).launch ?? null,
      })),
      stop: outcome,
      developmentSelected: outcome.selected,
      campaignStatus: campaign!.status,
      spentMicroUsd: campaign!.spentMicroUsd,
      reservedMicroUsd: campaign!.reservedMicroUsd,
      bestCheckpointPromoted: campaign!.best,
      reconciliation: await reconcile(dispatcher),
      governance: outcome.governance,
      promoted: outcome.promoted,
      faultsInjected: outcome.faultsInjected,
      final: null,
      vetoEnforcement: await vetoProbe(),
      evaluation: {
        batchSizeRequested: batch.requested,
        batchSizeUsed: batch.used,
        batchMatchesBatchOne: batch.identicalPredictions,
        numExamples: batch.numExamples,
        frozenBaselineCached: batch.baseline.cached,
        reusedFrozenBaselineInRuns: true,
      },
      worldGateways: gateways.map((gateway) => ({ ...gateway.stats })),
      humanSelectedExperiments: false,
      modelImprovementClaimed: false,
    });
    await writeFile(join(output, "research-report.json"), JSON.stringify(report, null, 2));
    await writeFile(join(output, "ledger.json"), JSON.stringify(ledger, null, 2));
    // Phase 8, after the report the release freeze verifies: freeze, package and rehearse.
    if (options.finalize) {
      const finalizing = setInterval(() => {
        exporting = exporting
          .then(() => exportLive(output, actors, profile.campaignId, "finalizing", about))
          .catch(() => undefined);
      }, 2000);
      let final: Document;
      try {
        final = await finalize({
          dispatcher,
          output,
          harnessPython: options.finalize.harnessPython,
          harness: harnessSettings(model.model_status === "fixture"),
          fixture: model.model_status === "fixture",
        });
      } finally {
        clearInterval(finalizing);
        await exporting;
      }
      report = document({ ...report, final });
      await writeFile(join(output, "research-report.json"), JSON.stringify(report, null, 2));
    }
    await exportLive(output, actors, profile.campaignId, "finished", about);
    return report;
  } finally {
    await relayTarget?.stop();
    await runtime.stop();
  }
}

if (import.meta.main) {
  const python = process.env.FORGE_PYTHON;
  if (!python) throw new Error("Set FORGE_PYTHON to the absolute training-core virtualenv Python");
  const name = process.argv[2] ?? "fixture";
  const output = resolve(
    (process.argv[3]?.startsWith("--") ? undefined : process.argv[3]) ??
      join(import.meta.dir, "../artifacts", `research-${name}-${Date.now()}`),
  );
  const flag = process.argv.indexOf("--start-adapter");
  const report = await runResearch(output, python, name, {
    tamperFirstReviewedCheckpoint: process.argv.includes("--inject-tamper"),
    fromSft: process.argv.includes("--from-sft"),
    startAdapter: flag > 0 ? process.argv[flag + 1] : undefined,
    // The harness extra may live in its own environment (FORGE_HARNESS_PYTHON).
    finalize: process.argv.includes("--finalize")
      ? { harnessPython: process.env.FORGE_HARNESS_PYTHON ?? python }
      : undefined,
    cacheDir:
      process.env.FORGE_EVALUATION_CACHE ?? join(import.meta.dir, "../artifacts/cache/evaluations"),
  });
  console.log(JSON.stringify({ output, ...report }, null, 2));
}
