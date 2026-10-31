import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { access, mkdir, readFile } from "node:fs/promises";
import { isAbsolute, join } from "node:path";
import { pathToFileURL } from "node:url";
import type { ForgeActorClient } from "../../../packages/terse-actors/src/client.js";
import type { Document } from "../../../packages/terse-actors/src/contracts.js";
import { digest, document } from "../../../packages/terse-actors/src/core.js";
import type { WorkflowMessage } from "../../../packages/terse-actors/src/workflow.js";
import type { SourcePartition } from "./recipes.js";
import { RelayDispatcher } from "./relay.js";
import {
  WorldGateway,
  loadWorldTasks,
  verifyWorldRollouts,
  type WorldEvidence,
  type WorldGatewayOptions,
} from "./worlds.js";

export async function readJson(path: string): Promise<Document> {
  return JSON.parse(await readFile(path, "utf8")) as Document;
}
export async function fileHash(path: string): Promise<string> {
  return createHash("sha256")
    .update(await readFile(path))
    .digest("hex");
}
export async function exists(path: string): Promise<boolean> {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}
export async function pythonJson(
  python: string,
  args: string[],
  options: { timeoutMs?: number; signal?: AbortSignal; env?: Record<string, string> } = {},
): Promise<Document> {
  if (!isAbsolute(python)) throw new Error("Python executable must be an absolute, trusted path");
  if (options.signal?.aborted) throw new Error("Python worker cancelled before launch");
  return await new Promise((resolve, reject) => {
    const child = spawn(python, args, {
      env: {
        PATH: process.env.PATH,
        HOME: process.env.HOME,
        TMPDIR: process.env.TMPDIR,
        HF_HUB_OFFLINE: "1",
        TRANSFORMERS_OFFLINE: "1",
        TOKENIZERS_PARALLELISM: "false",
        OMP_NUM_THREADS: "1",
        MKL_NUM_THREADS: "1",
        // Host-wide model-job lock (services/training-core locking.py) and Apple GPU
        // memory caps, so one job's footprint stays bounded on a shared machine.
        ...Object.fromEntries(
          [
            "FORGE_MODEL_LOCK",
            "FORGE_MODEL_LOCK_TIMEOUT",
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
            "PYTORCH_MPS_LOW_WATERMARK_RATIO",
          ]
            .filter((name) => process.env[name])
            .map((name) => [name, process.env[name]]),
        ),
        ...options.env,
      },
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "",
      stderr = "",
      aborted = false;
    let killTimer: ReturnType<typeof setTimeout> | undefined;
    const abort = () => {
      if (aborted) return;
      aborted = true;
      child.kill("SIGTERM");
      killTimer ??= setTimeout(() => child.kill("SIGKILL"), 7000);
    };
    const timer = setTimeout(abort, options.timeoutMs ?? 180000);
    options.signal?.addEventListener("abort", abort, { once: true });
    if (options.signal?.aborted) abort();
    // Decode across chunk boundaries, and wait for pipe closure before parsing JSON.
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    let outputBytes = 0;
    child.stdout.on("data", (value: string) => {
      if (aborted) return;
      outputBytes += Buffer.byteLength(value);
      if (outputBytes > 8_000_000) return abort();
      stdout += value;
    });
    child.stderr.on("data", (value) => {
      stderr = (stderr + String(value)).slice(-4000);
    });
    const cleanup = () => {
      clearTimeout(timer);
      if (killTimer) clearTimeout(killTimer);
      options.signal?.removeEventListener("abort", abort);
    };
    child.once("error", (error) => {
      cleanup();
      reject(error);
    });
    child.once("close", (code) => {
      cleanup();
      if (aborted || code !== 0) {
        reject(new Error(`Python worker ${aborted ? "cancelled" : `exited ${code}`}: ${stderr}`));
        return;
      }
      try {
        resolve(JSON.parse(stdout) as Document);
      } catch {
        reject(new Error("Python worker returned invalid JSON"));
      }
    });
  });
}

export interface TrainingJob {
  jobId: string;
  configPath: string;
  config: Document;
  configSha256: string;
  baselinePath: string;
  baselineSha256: string;
  trainingDatasetSha256: string;
  /** Evaluator-owned output; never inside the worker's run directory. */
  evaluatorDir: string;
  /** Reward-driven (GRPO) jobs: the allowed partition the task service scores against. */
  world?: { source: SourcePartition };
}
export async function prepareJob(
  python: string,
  jobId: string,
  configPath: string,
  baselinePath: string,
  evaluatorDir: string,
  timeoutMs = 180000,
  world?: { source: SourcePartition },
): Promise<TrainingJob> {
  const validated = await pythonJson(python, [
    "-m",
    "forge_training",
    "validate",
    "--config",
    configPath,
  ]);
  const config = validated.config as Document;
  // One frozen baseline serves every experiment of a campaign; protocol equality is
  // checked against each run's own baseline evaluation in `evidence`. It evaluates the
  // policy training starts from: the untrained model, or the adapter a run continues.
  const adapter = (config.grpo as Document | undefined)?.init_adapter as Document | undefined;
  if (!(await exists(baselinePath)))
    await pythonJson(
      python,
      [
        "-m",
        "forge_training",
        "evaluate",
        "--config",
        configPath,
        ...(adapter
          ? ["--checkpoint", String(adapter.path), "--expected-hash", String(adapter.sha256)]
          : []),
        "--output",
        baselinePath,
      ],
      { timeoutMs },
    );
  const baseline = await readJson(baselinePath);
  if (
    baseline.campaign_id !== config.campaign_id ||
    baseline.checkpoint_sha256 !== (adapter?.sha256 ?? null)
  )
    throw new Error("Baseline does not belong to this campaign's starting policy");
  return {
    jobId,
    configPath,
    config,
    configSha256: digest(config),
    baselinePath,
    baselineSha256: await fileHash(baselinePath),
    trainingDatasetSha256: String(
      ((validated.datasets as Document).train as Document).content_sha256,
    ),
    evaluatorDir,
    ...(world ? { world } : {}),
  };
}
const predictions = (report: Document) =>
  (report.samples as Document[]).map(({ id, prediction, correct }) => ({
    id,
    prediction,
    correct,
  }));

/** The trusted task service's settings for reward-driven jobs. */
export interface WorldSettings {
  /** The reward secret the world actors verify answer commitments with. */
  rewardSecret: string;
  lanes?: number;
  retryMs?: number;
  /** Labeled failure injection for the world gateway. */
  faults?: WorldGatewayOptions["faults"];
  /** Observes each attempt's gateway, for example to read its statistics. */
  onGateway?: (gateway: WorldGateway) => void;
}

/** Trusted local job registry: Relay messages can select IDs, never shell commands or paths. */
export class TrainingBridge {
  private readonly jobs = new Map<string, TrainingJob>();
  constructor(
    readonly python: string,
    readonly dispatcher: RelayDispatcher,
    readonly timeouts: { runMs: number; evaluationMs: number; leaseMs?: number } = {
      runMs: 420000,
      evaluationMs: 300000,
    },
    readonly worlds?: WorldSettings,
  ) {}
  register(job: TrainingJob): void {
    if (this.jobs.has(job.jobId)) throw new Error("Job ID is immutable");
    this.jobs.set(job.jobId, structuredClone(job));
  }
  get(jobId: string): TrainingJob | undefined {
    const job = this.jobs.get(jobId);
    return job && structuredClone(job);
  }
  private job(message: WorkflowMessage): TrainingJob {
    const job = this.jobs.get(String(message.body.jobId));
    if (
      !job ||
      job.config.campaign_id !== message.campaignId ||
      job.config.experiment_id !== message.experimentId ||
      job.configSha256 !== message.body.configSha256
    )
      throw new Error("JOB_MISMATCH: assignment is outside the local allowlist");
    return job;
  }
  private async snapshot(actor: ForgeActorClient, id: string): Promise<Document> {
    return (await actor.read("ExperimentActor", id, "getSnapshot"))!;
  }
  /**
   * Idempotent: a terminal experiment is reconciled from its stored bundle and announced,
   * never optimized again. `afterComplete` is failure injection for a crash between the
   * authoritative completion and its announcement.
   */
  async train(message: WorkflowMessage, afterComplete?: () => Promise<void>): Promise<void> {
    if (message.kind !== "TRAINING_ASSIGNMENT") throw new Error("Expected training assignment");
    const job = this.job(message),
      { campaignId, actors } = this.dispatcher;
    const state = await this.snapshot(actors.worker, message.experimentId);
    if (state.grantId !== message.body.grantId) throw new Error("GRANT_MISMATCH");
    if (state.status === "failed") throw new Error("EXPERIMENT_FAILED: no attempts remain");
    const proposal = state.proposal as Document;
    if (proposal.configSha256 !== job.configSha256) throw new Error("CONFIG_MISMATCH");
    const model = job.config.model as Document;
    if (
      digest(proposal.model) !== digest({ modelId: model.model_id, revision: model.revision }) ||
      proposal.trainingDatasetSha256 !== job.trainingDatasetSha256
    )
      throw new Error("PROPOSAL_MISMATCH");
    if (state.status !== "completed" && state.status !== "evaluated") {
      await this.optimize(job, message, proposal, Number(state.maxLeaseMs));
      if (afterComplete) await afterComplete();
    }
    const evidence = await this.evidence(job, actors.worker),
      runDir = String(job.config.output_dir);
    await actors.worker.command(
      "ArtifactRegistryActor",
      campaignId,
      campaignId,
      "register",
      {
        artifactId: `checkpoint-${job.jobId}`,
        experimentId: message.experimentId,
        kind: "checkpoint",
        sha256: evidence.result.checkpoint_sha256,
        uri: pathToFileURL(join(runDir, "checkpoint")).href,
        sizeBytes: evidence.checkpointBytes,
        provenanceSha256: evidence.manifestSha256,
      },
      `artifact-checkpoint-${job.jobId}`,
    );
    await this.dispatcher.publish(
      "worker",
      `trained-${job.jobId}`,
      message.experimentId,
      "TRAINING_COMPLETED",
      {
        jobId: job.jobId,
        configSha256: job.configSha256,
        manifestSha256: evidence.manifestSha256,
        checkpointSha256: evidence.result.checkpoint_sha256,
        ...(evidence.world ? { world: document(evidence.world) } : {}),
      },
    );
  }
  /** Reward-driven jobs score rollouts through a gateway scoped to this lease generation. */
  private async gateway(job: TrainingJob, experimentId: string, generation: number) {
    if (!job.world) return null;
    if (!this.worlds) throw new Error("WORLD_SETTINGS: reward-driven jobs need a task service");
    const train = job.config.train as Document,
      training = job.config.training as Document;
    const { tasks, taskVersion } = await loadWorldTasks({
      sourceRecords: job.world.source.records,
      sourceManifest: job.world.source.manifest,
      tasksRecords: String(train.records),
      tasksManifest: String(train.manifest),
    });
    const gateway = new WorldGateway({
      campaignId: this.dispatcher.campaignId,
      experimentId,
      jobId: job.jobId,
      attempt: generation,
      tasks,
      taskVersion,
      seed: Number(job.config.seed ?? 42),
      rewardSecret: this.worlds.rewardSecret,
      actors: () => ({ data: this.dispatcher.actors.data, worker: this.dispatcher.actors.worker }),
      rolloutsPerUpdate: Number(training.batch_size) * Number(training.gradient_accumulation_steps),
      maxUpdates: Number(training.max_steps),
      lanes: this.worlds.lanes,
      retryMs: this.worlds.retryMs,
      faults: this.worlds.faults,
    }).start();
    this.worlds.onGateway?.(gateway);
    return gateway;
  }
  private async optimize(
    job: TrainingJob,
    message: WorkflowMessage,
    proposal: Document,
    maxLeaseMs: number,
  ) {
    const { campaignId, actors } = this.dispatcher,
      runDir = String(job.config.output_dir);
    // Every renewal is a durable actor write; renew at a third of the campaign's lease.
    const leaseMs = Math.min(this.timeouts.leaseMs ?? 60000, maxLeaseMs || 10000);
    const lease = await actors.worker.command(
      "ExperimentActor",
      message.experimentId,
      campaignId,
      "acquireLease",
      { leaseMs },
    );
    const abort = new AbortController();
    let heartbeatFailure: unknown;
    let renewing = false;
    let inflight: Promise<void> = Promise.resolve();
    const timer = setInterval(
      () => {
        if (renewing) return;
        renewing = true;
        inflight = actors.worker
          .command("ExperimentActor", message.experimentId, campaignId, "renewLease", {
            generation: lease.generation,
            leaseMs,
          })
          .then(() => undefined)
          .catch((error) => {
            heartbeatFailure = error;
            abort.abort();
          })
          .finally(() => {
            renewing = false;
          });
      },
      Math.max(500, Math.floor(leaseMs / 3)),
    );
    let gateway: WorldGateway | null = null;
    try {
      // A completed bundle from a lost completion reply is reconciled, never rerun.
      if (!(await exists(join(runDir, "manifest.json")))) {
        // Partial output from a crashed attempt is preserved evidence, never resumed or reused.
        if (await exists(runDir))
          throw new Error("PARTIAL_RUN_DIRECTORY: an earlier attempt left incomplete output");
        const validated = await pythonJson(
          this.python,
          ["-m", "forge_training", "validate", "--config", job.configPath],
          { signal: abort.signal },
        );
        if (digest(validated.config) !== job.configSha256)
          throw new Error("CONFIG_CHANGED: refuse modified job configuration");
        const datasets = validated.datasets as Document;
        if (
          (datasets.train as Document).content_sha256 !== proposal.trainingDatasetSha256 ||
          (datasets.development as Document).content_sha256 !== proposal.developmentDatasetSha256
        )
          throw new Error("DATASET_MISMATCH");
        gateway = await this.gateway(job, message.experimentId, Number(lease.generation));
        // The run reuses the campaign's frozen untrained evaluation instead of repeating it.
        await pythonJson(
          this.python,
          [
            "-m",
            "forge_training",
            "run",
            "--config",
            job.configPath,
            "--frozen-baseline",
            job.baselinePath,
            "--frozen-baseline-sha256",
            job.baselineSha256,
          ],
          { signal: abort.signal, timeoutMs: this.timeouts.runMs, env: gateway?.environment() },
        );
        await gateway?.stop();
      }
      // Rewards must be the worlds' own: resend every recorded step with its request ID.
      const evidence = await this.evidence(job, actors.worker, actors.worker);
      clearInterval(timer);
      await inflight;
      if (heartbeatFailure) throw heartbeatFailure;
      await actors.worker.command(
        "ExperimentActor",
        message.experimentId,
        campaignId,
        "complete",
        {
          generation: lease.generation,
          result: {
            checkpointArtifactId: `checkpoint-${job.jobId}`,
            checkpointSha256: evidence.result.checkpoint_sha256,
            manifestSha256: evidence.manifestSha256,
            configSha256: job.configSha256,
            weightsChanged: true,
          },
        },
        `complete-${job.jobId}-${lease.generation}`,
      );
    } catch (error) {
      clearInterval(timer);
      await inflight;
      // Report only with the current fence. A lost fence is reconciled by the coordinator.
      await actors.worker
        .command("ExperimentActor", message.experimentId, campaignId, "reportFailure", {
          generation: lease.generation,
          reason: error instanceof Error ? error.message.slice(0, 900) : "Worker failed",
        })
        .catch(() => undefined);
      throw error;
    } finally {
      clearInterval(timer);
      await gateway?.stop();
    }
  }
  /**
   * The evaluator reloads the checkpoint in its own process and must reproduce the worker's
   * development predictions exactly before recording an evaluation. This is reproduction on
   * the same host and code, not Phase 5 independent infrastructure.
   */
  async evaluate(message: WorkflowMessage): Promise<void> {
    if (message.kind !== "TRAINING_COMPLETED") throw new Error("Expected completed training");
    const job = this.job(message),
      evidence = await this.evidence(job, this.dispatcher.actors.evaluator);
    const checkpointSha256 = evidence.result.checkpoint_sha256;
    if (
      message.body.manifestSha256 !== evidence.manifestSha256 ||
      message.body.checkpointSha256 !== checkpointSha256
    )
      throw new Error("EVIDENCE_CHANGED");
    const reportPath = join(job.evaluatorDir, "candidate.json");
    if (!(await exists(reportPath))) {
      await mkdir(job.evaluatorDir, { recursive: true });
      await pythonJson(
        this.python,
        [
          "-m",
          "forge_training",
          "evaluate",
          "--config",
          job.configPath,
          "--checkpoint",
          join(String(job.config.output_dir), "checkpoint"),
          "--expected-hash",
          String(checkpointSha256),
          "--output",
          reportPath,
        ],
        { timeoutMs: this.timeouts.evaluationMs },
      );
    }
    const reproduced = await readJson(reportPath);
    if (
      reproduced.checkpoint_sha256 !== checkpointSha256 ||
      digest(reproduced.protocol) !== digest(evidence.candidate.protocol) ||
      digest(reproduced.dataset) !== digest(evidence.frozen.dataset) ||
      reproduced.num_correct !== evidence.candidate.num_correct ||
      digest(predictions(reproduced)) !== digest(predictions(evidence.candidate))
    )
      throw new Error("EVALUATION_NOT_REPRODUCED: evaluator and worker predictions differ");
    const { actors, campaignId } = this.dispatcher,
      reportSha256 = await fileHash(reportPath);
    await actors.evaluator.command(
      "ArtifactRegistryActor",
      campaignId,
      campaignId,
      "register",
      {
        artifactId: `evaluation-${job.jobId}`,
        experimentId: message.experimentId,
        kind: "evaluation",
        sha256: reportSha256,
        uri: pathToFileURL(reportPath).href,
        sizeBytes: (await readFile(reportPath)).length,
        provenanceSha256: evidence.manifestSha256,
      },
      `artifact-evaluation-${job.jobId}`,
    );
    await actors.evaluator.command(
      "ExperimentActor",
      message.experimentId,
      campaignId,
      "recordEvaluation",
      {
        reportArtifactId: `evaluation-${job.jobId}`,
        reportSha256,
        baselineReportSha256: job.baselineSha256,
        checkpointSha256,
        datasetSha256: (reproduced.dataset as Document).content_sha256,
        protocolSha256: digest(reproduced.protocol),
        evaluationRole: "development",
        numExamples: reproduced.num_examples,
        baselineCorrect: evidence.frozen.num_correct,
        candidateCorrect: reproduced.num_correct,
      },
      `evaluated-${job.jobId}`,
    );
    await this.dispatcher.publish(
      "evaluator",
      `evaluation-${job.jobId}`,
      message.experimentId,
      "EVALUATION_COMPLETED",
      {
        jobId: job.jobId,
        configSha256: job.configSha256,
        reportSha256,
        workerReportSha256: evidence.candidateSha256,
        checkpointSha256,
        numExamples: reproduced.num_examples,
        baselineCorrect: evidence.frozen.num_correct,
        candidateCorrect: reproduced.num_correct,
        reproducedWorkerPredictions: true,
        ...(evidence.world ? { world: document(evidence.world) } : {}),
        costStatus: "not_measured_by_local_worker",
        promotionAuthorized: false,
      },
    );
  }
  private async evidence(job: TrainingJob, reader: ForgeActorClient, replayer?: ForgeActorClient) {
    const runDir = String(job.config.output_dir);
    const manifestSha256 = await fileHash(join(runDir, "manifest.json"));
    await pythonJson(this.python, ["-m", "forge_training", "verify", "--run-dir", runDir]);
    if (
      manifestSha256 !== (await fileHash(join(runDir, "manifest.json"))) ||
      job.baselineSha256 !== (await fileHash(job.baselinePath))
    )
      throw new Error("EVIDENCE_CHANGED");
    const [result, baseline, candidate, resolved, frozen, manifest] = await Promise.all([
      readJson(join(runDir, "result.json")),
      readJson(join(runDir, "baseline.json")),
      readJson(join(runDir, "candidate.json")),
      readJson(join(runDir, "resolved-config.json")),
      readJson(job.baselinePath),
      readJson(join(runDir, "manifest.json")),
    ]);
    if (
      result.campaign_id !== job.config.campaign_id ||
      result.experiment_id !== job.config.experiment_id ||
      result.weights_changed !== true ||
      result.status !== "completed"
    )
      throw new Error("INVALID_TRAINING_EVIDENCE");
    // Worker stages dataset and starting-adapter paths (the adapter's hash still binds it);
    // all other resolved configuration must match the approved job.
    const unstaged = (config: Document) => {
      const { train: _train, development: _dev, ...rest } = config;
      const grpo = rest.grpo as Document | null | undefined;
      const adapter = grpo?.init_adapter as Document | null | undefined;
      if (!adapter) return rest;
      const { path: _path, ...identity } = adapter;
      return { ...rest, grpo: { ...grpo, init_adapter: identity } };
    };
    if (digest(unstaged(resolved)) !== digest(unstaged(job.config)))
      throw new Error("CONFIG_MISMATCH");
    const trainingManifest = await readJson(join(runDir, "inputs/train.manifest.json"));
    if (trainingManifest.content_sha256 !== job.trainingDatasetSha256)
      throw new Error("TRAINING_DATASET_MISMATCH");
    if (
      digest(baseline.protocol) !== digest(frozen.protocol) ||
      digest(baseline.samples) !== digest(frozen.samples) ||
      digest(baseline.dataset) !== digest(frozen.dataset)
    )
      throw new Error("BASELINE_MISMATCH");
    let checkpointBytes = 0;
    for (const name of Object.keys(manifest.files as Document))
      if (name.startsWith("checkpoint/"))
        checkpointBytes += (await readFile(join(runDir, name))).length;
    // Reward-driven runs: every recorded reward must be a transition in the world journals.
    const world: WorldEvidence | null = job.world
      ? await verifyWorldRollouts({
          reader,
          replayer,
          campaignId: this.dispatcher.campaignId,
          jobId: job.jobId,
          runDir,
        })
      : null;
    if (Boolean(job.world) !== (result.training_method === "grpo"))
      throw new Error("INVALID_TRAINING_EVIDENCE: training method differs from the job");
    return {
      result,
      frozen,
      candidate,
      candidateSha256: await fileHash(join(runDir, "candidate.json")),
      manifestSha256,
      checkpointBytes,
      world,
    };
  }
}

/** Retry only after an attempt advances; policy exhaustion must release the reservation. */
export async function trainWithRetries(
  bridge: Pick<TrainingBridge, "train">,
  dispatcher: RelayDispatcher,
  message: WorkflowMessage,
): Promise<void> {
  const { actors, campaignId } = dispatcher;
  const snapshot = async () =>
    (await actors.worker.read("ExperimentActor", message.experimentId, "getSnapshot"))!;
  let previous = await snapshot();
  for (;;) {
    try {
      return await bridge.train(message);
    } catch (error) {
      let state = await snapshot();
      const lease = state.lease as Document | null;
      const leaseExpired = !lease || Number(lease.expiresAtMs) <= Date.now();
      if (
        ["ready", "retryable", "running"].includes(String(state.status)) &&
        leaseExpired &&
        /\b(?:RUNTIME_LIMIT|RETRY_LIMIT):/.test(String(error))
      ) {
        // A refused acquireLease does not mutate actor state. Close the exhausted job
        // explicitly, after confirming there is no live worker lease, so sweep can settle it.
        const campaign = (await actors.coordinator.read(
          "ResearchCampaignActor",
          campaignId,
          "getSnapshot",
        ))!;
        const reservation = (campaign.reservations as Document)[message.experimentId] as Document;
        await actors.coordinator.command(
          "ExperimentActor",
          message.experimentId,
          campaignId,
          "abort",
          { grant: reservation.grant, reason: String(error).slice(0, 1000) },
          `abort-exhausted-${message.body.jobId}`,
        );
        state = await snapshot();
      }
      if (state.status === "failed")
        return dispatcher.publish(
          "worker",
          `${message.body.jobId}-failed`,
          message.experimentId,
          "TRAINING_FAILED",
          { jobId: message.body.jobId, reason: String(state.failure ?? error) },
        );
      const attempted = Number(state.attempts) > Number(previous.attempts);
      const completed = ["completed", "evaluated"].includes(String(state.status));
      if (
        (attempted &&
          (state.status === "retryable" || (state.status === "running" && leaseExpired))) ||
        (completed && state.status !== previous.status)
      ) {
        previous = state;
        continue;
      }
      // Configuration/transport failures before a new attempt cannot consume retries.
      // Leave the message pending for reconciliation instead of spinning in place.
      throw error;
    }
  }
}
