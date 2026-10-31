import { appendFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import type { Document, Role } from "../../../packages/terse-actors/src/contracts.js";
import { digest, document, scopedId } from "../../../packages/terse-actors/src/core.js";
import type { WorkflowMessage } from "../../../packages/terse-actors/src/workflow.js";
import {
  POLICY_VERSION,
  compare,
  diagnose,
  initialDecision,
  measure,
  nextDecision,
  revise,
  type Candidate,
  type Decision,
  type History,
  type Measurement,
  type PolicyLimits,
  type Recipe,
} from "./director.js";
import type { GovernanceReviewer } from "./governance.js";
import { materialize, verifyRecipe, type MethodSettings, type SourcePartition } from "./recipes.js";
import { settleAtReservedBound } from "./reconcile.js";
import { participantRoles, type ParticipantRole, type RelayDispatcher } from "./relay.js";
import {
  exists,
  fileHash,
  prepareJob,
  pythonJson,
  readJson,
  trainWithRetries,
  type TrainingBridge,
} from "./worker.js";

export interface ResearchProfile {
  name: string;
  campaignId: string;
  /** Campaign config template: model, development data and the frozen protocol. */
  templatePath: string;
  source: SourcePartition;
  base: Recipe;
  limits: PolicyLimits;
  costMicroUsd: number;
  maxExperiments: number;
  timeouts: { runMs: number; evaluationMs: number };
  /** Training method of every recipe; defaults to supervised `sft_lora`. */
  method?: MethodSettings;
}
export interface ResearchContext {
  output: string;
  python: string;
  profile: ResearchProfile;
  dispatcher: RelayDispatcher;
  bridge: TrainingBridge;
  baselinePath: string;
  baselineSha256: string;
  directorKeyId: string;
  /** Phase 5: independent review of the development-selected candidates before promotion. */
  governance?: GovernanceReviewer;
  maxReviews?: number;
  /** Labeled failure injection: alter the first reviewed checkpoint after registration. */
  faults?: { tamperFirstReviewedCheckpoint?: boolean };
}
export interface GovernanceRecord {
  label: string;
  experimentId: string;
  decision: string;
  reasonCodes: string[];
  evidenceSha256: string;
  promotion: string | null;
}
export interface CampaignOutcome {
  rule: string;
  selected: string | null;
  promoted: string | null;
  governance: GovernanceRecord[];
  faultsInjected: string[];
}

/**
 * Message-driven research campaign. Every role acts only on Relay-delivered workflow
 * messages; the director decides only after a round's measurements are in actor state.
 * One process hosts all roles, as in Phase 3; the ledger, not this object, is the record.
 */
export class ResearchCampaign {
  private template: Document = {};
  private history!: History;
  private readonly recipes: Record<string, Recipe> = {};
  private readonly launched = new Set<string>();
  private round = 1;
  private lastEntry = "diagnosis";
  private closed: CampaignOutcome | null = null;
  private promotion: { decision: Decision & { kind: "stop" }; queue: Measurement[] } | null = null;
  private readonly governance: GovernanceRecord[] = [];
  private readonly faults: string[] = [];
  private promoted: string | null = null;
  constructor(readonly ctx: ResearchContext) {}

  private get campaignId() {
    return this.ctx.profile.campaignId;
  }
  private get method(): MethodSettings {
    return this.ctx.profile.method ?? { method: "sft_lora" };
  }
  private actor(role: Role) {
    return this.ctx.dispatcher.actors[role];
  }
  private experimentId(label: string) {
    return scopedId("exp", this.campaignId, label);
  }

  async run(): Promise<CampaignOutcome> {
    this.template = await readJson(this.ctx.profile.templatePath);
    const baseline = measure(
      "baseline",
      null,
      this.ctx.baselineSha256,
      await readJson(this.ctx.baselinePath),
    );
    await this.record({
      entryId: "diagnosis",
      kind: "diagnosis",
      summary: diagnose(baseline).summary,
      parents: [],
      evidence: [{ source: "baseline", reportSha256: this.ctx.baselineSha256 }],
      body: document({ ...diagnose(baseline), policyVersion: POLICY_VERSION }),
    });
    this.history = {
      baseline,
      measured: [],
      failed: [],
      round: [],
      revisions: 0,
      ...(await this.capacity()),
    };
    await this.apply(
      initialDecision(
        baseline,
        this.ctx.profile.base,
        this.ctx.profile.costMicroUsd,
        this.history,
        this.method.method,
      ),
    );
    await this.pump();
    if (!this.closed) {
      // A rejected preflight never reaches a worker; close explicitly instead of hanging.
      await this.finish({
        kind: "stop",
        rule: "STALLED",
        selected: [...this.history.measured].sort((a, b) => b.correct - a.correct)[0] ?? null,
        rejected: [],
        summary: "Launched work stopped making progress, for example after a preflight rejection.",
      });
    }
    return this.closed!;
  }

  private async capacity() {
    const campaign = (await this.actor("director").read(
      "ResearchCampaignActor",
      this.campaignId,
      "getSnapshot",
    ))!;
    return {
      remainingMicroUsd: Number(campaign.remainingMicroUsd),
      experimentsRemaining:
        this.ctx.profile.maxExperiments - Object.keys(campaign.proposals as Document).length,
    };
  }
  private evidence(): Document[] {
    return [
      { source: "baseline", reportSha256: this.ctx.baselineSha256 },
      ...this.history.measured.map((m) => ({
        source: "evaluation",
        experimentId: m.experimentId,
        reportSha256: m.reportSha256,
      })),
      ...this.history.failed.map((label) => ({
        source: "failure",
        experimentId: this.experimentId(label),
      })),
    ];
  }
  private async record(entry: Document): Promise<void> {
    await this.actor("director").command(
      "ResearchLedgerActor",
      this.campaignId,
      this.campaignId,
      "record",
      entry,
      `ledger-${entry.entryId}`,
    );
  }
  private summary(m: Measurement | null): Document | null {
    if (!m) return null;
    const { outcomes: _outcomes, ...rest } = m;
    return document(rest);
  }
  /** What the director would have chosen had the selected candidate measured otherwise. */
  private counterfactuals(best: Measurement): Document {
    const { baseline } = this.history;
    const as = (changes: Partial<Measurement>) =>
      revise(
        { ...best, ...changes },
        this.recipes[best.label],
        baseline,
        this.ctx.profile.limits,
        0,
      ).candidate?.operator ?? "stop";
    return {
      ifPredictionsUnchanged: as({ ...baseline, label: best.label }),
      ifImproved: as({ correct: baseline.correct + 1, predictionsSha256: digest("changed") }),
      ifRegressed: baseline.correct
        ? as({ correct: baseline.correct - 1, predictionsSha256: digest("changed") })
        : null,
    };
  }

  private async apply(decision: Decision): Promise<void> {
    const parents = [this.lastEntry];
    if (decision.kind === "stop") {
      if (this.ctx.governance && !this.promotion && decision.selected) {
        // Review in development rank order; a veto falls back to the next candidate.
        this.promotion = {
          decision,
          queue: [...this.history.measured].sort(compare).slice(0, this.ctx.maxReviews ?? 2),
        };
        return this.nextReview();
      }
      return this.finish(decision);
    }
    const hypotheses: string[] = [];
    for (const candidate of decision.launch) {
      this.recipes[candidate.label] = candidate.recipe;
      const entryId = `hypothesis-${candidate.label}`;
      await this.record({
        entryId,
        kind: "hypothesis",
        summary: candidate.hypothesis,
        parents,
        evidence: [],
        body: document(candidate),
      });
      hypotheses.push(entryId);
    }
    const entryId = `decision-${this.round}`;
    await this.record({
      entryId,
      kind: "decision",
      summary: decision.summary,
      parents: [...parents, ...hypotheses],
      evidence: this.evidence(),
      body: document({
        round: this.round,
        policyVersion: POLICY_VERSION,
        selected: this.summary(decision.selected),
        rejected: decision.rejected,
        counterfactuals: decision.selected ? this.counterfactuals(decision.selected) : null,
        launch: decision.launch.map((c) => ({
          experimentId: this.experimentId(c.label),
          label: c.label,
          recipe: c.recipe,
          operator: c.operator,
          parent: c.parent,
        })),
      }),
    });
    this.lastEntry = entryId;
    this.history.round = decision.launch;
    this.history.revisions += decision.launch.filter((c) => c.operator !== "explore").length;
    await this.advance();
  }

  vetoed(): GovernanceRecord[] {
    return this.governance.filter((g) => g.decision === "rejected");
  }
  private async finish(decision: Decision & { kind: "stop" }): Promise<void> {
    const outcome = this.promoted
      ? ` Promoted ${this.promoted} after independent approval.`
      : this.ctx.governance
        ? " No candidate was promoted."
        : "";
    await this.record({
      entryId: "stop",
      kind: "stop",
      summary: `${decision.summary}${outcome}`.slice(0, 2000),
      parents: [this.lastEntry],
      evidence: this.evidence(),
      body: document({
        rule: decision.rule,
        policyVersion: POLICY_VERSION,
        selected: this.summary(decision.selected),
        rejected: decision.rejected,
        promoted: this.promoted,
        governance: this.governance,
      }),
    });
    await this.actor("director").command(
      "ResearchCampaignActor",
      this.campaignId,
      this.campaignId,
      "stop",
      { reason: `${decision.rule}: ${decision.summary}`.slice(0, 1000) },
      "stop-campaign",
    );
    this.closed = {
      rule: decision.rule,
      selected: decision.selected?.label ?? null,
      promoted: this.promoted,
      governance: this.governance,
      faultsInjected: this.faults,
    };
  }

  /** Director: request independent review of the next-ranked candidate, or finish. */
  private async nextReview(): Promise<void> {
    const candidate = this.promotion!.queue.shift();
    if (!candidate) return this.finish(this.promotion!.decision);
    if (this.ctx.faults?.tamperFirstReviewedCheckpoint && !this.faults.length) {
      const job = this.ctx.bridge.get(candidate.label)!;
      const target = join(String(job.config.output_dir), "checkpoint", "adapter_config.json");
      await appendFile(target, "\n");
      this.faults.push(`INJECTED_CHECKPOINT_TAMPER:${candidate.label}`);
    }
    await this.ctx.dispatcher.publish(
      "director",
      `${candidate.label}-governance-request`,
      candidate.experimentId!,
      "REVIEW_REQUEST",
      { stage: "checkpoint", jobId: candidate.label, label: candidate.label },
    );
  }

  /** Director: promote an approved candidate; record a corrective decision after a veto. */
  private async onGovernance(message: WorkflowMessage, label: string): Promise<void> {
    if (this.governance.some((g) => g.label === label)) return;
    const body = message.body;
    const record: GovernanceRecord = {
      label,
      experimentId: message.experimentId,
      decision: String(body.decision),
      reasonCodes: (body.reasonCodes as string[]) ?? [],
      evidenceSha256: String(body.evidenceSha256),
      promotion: null,
    };
    this.governance.push(record);
    if (record.decision === "approved") {
      try {
        await this.actor("director").command(
          "ResearchCampaignActor",
          this.campaignId,
          this.campaignId,
          "promote",
          { experimentId: message.experimentId },
          `promote-${label}`,
        );
        record.promotion = "PROMOTED";
        this.promoted = label;
      } catch (error) {
        // Lower-ranked candidates cannot clear an improvement bar this one missed.
        record.promotion = String(error instanceof Error ? error.message : error).split(":")[0];
      }
      return this.finish(this.promotion!.decision);
    }
    const next = this.promotion!.queue[0]?.label ?? null;
    const entryId = `governance-veto-${label}`;
    await this.record({
      entryId,
      kind: "decision",
      summary: `Governance vetoed ${label} (${record.reasonCodes.join(", ")}). ${
        next
          ? `Corrective action: review the next-ranked candidate ${next}.`
          : "No candidate remains to review."
      }`,
      parents: [this.lastEntry],
      evidence: this.evidence(),
      body: document({
        policyVersion: POLICY_VERSION,
        action: next ? "review-next-ranked" : "stop-without-promotion",
        governanceEvidenceSha256: record.evidenceSha256,
        rejected: [
          {
            option: label,
            reason: `Vetoed (${record.reasonCodes.join(", ")}); evidence ${record.evidenceSha256}`,
          },
        ],
        launch: [],
      }),
    });
    this.lastEntry = entryId;
    return this.nextReview();
  }

  /** Launch the next candidate of the round, or decide once every candidate is terminal. */
  private async advance(): Promise<void> {
    const done = new Set([...this.history.measured.map((m) => m.label), ...this.history.failed]);
    const next = this.history.round.find((c) => !this.launched.has(c.label));
    if (next) return this.launch(next);
    if (this.history.round.some((c) => !done.has(c.label))) return;
    this.round++;
    Object.assign(this.history, await this.capacity());
    await this.apply(
      nextDecision(
        this.history,
        this.recipes,
        this.ctx.profile.limits,
        this.ctx.profile.costMicroUsd,
        this.round,
      ),
    );
  }
  private async launch(candidate: Candidate): Promise<void> {
    this.launched.add(candidate.label);
    await this.ctx.dispatcher.publish(
      "director",
      `${candidate.label}-assignment`,
      this.experimentId(candidate.label),
      "RESEARCH_ASSIGNMENT",
      document({
        jobId: candidate.label,
        label: candidate.label,
        recipe: candidate.recipe,
        decisionEntryId: this.lastEntry,
      }),
    );
  }

  /** Flush, deliver to every role, reconcile settlements; repeat until nothing moves. */
  private async pump(): Promise<void> {
    for (let i = 0; i < 500; i++) {
      await this.sweep();
      const sent = await this.ctx.dispatcher.flush();
      let handled = 0;
      for (const role of participantRoles) {
        const receipt = await this.ctx.dispatcher.receive(role, (m) => this.handle(role, m));
        handled += receipt.viaRelay + receipt.viaReconciliation;
      }
      if (!sent && !handled) return this.sweep();
    }
    throw new Error("Research pump did not quiesce");
  }
  /** Coordinator: settle terminal reservations so the next one can start. */
  private async sweep(): Promise<void> {
    const campaign = (await this.actor("coordinator").read(
      "ResearchCampaignActor",
      this.campaignId,
      "getSnapshot",
    ))!;
    for (const [experimentId, reservation] of Object.entries(
      campaign.reservations as Record<string, Document>,
    )) {
      if (reservation.settled) continue;
      const experiment = (await this.actor("coordinator").read(
        "ExperimentActor",
        experimentId,
        "getSnapshot",
      ))!;
      if (experiment.status === "evaluated" || experiment.status === "failed")
        await settleAtReservedBound(
          this.ctx.dispatcher,
          experimentId,
          join(this.ctx.output, "cost", `${experimentId}.json`),
        );
    }
  }

  private async handle(role: ParticipantRole, message: WorkflowMessage): Promise<void> {
    const body = message.body;
    const label = String(body.label ?? body.jobId ?? "");
    switch (`${role}:${message.kind}`) {
      case "data:RESEARCH_ASSIGNMENT":
        return this.prepare(message);
      case "director:EXPERIMENT_PROPOSAL":
        await this.actor("director").command(
          "ResearchCampaignActor",
          this.campaignId,
          this.campaignId,
          "propose",
          body.proposal as Document,
          `propose-${label}`,
        );
        return this.ctx.dispatcher.publish(
          "director",
          `${label}-review-request`,
          message.experimentId,
          "REVIEW_REQUEST",
          { jobId: label, label, configSha256: body.configSha256 },
        );
      case "reviewer:REVIEW_REQUEST":
        return body.stage === "checkpoint"
          ? this.checkpointReview(message)
          : this.preflight(message);
      case "coordinator:PREFLIGHT_REVIEWED":
        return this.authorize(message);
      case "worker:TRAINING_ASSIGNMENT":
        return trainWithRetries(this.ctx.bridge, this.ctx.dispatcher, message);
      case "evaluator:TRAINING_COMPLETED":
        return this.ctx.bridge.evaluate(message);
      case "reviewer:EVALUATION_COMPLETED":
        // Regression suites and governance decisions are Phase 5. Recording a terminal
        // decision now would make these checkpoints permanently ineligible for it.
        return this.ctx.dispatcher.publish(
          "reviewer",
          `${label}-checkpoint-review`,
          message.experimentId,
          "CHECKPOINT_REVIEWED",
          {
            jobId: label,
            decision: "deferred",
            reasonCodes: ["REGRESSION_SUITE_NOT_IMPLEMENTED"],
            reportSha256: body.reportSha256,
            governanceRecorded: false,
          },
        );
      case "director:CHECKPOINT_REVIEWED":
        if (body.stage === "checkpoint") return this.onGovernance(message, label);
        if (this.history.measured.some((m) => m.label === label)) return;
        this.history.measured.push(await this.measurement(message.experimentId, label));
        return this.advance();
      case "director:TRAINING_FAILED":
        if (this.history.failed.includes(label)) return;
        this.history.failed.push(label);
        return this.advance();
      default:
        throw new Error(`UNHANDLED_MESSAGE: ${role} cannot handle ${message.kind}`);
    }
  }

  /** Data role: materialize the recipe exactly as assigned; never invent parameters. */
  private async prepare(message: WorkflowMessage): Promise<void> {
    const { profile, python, bridge, output, baselinePath } = this.ctx;
    const label = String(message.body.jobId),
      recipe = message.body.recipe as unknown as Recipe;
    const directory = join(output, "experiments", label);
    const configPath = join(directory, "experiment.json");
    // A re-delivered assignment reuses its files; the reviewer verifies them either way.
    if (!(await exists(configPath)))
      await materialize({
        directory,
        template: this.template,
        templatePath: profile.templatePath,
        source: profile.source,
        campaignId: this.campaignId,
        experimentId: message.experimentId,
        recipe,
        method: this.method,
      });
    let job = bridge.get(label);
    const grpo = this.method.method === "grpo";
    if (!job) {
      job = await prepareJob(
        python,
        label,
        configPath,
        baselinePath,
        join(directory, "evaluator"),
        undefined,
        // Reward-driven jobs are scored by the task service against the allowed partition.
        grpo ? { source: profile.source } : undefined,
      );
      bridge.register(job);
    }
    const model = job.config.model as Document;
    const proposal = {
      experimentId: message.experimentId,
      strategy: this.method.method,
      model: { modelId: model.model_id, revision: model.revision },
      configSha256: job.configSha256,
      trainingDatasetSha256: job.trainingDatasetSha256,
      developmentDatasetSha256: ((await readJson(this.ctx.baselinePath)).dataset as Document)
        .content_sha256,
      hypothesis: grpo
        ? `${label}: GRPO with world rewards on ${recipe.data} tasks, lr ${recipe.learningRate}, ${recipe.maxSteps} updates`
        : `${label}: ${recipe.data} solutions, lr ${recipe.learningRate}, ${recipe.maxSteps} steps`,
      maxCostMicroUsd: profile.costMicroUsd,
    };
    await this.ctx.dispatcher.publish(
      "data",
      `${label}-proposal`,
      message.experimentId,
      "EXPERIMENT_PROPOSAL",
      document({ jobId: label, label, configSha256: job.configSha256, proposal }),
    );
  }

  /** Reviewer: deterministic preflight against the ledger launch and campaign records. */
  private async preflight(message: WorkflowMessage): Promise<void> {
    const reviewer = this.actor("reviewer");
    const label = String(message.body.jobId);
    const problems: string[] = [];
    const launch = await reviewer.read("ResearchLedgerActor", this.campaignId, "getLaunch", [
      message.experimentId,
    ]);
    const campaign = (await reviewer.read(
      "ResearchCampaignActor",
      this.campaignId,
      "getSnapshot",
    ))!;
    const proposal = (campaign.proposals as Record<string, Document>)[message.experimentId];
    const job = this.ctx.bridge.get(label);
    if (!launch || launch.authorKeyId !== this.ctx.directorKeyId)
      problems.push("no director ledger decision launched this experiment");
    if (!job || !proposal || proposal.configSha256 !== job.configSha256)
      problems.push("campaign proposal does not match the local job");
    if (!problems.length) {
      const validated = await pythonJson(this.ctx.python, [
        "-m",
        "forge_training",
        "validate",
        "--config",
        job!.configPath,
      ]);
      const config = validated.config as Document;
      if (digest(config) !== job!.configSha256)
        problems.push("configuration changed after proposal");
      const development = (validated.datasets as Document).development as Document;
      const spec = campaign.spec as Document;
      if (development.content_sha256 !== (spec.baseline as Document).datasetSha256)
        problems.push("development data differs from the frozen baseline");
      problems.push(
        ...(await verifyRecipe({
          recipe: (launch!.launch as Document).recipe as unknown as Recipe,
          limits: this.ctx.profile.limits,
          source: this.ctx.profile.source,
          template: this.template,
          validatedConfig: config,
          trainingRecordsPath: String((config.train as Document).records),
          method: this.method,
        })),
      );
    }
    await reviewer.command(
      "ResearchWorkflowActor",
      this.campaignId,
      this.campaignId,
      "review",
      {
        messageId: `${label}-preflight`,
        experimentId: message.experimentId,
        configSha256: String(message.body.configSha256),
        approved: !problems.length,
        reason: problems.length
          ? `Rejected: ${problems.join("; ")}`.slice(0, 2000)
          : "Ledger launch, recomputed recipe and data selection, frozen protocol, limits and validated configuration all match",
      },
      `review-${label}`,
    );
  }

  /** Reviewer: independent governance of a selected checkpoint; terminal decision. */
  private async checkpointReview(message: WorkflowMessage): Promise<void> {
    const label = String(message.body.jobId);
    const launch = (await this.actor("reviewer").read(
      "ResearchLedgerActor",
      this.campaignId,
      "getLaunch",
      [message.experimentId],
    ))!;
    const outcome = await this.ctx.governance!.review(
      this.ctx.bridge.get(label)!,
      message.experimentId,
      (launch.launch as Document).recipe as unknown as Recipe,
    );
    await this.ctx.dispatcher.publish(
      "reviewer",
      `${label}-governance`,
      message.experimentId,
      "CHECKPOINT_REVIEWED",
      {
        stage: "checkpoint",
        jobId: label,
        decision: outcome.decision,
        reasonCodes: outcome.reasonCodes,
        evidenceSha256: outcome.evidenceSha256,
      },
    );
  }

  /** Coordinator: reserve and assign only an approved preflight. */
  private async authorize(message: WorkflowMessage): Promise<void> {
    if (message.body.approved !== true) return;
    const coordinator = this.actor("coordinator");
    const launch = (await coordinator.read("ResearchLedgerActor", this.campaignId, "getLaunch", [
      message.experimentId,
    ]))!;
    const label = String((launch.launch as Document).label);
    const reservation = await coordinator.command(
      "ResearchCampaignActor",
      this.campaignId,
      this.campaignId,
      "reserve",
      { experimentId: message.experimentId },
      `reserve-${label}`,
    );
    await coordinator.command(
      "ExperimentActor",
      message.experimentId,
      this.campaignId,
      "initialize",
      reservation,
      `initialize-${label}`,
    );
    await coordinator.command(
      "ResearchWorkflowActor",
      this.campaignId,
      this.campaignId,
      "assign",
      { messageId: `${label}-train`, jobId: label, ...reservation },
      `assign-${label}`,
    );
  }

  /** Director: read the measurement from authoritative records, verifying file bytes. */
  private async measurement(experimentId: string, label: string): Promise<Measurement> {
    const director = this.actor("director");
    const experiment = (await director.read("ExperimentActor", experimentId, "getSnapshot"))!;
    const evaluation = experiment.evaluation as Document | null;
    if (experiment.status !== "evaluated" || !evaluation)
      throw new Error("MEASUREMENT_MISSING: experiment has no recorded evaluation");
    const artifact = await director.read("ArtifactRegistryActor", this.campaignId, "getArtifact", [
      evaluation.reportArtifactId,
    ]);
    const path = fileURLToPath(String(artifact?.uri));
    if (artifact?.sha256 !== evaluation.reportSha256 || (await fileHash(path)) !== artifact.sha256)
      throw new Error("EVIDENCE_CHANGED: evaluation report bytes differ from the record");
    return measure(label, experimentId, String(evaluation.reportSha256), await readJson(path));
  }
}
