import { Actor, Atomic, Persisted } from "durable-actors";
import { z } from "zod";
import {
  campaignSchema,
  hash,
  id,
  money,
  proposalSchema,
  type CampaignState,
  type Document,
  type SignedRequest,
} from "./contracts.js";
import {
  actorRead,
  authorize,
  commit,
  digest,
  document,
  duplicate,
  events,
  fail,
  journal,
  requireScopedId,
  signGrant,
} from "./core.js";

const actorName = "ResearchCampaignActor";
export class ResearchCampaignActor extends Actor {
  @Persisted private state: CampaignState = {
    ...journal(),
    spec: null,
    objectiveSha256: null,
    status: "uninitialized",
    proposals: {},
    reservations: {},
    reservedMicroUsd: 0,
    spentMicroUsd: 0,
    best: null,
  };

  @Atomic
  async initialize(request: SignedRequest): Promise<Document> {
    const auth = authorize(request, actorName, this.id, "initialize", ["owner"], campaignSchema);
    if (request.campaignId !== this.id) fail("CAMPAIGN_SCOPE", "Campaign ID must match actor ID");
    const previous = duplicate(this.db, auth);
    if (previous) return previous;
    if (this.state.spec) fail("OBJECTIVE_IMMUTABLE", "Campaign is already initialized");
    const objectiveSha256 = digest(auth.payload);
    const result = { objectiveSha256, status: "active" };
    this.state = commit(
      this.db,
      { ...this.state, spec: auth.payload, objectiveSha256, status: "active" },
      auth,
      "CAMPAIGN_INITIALIZED",
      result,
      result,
    );
    return result;
  }

  @Atomic
  async propose(request: SignedRequest): Promise<Document> {
    const auth = authorize(request, actorName, this.id, "propose", ["director"], proposalSchema);
    if (request.campaignId !== this.id) fail("CAMPAIGN_SCOPE", "Wrong campaign");
    const previous = duplicate(this.db, auth);
    if (previous) return previous;
    const spec = this.state.spec;
    if (!spec || this.state.status !== "active")
      fail("CAMPAIGN_STOPPED", "Campaign does not accept new work");
    if (Object.keys(this.state.proposals).length >= spec.maxExperiments)
      fail("EXPERIMENT_LIMIT", "Experiment limit reached");
    const proposal = auth.payload;
    requireScopedId("exp", this.id, proposal.experimentId);
    if (this.state.proposals[proposal.experimentId])
      fail("EXPERIMENT_EXISTS", "Experiment IDs are immutable");
    if (
      !spec.allowedStrategies.includes(proposal.strategy) ||
      digest(spec.model) !== digest(proposal.model) ||
      proposal.developmentDatasetSha256 !== spec.baseline.datasetSha256
    ) {
      fail("OBJECTIVE_MISMATCH", "Proposal does not satisfy the immutable campaign specification");
    }
    const result = { experimentId: proposal.experimentId, configSha256: proposal.configSha256 };
    this.state = commit(
      this.db,
      { ...this.state, proposals: { ...this.state.proposals, [proposal.experimentId]: proposal } },
      auth,
      "EXPERIMENT_PROPOSED",
      result,
      result,
    );
    return result;
  }

  @Atomic
  async reserve(request: SignedRequest): Promise<Document> {
    const auth = authorize(
      request,
      actorName,
      this.id,
      "reserve",
      ["director", "coordinator"],
      z.strictObject({ experimentId: id }),
    );
    if (request.campaignId !== this.id) fail("CAMPAIGN_SCOPE", "Wrong campaign");
    const previous = duplicate(this.db, auth);
    if (previous) return previous;
    const spec = this.state.spec;
    if (!spec || this.state.status !== "active")
      fail("CAMPAIGN_STOPPED", "Campaign does not accept reservations");
    const proposal = this.state.proposals[auth.payload.experimentId];
    if (!proposal) fail("UNKNOWN_EXPERIMENT", "Propose an experiment before reserving compute");
    if (this.state.reservations[proposal.experimentId])
      fail("ALREADY_RESERVED", "Experiment already has a reservation");
    if (
      this.state.spentMicroUsd + this.state.reservedMicroUsd + proposal.maxCostMicroUsd >
      spec.maxBudgetMicroUsd
    ) {
      fail("BUDGET_EXCEEDED", "Reservation would exceed the campaign budget");
    }
    if (
      Object.values(this.state.reservations).filter((value) => !value.settled).length >=
      spec.maxConcurrent
    ) {
      fail("CONCURRENCY_LIMIT", "All experiment slots are reserved");
    }
    const grant = signGrant({
      ...(spec.requirePreflight ? { requirePreflight: true } : {}),
      version: 1,
      projectId: request.projectId,
      campaignId: this.id,
      experimentId: proposal.experimentId,
      objectiveSha256: this.state.objectiveSha256!,
      proposal,
      maxAttempts: spec.maxAttempts,
      maxLeaseMs: spec.maxLeaseMs,
      maxRunMs: spec.maxRunMs,
      issuedAtMs: Date.now(),
      grantId: request.requestId,
    });
    const result = document({ grant });
    this.state = commit(
      this.db,
      {
        ...this.state,
        reservedMicroUsd: this.state.reservedMicroUsd + proposal.maxCostMicroUsd,
        reservations: {
          ...this.state.reservations,
          [proposal.experimentId]: { grant, settled: false, spentMicroUsd: null },
        },
      },
      auth,
      "COMPUTE_RESERVED",
      result,
      { experimentId: proposal.experimentId, amountMicroUsd: proposal.maxCostMicroUsd },
    );
    return result;
  }

  @Atomic
  async settle(request: SignedRequest): Promise<Document> {
    const auth = authorize(
      request,
      actorName,
      this.id,
      "settle",
      ["coordinator"],
      z.strictObject({ experimentId: id, actualMicroUsd: money, costEvidenceSha256: hash }),
    );
    if (request.campaignId !== this.id) fail("CAMPAIGN_SCOPE", "Wrong campaign");
    const previous = duplicate(this.db, auth);
    if (previous) return previous;
    const reservation = this.state.reservations[auth.payload.experimentId];
    if (!reservation || reservation.settled)
      fail("RESERVATION_STATE", "Reservation is missing or already settled");
    if (auth.payload.actualMicroUsd > reservation.grant.claims.proposal.maxCostMicroUsd) {
      fail("BUDGET_EXCEEDED", "Actual cost exceeds the authorized reservation");
    }
    const experiment = z
      .object({ campaignId: id, grantId: id, status: z.enum(["completed", "evaluated", "failed"]) })
      .safeParse(await actorRead("ExperimentActor", auth.payload.experimentId, "getSnapshot"));
    if (
      !experiment.success ||
      experiment.data.campaignId !== this.id ||
      experiment.data.grantId !== reservation.grant.claims.grantId
    )
      fail("EXPERIMENT_NOT_TERMINAL", "A terminal experiment with the same grant is required");
    const result = {
      experimentId: auth.payload.experimentId,
      spentMicroUsd: auth.payload.actualMicroUsd,
    };
    this.state = commit(
      this.db,
      {
        ...this.state,
        reservedMicroUsd:
          this.state.reservedMicroUsd - reservation.grant.claims.proposal.maxCostMicroUsd,
        spentMicroUsd: this.state.spentMicroUsd + auth.payload.actualMicroUsd,
        reservations: {
          ...this.state.reservations,
          [auth.payload.experimentId]: {
            ...reservation,
            settled: true,
            spentMicroUsd: auth.payload.actualMicroUsd,
          },
        },
      },
      auth,
      "COMPUTE_SETTLED",
      result,
      { ...result, costEvidenceSha256: auth.payload.costEvidenceSha256 },
    );
    return result;
  }

  @Atomic
  async promote(request: SignedRequest): Promise<Document> {
    const auth = authorize(
      request,
      actorName,
      this.id,
      "promote",
      ["director", "coordinator"],
      z.strictObject({ experimentId: id }),
    );
    if (request.campaignId !== this.id) fail("CAMPAIGN_SCOPE", "Wrong campaign");
    const previous = duplicate(this.db, auth);
    if (previous) return previous;
    const spec = this.state.spec;
    if (!spec) fail("UNINITIALIZED", "Initialize the campaign first");
    const reservation = this.state.reservations[auth.payload.experimentId];
    if (!reservation || !reservation.settled)
      fail("UNSETTLED_EXPERIMENT", "Settle compute before promotion");
    const experiment = z
      .object({
        campaignId: id,
        grantId: id,
        status: z.literal("evaluated"),
        training: z.object({
          checkpointArtifactId: id,
          checkpointSha256: hash,
          configSha256: hash,
          weightsChanged: z.literal(true),
        }),
        evaluation: z.object({
          reportArtifactId: id,
          reportSha256: hash,
          checkpointSha256: hash,
          datasetSha256: hash,
          protocolSha256: hash,
          baselineReportSha256: hash,
          evaluationRole: z.literal("development"),
          numExamples: z.number().int().positive(),
          baselineCorrect: z.number().int().nonnegative(),
          candidateCorrect: z.number().int().nonnegative(),
        }),
      })
      .safeParse(await actorRead("ExperimentActor", auth.payload.experimentId, "getSnapshot"));
    if (!experiment.success)
      fail("INCOMPLETE_EVIDENCE", "A completed training and development evaluation are required");
    const exp = experiment.data;
    if (
      exp.campaignId !== this.id ||
      exp.grantId !== reservation.grant.claims.grantId ||
      exp.training.configSha256 !== reservation.grant.claims.proposal.configSha256 ||
      exp.training.checkpointSha256 !== exp.evaluation.checkpointSha256
    )
      fail("EVIDENCE_MISMATCH", "Experiment evidence does not match its reservation");
    const [decision, checkpoint, evaluation] = await Promise.all([
      actorRead("GovernanceActor", this.id, "getDecision", [exp.training.checkpointSha256]),
      actorRead("ArtifactRegistryActor", this.id, "getArtifact", [
        exp.training.checkpointArtifactId,
      ]),
      actorRead("ArtifactRegistryActor", this.id, "getArtifact", [exp.evaluation.reportArtifactId]),
    ]);
    const review = z
      .object({
        decision: z.literal("approved"),
        experimentId: id,
        checkpointSha256: hash,
        evaluationReportSha256: hash,
        objectiveSha256: hash,
        policyVersion: id,
      })
      .safeParse(decision);
    if (!review.success)
      fail("GOVERNANCE_VETO", "Checkpoint has no authoritative terminal approval");
    if (
      review.data.experimentId !== auth.payload.experimentId ||
      review.data.checkpointSha256 !== exp.training.checkpointSha256 ||
      review.data.evaluationReportSha256 !== exp.evaluation.reportSha256 ||
      review.data.objectiveSha256 !== this.state.objectiveSha256 ||
      review.data.policyVersion !== spec.policyVersion
    )
      fail("REVIEW_MISMATCH", "Review is bound to different evidence or policy");
    const artifact = z.object({ kind: z.string(), sha256: hash, experimentId: id });
    const c = artifact.safeParse(checkpoint),
      e = artifact.safeParse(evaluation);
    if (
      !c.success ||
      !e.success ||
      c.data.kind !== "checkpoint" ||
      e.data.kind !== "evaluation" ||
      c.data.experimentId !== auth.payload.experimentId ||
      e.data.experimentId !== auth.payload.experimentId ||
      c.data.sha256 !== exp.training.checkpointSha256 ||
      e.data.sha256 !== exp.evaluation.reportSha256
    ) {
      fail("ARTIFACT_MISMATCH", "Required immutable artifacts are missing or mismatched");
    }
    const result = exp.evaluation;
    if (
      result.baselineReportSha256 !== spec.baseline.reportSha256 ||
      result.datasetSha256 !== spec.baseline.datasetSha256 ||
      result.protocolSha256 !== spec.baseline.protocolSha256 ||
      result.numExamples !== spec.baseline.numExamples ||
      result.baselineCorrect !== spec.baseline.numCorrect ||
      result.candidateCorrect > result.numExamples
    ) {
      fail("BASELINE_MISMATCH", "Evaluation differs from the frozen baseline protocol");
    }
    if (
      (result.candidateCorrect - result.baselineCorrect) * 10_000 <
        spec.minImprovementBps * result.numExamples ||
      (this.state.best && result.candidateCorrect <= this.state.best.candidateCorrect)
    ) {
      fail(
        "NO_VALID_IMPROVEMENT",
        "Candidate does not improve the baseline and current approved best",
      );
    }
    const best = {
      experimentId: auth.payload.experimentId,
      checkpointSha256: exp.training.checkpointSha256,
      candidateCorrect: result.candidateCorrect,
      numExamples: result.numExamples,
    };
    this.state = commit(
      this.db,
      { ...this.state, best },
      auth,
      "CHECKPOINT_PROMOTED",
      document(best),
      document(best),
    );
    return document(best);
  }

  @Atomic
  async stop(request: SignedRequest): Promise<Document> {
    const auth = authorize(
      request,
      actorName,
      this.id,
      "stop",
      ["owner", "director"],
      z.strictObject({ reason: z.string().min(1).max(1000) }),
    );
    if (request.campaignId !== this.id || !this.state.spec)
      fail("CAMPAIGN_SCOPE", "Unknown campaign");
    const previous = duplicate(this.db, auth);
    if (previous) return previous;
    const result = { status: "stopped", reason: auth.payload.reason };
    this.state = commit(
      this.db,
      { ...this.state, status: "stopped" },
      auth,
      "CAMPAIGN_STOPPED",
      result,
      result,
    );
    return result;
  }

  async getSnapshot(): Promise<Document> {
    return document({
      campaignId: this.id,
      status: this.state.status,
      version: this.state.version,
      spec: this.state.spec,
      objectiveSha256: this.state.objectiveSha256,
      spentMicroUsd: this.state.spentMicroUsd,
      reservedMicroUsd: this.state.reservedMicroUsd,
      remainingMicroUsd: this.state.spec
        ? this.state.spec.maxBudgetMicroUsd - this.state.spentMicroUsd - this.state.reservedMicroUsd
        : 0,
      proposals: this.state.proposals,
      reservations: this.state.reservations,
      best: this.state.best,
    });
  }
  async getEvents(after: number, limit: number): Promise<Document> {
    return events(this.db, this.state.version, after, limit);
  }
}
