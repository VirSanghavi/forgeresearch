import { Actor, Atomic, Persisted } from "durable-actors";
import { z } from "zod";
import {
  evaluationSchema,
  grantSchema,
  integer,
  trainingResultSchema,
  type Document,
  type ExperimentState,
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
  verifyGrant,
  type Authorization,
} from "./core.js";

const actorName = "ExperimentActor";
function scope(state: ExperimentState, campaignId: string): void {
  if (!state.grant || state.grant.claims.campaignId !== campaignId)
    fail("CAMPAIGN_SCOPE", "Experiment is not initialized for this campaign");
}
function activeLease(state: ExperimentState, auth: Authorization, generation: number): void {
  scope(state, auth.request.campaignId);
  if (
    state.status !== "running" ||
    !state.lease ||
    state.lease.workerKeyId !== auth.keyId ||
    state.lease.generation !== generation ||
    state.lease.expiresAtMs <= Date.now()
  ) {
    fail("STALE_LEASE", "Worker lease is expired, superseded, or belongs to another identity");
  }
}
export class ExperimentActor extends Actor {
  @Persisted private state: ExperimentState = {
    ...journal(),
    grant: null,
    status: "uninitialized",
    generation: 0,
    attempts: 0,
    firstStartedAtMs: null,
    lease: null,
    training: null,
    evaluation: null,
    failure: null,
  };

  @Atomic
  async initialize(request: SignedRequest): Promise<Document> {
    const auth = authorize(
      request,
      actorName,
      this.id,
      "initialize",
      ["coordinator"],
      z.strictObject({ grant: grantSchema }),
    );
    const previous = duplicate(this.db, auth);
    if (previous) return previous;
    if (this.state.grant)
      fail("EXPERIMENT_EXISTS", "Experiment identity and configuration are immutable");
    const grant = verifyGrant(auth.payload.grant, this.id, request.campaignId);
    const result = { status: "ready", grantId: grant.claims.grantId };
    this.state = commit(
      this.db,
      { ...this.state, grant, status: "ready" },
      auth,
      "EXPERIMENT_INITIALIZED",
      result,
      result,
    );
    return result;
  }

  @Atomic
  async acquireLease(request: SignedRequest): Promise<Document> {
    const auth = authorize(
      request,
      actorName,
      this.id,
      "acquireLease",
      ["worker"],
      z.strictObject({ leaseMs: integer.min(50) }),
    );
    const previous = duplicate(this.db, auth);
    if (previous) return previous;
    scope(this.state, request.campaignId);
    if (this.state.grant!.claims.requirePreflight) {
      const approval = z
        .object({ approved: z.literal(true), configSha256: z.string() })
        .safeParse(
          await actorRead("ResearchWorkflowActor", request.campaignId, "getAuthorization", [
            this.id,
          ]),
        );
      if (
        !approval.success ||
        approval.data.configSha256 !== this.state.grant!.claims.proposal.configSha256
      )
        fail(
          "PREFLIGHT_REQUIRED",
          "A matching reviewed workflow assignment is required before acquiring compute",
        );
    }
    const claims = this.state.grant!.claims,
      now = Date.now();
    if (!["ready", "running", "retryable"].includes(this.state.status))
      fail("TERMINAL_EXPERIMENT", "Experiment cannot acquire more work");
    if (this.state.lease && this.state.lease.expiresAtMs > now)
      fail("LEASE_HELD", "An active worker already owns this experiment");
    if (this.state.attempts >= claims.maxAttempts)
      fail("RETRY_LIMIT", "Worker attempt limit reached");
    if (auth.payload.leaseMs > claims.maxLeaseMs)
      fail("LEASE_LIMIT", "Requested lease exceeds campaign policy");
    const firstStartedAtMs = this.state.firstStartedAtMs ?? now;
    if (now >= firstStartedAtMs + claims.maxRunMs)
      fail("RUNTIME_LIMIT", "Experiment runtime limit reached");
    const generation = this.state.generation + 1;
    const lease = {
      workerKeyId: auth.keyId,
      generation,
      expiresAtMs: Math.min(now + auth.payload.leaseMs, firstStartedAtMs + claims.maxRunMs),
    };
    const result = document(lease);
    this.state = commit(
      this.db,
      {
        ...this.state,
        lease,
        generation,
        firstStartedAtMs,
        attempts: this.state.attempts + 1,
        status: "running",
        failure: null,
      },
      auth,
      "WORKER_LEASE_ACQUIRED",
      result,
      result,
    );
    return result;
  }

  @Atomic
  async renewLease(request: SignedRequest): Promise<Document> {
    const auth = authorize(
      request,
      actorName,
      this.id,
      "renewLease",
      ["worker"],
      z.strictObject({ generation: integer.min(1), leaseMs: integer.min(50) }),
    );
    const previous = duplicate(this.db, auth);
    if (previous) return previous;
    activeLease(this.state, auth, auth.payload.generation);
    const claims = this.state.grant!.claims;
    if (auth.payload.leaseMs > claims.maxLeaseMs)
      fail("LEASE_LIMIT", "Requested lease exceeds campaign policy");
    const lease = {
      ...this.state.lease!,
      expiresAtMs: Math.min(
        Date.now() + auth.payload.leaseMs,
        this.state.firstStartedAtMs! + claims.maxRunMs,
      ),
    };
    const result = document(lease);
    this.state = commit(
      this.db,
      { ...this.state, lease },
      auth,
      "WORKER_LEASE_RENEWED",
      result,
      result,
    );
    return result;
  }

  @Atomic
  async complete(request: SignedRequest): Promise<Document> {
    const auth = authorize(
      request,
      actorName,
      this.id,
      "complete",
      ["worker"],
      z.strictObject({ generation: integer.min(1), result: trainingResultSchema }),
    );
    const previous = duplicate(this.db, auth);
    if (previous) return previous;
    activeLease(this.state, auth, auth.payload.generation);
    if (auth.payload.result.configSha256 !== this.state.grant!.claims.proposal.configSha256)
      fail("CONFIG_MISMATCH", "Training result belongs to another configuration");
    const result = { status: "completed", checkpointSha256: auth.payload.result.checkpointSha256 };
    this.state = commit(
      this.db,
      { ...this.state, status: "completed", lease: null, training: auth.payload.result },
      auth,
      "TRAINING_RESULT_REGISTERED",
      result,
      result,
    );
    return result;
  }

  @Atomic
  async recordEvaluation(request: SignedRequest): Promise<Document> {
    const auth = authorize(
      request,
      actorName,
      this.id,
      "recordEvaluation",
      ["evaluator"],
      evaluationSchema,
    );
    const previous = duplicate(this.db, auth);
    if (previous) return previous;
    scope(this.state, request.campaignId);
    if (this.state.status !== "completed" || !this.state.training)
      fail("TRAINING_NOT_COMPLETED", "Register the training checkpoint before its evaluation");
    if (
      auth.payload.checkpointSha256 !== this.state.training.checkpointSha256 ||
      auth.payload.datasetSha256 !== this.state.grant!.claims.proposal.developmentDatasetSha256
    )
      fail("EVALUATION_MISMATCH", "Evaluation belongs to another checkpoint or dataset");
    const result = { status: "evaluated", reportSha256: auth.payload.reportSha256 };
    this.state = commit(
      this.db,
      { ...this.state, status: "evaluated", evaluation: auth.payload },
      auth,
      "DEVELOPMENT_EVALUATED",
      result,
      result,
    );
    return result;
  }

  @Atomic
  async reportFailure(request: SignedRequest): Promise<Document> {
    const auth = authorize(
      request,
      actorName,
      this.id,
      "reportFailure",
      ["worker"],
      z.strictObject({ generation: integer.min(1), reason: z.string().min(1).max(1000) }),
    );
    const previous = duplicate(this.db, auth);
    if (previous) return previous;
    activeLease(this.state, auth, auth.payload.generation);
    const status =
      this.state.attempts < this.state.grant!.claims.maxAttempts ? "retryable" : "failed";
    const result = { status, reason: auth.payload.reason };
    this.state = commit(
      this.db,
      { ...this.state, status, lease: null, failure: auth.payload.reason },
      auth,
      "WORKER_FAILED",
      result,
      result,
    );
    return result;
  }

  @Atomic
  async abort(request: SignedRequest): Promise<Document> {
    const auth = authorize(
      request,
      actorName,
      this.id,
      "abort",
      ["coordinator"],
      z.strictObject({ grant: grantSchema, reason: z.string().min(1).max(1000) }),
    );
    const previous = duplicate(this.db, auth);
    if (previous) return previous;
    const grant = verifyGrant(auth.payload.grant, this.id, request.campaignId);
    if (this.state.grant && digest(this.state.grant) !== digest(grant))
      fail("GRANT_MISMATCH", "Experiment already has a different reservation");
    if (["completed", "evaluated", "failed"].includes(this.state.status))
      fail("TERMINAL_EXPERIMENT", "Terminal results are immutable");
    const result = { status: "failed", reason: auth.payload.reason };
    this.state = commit(
      this.db,
      {
        ...this.state,
        grant,
        status: "failed",
        generation: this.state.generation + 1,
        lease: null,
        failure: auth.payload.reason,
      },
      auth,
      "EXPERIMENT_ABORTED",
      result,
      result,
    );
    return result;
  }

  async getSnapshot(): Promise<Document> {
    return document({
      experimentId: this.id,
      campaignId: this.state.grant?.claims.campaignId ?? null,
      grantId: this.state.grant?.claims.grantId ?? null,
      maxLeaseMs: this.state.grant?.claims.maxLeaseMs ?? null,
      status: this.state.status,
      version: this.state.version,
      proposal: this.state.grant?.claims.proposal ?? null,
      attempts: this.state.attempts,
      generation: this.state.generation,
      lease: this.state.lease,
      training: this.state.training,
      evaluation: this.state.evaluation,
      failure: this.state.failure,
    });
  }
  async getEvents(after: number, limit: number): Promise<Document> {
    return events(this.db, this.state.version, after, limit);
  }
}
