import { Actor, Atomic, Persisted } from "durable-actors";
import { z } from "zod";
import {
  grantSchema,
  hash,
  id,
  type Document,
  type Journal,
  type Role,
  type SignedRequest,
} from "./contracts.js";
import {
  authorize,
  commit,
  document,
  duplicate,
  events,
  fail,
  journal,
  requireScopedId,
  verifyGrant,
} from "./core.js";

export const routes = {
  RESEARCH_ASSIGNMENT: ["director", "data"],
  EXPERIMENT_PROPOSAL: ["data", "director"],
  REVIEW_REQUEST: ["director", "reviewer"],
  TRAINING_COMPLETED: ["worker", "evaluator"],
  TRAINING_FAILED: ["worker", "director"],
  EVALUATION_COMPLETED: ["evaluator", "reviewer"],
  CHECKPOINT_REVIEWED: ["reviewer", "director"],
  CORRECTION: ["director", "data"],
} as const;
export type MessageKind = keyof typeof routes | "PREFLIGHT_REVIEWED" | "TRAINING_ASSIGNMENT";
export interface WorkflowMessage {
  schemaVersion: 1;
  campaignId: string;
  messageId: string;
  experimentId: string;
  kind: MessageKind;
  senderRole: Role;
  senderKeyId: string;
  recipientRole: Role;
  body: Document;
}
/** `sequence` is the workflow version that queued the message: dispatch in causal order. */
export interface Delivery {
  sequence: number;
  message: WorkflowMessage;
  relayMessageId: string | null;
  acknowledged: boolean;
}
interface WorkflowState extends Journal {
  deliveries: Record<string, Delivery>;
  reviews: Record<string, { configSha256: string; approved: boolean; reason: string }>;
  assignments: Record<string, string>;
}
const actorName = "ResearchWorkflowActor";
const publishSchema = z.strictObject({
  messageId: id,
  experimentId: id,
  kind: z.enum([
    "RESEARCH_ASSIGNMENT",
    "EXPERIMENT_PROPOSAL",
    "REVIEW_REQUEST",
    "TRAINING_COMPLETED",
    "TRAINING_FAILED",
    "EVALUATION_COMPLETED",
    "CHECKPOINT_REVIEWED",
    "CORRECTION",
  ]),
  body: z.record(z.string(), z.json()),
});
function append(state: WorkflowState, message: WorkflowMessage): WorkflowState {
  if (state.deliveries[message.messageId]) fail("MESSAGE_IMMUTABLE", "Message ID already exists");
  if (Object.keys(state.deliveries).length >= 2000)
    fail("OUTBOX_LIMIT", "Create a new workflow before reaching capacity");
  return {
    ...state,
    deliveries: {
      ...state.deliveries,
      [message.messageId]: {
        sequence: state.version + 1,
        message,
        relayMessageId: null,
        acknowledged: false,
      },
    },
  };
}
function scope(request: SignedRequest, actorId: string) {
  if (request.campaignId !== actorId)
    fail("CAMPAIGN_SCOPE", "Workflow belongs to another campaign");
}
export class ResearchWorkflowActor extends Actor {
  @Persisted private state: WorkflowState = {
    ...journal(),
    deliveries: {},
    reviews: {},
    assignments: {},
  };

  @Atomic
  async publish(request: SignedRequest): Promise<Document> {
    const auth = authorize(
      request,
      actorName,
      this.id,
      "publish",
      ["director", "data", "worker", "evaluator", "reviewer"],
      publishSchema,
    );
    scope(request, this.id);
    const previous = duplicate(this.db, auth);
    if (previous) return previous;
    const [senderRole, recipientRole] = routes[auth.payload.kind];
    if (auth.role !== senderRole)
      fail("MESSAGE_ROLE", "This role cannot publish this message kind");
    requireScopedId("exp", this.id, auth.payload.experimentId);
    const message: WorkflowMessage = {
      ...auth.payload,
      body: document(auth.payload.body),
      schemaVersion: 1,
      campaignId: this.id,
      senderRole,
      recipientRole,
      senderKeyId: auth.keyId,
    };
    const result = document(message);
    this.state = commit(this.db, append(this.state, message), auth, "MESSAGE_QUEUED", result, {
      messageId: message.messageId,
      kind: message.kind,
    });
    return result;
  }

  @Atomic
  async review(request: SignedRequest): Promise<Document> {
    const auth = authorize(
      request,
      actorName,
      this.id,
      "review",
      ["reviewer"],
      z.strictObject({
        messageId: id,
        experimentId: id,
        configSha256: hash,
        approved: z.boolean(),
        reason: z.string().min(1).max(2000),
      }),
    );
    scope(request, this.id);
    const previous = duplicate(this.db, auth);
    if (previous) return previous;
    requireScopedId("exp", this.id, auth.payload.experimentId);
    if (this.state.reviews[auth.payload.experimentId])
      fail("REVIEW_FINAL", "Preflight review cannot be overwritten");
    const { messageId, experimentId, ...review } = auth.payload;
    const message: WorkflowMessage = {
      schemaVersion: 1,
      campaignId: this.id,
      messageId,
      experimentId,
      kind: "PREFLIGHT_REVIEWED",
      senderRole: "reviewer",
      senderKeyId: auth.keyId,
      recipientRole: "coordinator",
      body: review,
    };
    this.state = commit(
      this.db,
      append(
        { ...this.state, reviews: { ...this.state.reviews, [experimentId]: review } },
        message,
      ),
      auth,
      "PREFLIGHT_REVIEWED",
      document(message),
      { messageId, experimentId, approved: review.approved },
    );
    return document(message);
  }

  @Atomic
  async assign(request: SignedRequest): Promise<Document> {
    const auth = authorize(
      request,
      actorName,
      this.id,
      "assign",
      ["coordinator"],
      z.strictObject({ messageId: id, jobId: id, grant: grantSchema }),
    );
    scope(request, this.id);
    const previous = duplicate(this.db, auth);
    if (previous) return previous;
    const grant = verifyGrant(auth.payload.grant, auth.payload.grant.claims.experimentId, this.id);
    const { experimentId, proposal } = grant.claims;
    const review = this.state.reviews[experimentId];
    if (!review?.approved || review.configSha256 !== proposal.configSha256)
      fail("PREFLIGHT_REQUIRED", "Matching independent preflight approval is required");
    if (this.state.assignments[experimentId])
      fail("ALREADY_ASSIGNED", "Experiment already has an assignment");
    const message: WorkflowMessage = {
      schemaVersion: 1,
      campaignId: this.id,
      messageId: auth.payload.messageId,
      experimentId,
      kind: "TRAINING_ASSIGNMENT",
      senderRole: "coordinator",
      senderKeyId: auth.keyId,
      recipientRole: "worker",
      body: {
        jobId: auth.payload.jobId,
        configSha256: proposal.configSha256,
        grantId: grant.claims.grantId,
      },
    };
    this.state = commit(
      this.db,
      append(
        {
          ...this.state,
          assignments: { ...this.state.assignments, [experimentId]: message.messageId },
        },
        message,
      ),
      auth,
      "TRAINING_ASSIGNED",
      document(message),
      { messageId: message.messageId, experimentId },
    );
    return document(message);
  }

  @Atomic
  async markSent(request: SignedRequest): Promise<Document> {
    const auth = authorize(
      request,
      actorName,
      this.id,
      "markSent",
      ["coordinator"],
      z.strictObject({ messageId: id, relayMessageId: id }),
    );
    scope(request, this.id);
    const previous = duplicate(this.db, auth);
    if (previous) return previous;
    const entry = this.state.deliveries[auth.payload.messageId];
    if (!entry) fail("UNKNOWN_MESSAGE", "No durable outbox message");
    if (entry.relayMessageId && entry.relayMessageId !== auth.payload.relayMessageId)
      fail("DELIVERY_CONFLICT", "Message already has a different delivery identity");
    const result = document(auth.payload);
    this.state = commit(
      this.db,
      {
        ...this.state,
        deliveries: {
          ...this.state.deliveries,
          [auth.payload.messageId]: { ...entry, relayMessageId: auth.payload.relayMessageId },
        },
      },
      auth,
      "MESSAGE_SENT",
      result,
      result,
    );
    return result;
  }

  @Atomic
  async acknowledge(request: SignedRequest): Promise<Document> {
    const auth = authorize(
      request,
      actorName,
      this.id,
      "acknowledge",
      ["director", "data", "worker", "evaluator", "reviewer", "coordinator"],
      z.strictObject({ messageId: id }),
    );
    scope(request, this.id);
    const previous = duplicate(this.db, auth);
    if (previous) return previous;
    const entry = this.state.deliveries[auth.payload.messageId];
    if (!entry || entry.message.recipientRole !== auth.role)
      fail("MESSAGE_ROLE", "Only the assigned recipient role may acknowledge");
    if (!entry.relayMessageId) fail("NOT_DELIVERED", "Relay delivery must be recorded first");
    const result = { messageId: auth.payload.messageId, acknowledged: true };
    this.state = commit(
      this.db,
      {
        ...this.state,
        deliveries: {
          ...this.state.deliveries,
          [auth.payload.messageId]: { ...entry, acknowledged: true },
        },
      },
      auth,
      "MESSAGE_ACKNOWLEDGED",
      result,
      result,
    );
    return result;
  }
  async getAuthorization(experimentId: string): Promise<Document | null> {
    id.parse(experimentId);
    const review = this.state.reviews[experimentId];
    return this.state.assignments[experimentId] && review ? document(review) : null;
  }
  async getSnapshot(): Promise<Document> {
    return document({
      campaignId: this.id,
      deliveries: this.state.deliveries,
      reviews: this.state.reviews,
      assignments: this.state.assignments,
      version: this.state.version,
    });
  }
  async getEvents(after: number, limit: number): Promise<Document> {
    return events(this.db, this.state.version, after, limit);
  }
}
