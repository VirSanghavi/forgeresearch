import { Actor, Atomic, Persisted } from "durable-actors";
import { z } from "zod";
import {
  hash,
  id,
  reviewSchema,
  type Document,
  type GovernanceState,
  type SignedRequest,
} from "./contracts.js";
import { authorize, commit, document, duplicate, events, fail, journal } from "./core.js";

const actorName = "GovernanceActor";
export class GovernanceActor extends Actor {
  @Persisted private state: GovernanceState = {
    ...journal(),
    campaignId: null,
    objectiveSha256: null,
    policyVersion: null,
    decisions: {},
  };

  @Atomic
  async initialize(request: SignedRequest): Promise<Document> {
    const auth = authorize(
      request,
      actorName,
      this.id,
      "initialize",
      ["owner"],
      z.strictObject({ objectiveSha256: hash, policyVersion: id }),
    );
    if (request.campaignId !== this.id) fail("CAMPAIGN_SCOPE", "Wrong governance authority");
    const previous = duplicate(this.db, auth);
    if (previous) return previous;
    if (this.state.campaignId) fail("POLICY_IMMUTABLE", "Governance policy is already initialized");
    const result = { ...auth.payload, campaignId: this.id };
    this.state = commit(
      this.db,
      { ...this.state, ...result },
      auth,
      "GOVERNANCE_INITIALIZED",
      result,
      result,
    );
    return result;
  }

  @Atomic
  async recordDecision(request: SignedRequest): Promise<Document> {
    const auth = authorize(
      request,
      actorName,
      this.id,
      "recordDecision",
      ["reviewer"],
      reviewSchema,
    );
    if (request.campaignId !== this.id || this.state.campaignId !== this.id)
      fail("CAMPAIGN_SCOPE", "Unknown governance authority");
    const previous = duplicate(this.db, auth);
    if (previous) return previous;
    const review = { ...auth.payload, reviewerKeyId: auth.keyId };
    if (
      review.objectiveSha256 !== this.state.objectiveSha256 ||
      review.policyVersion !== this.state.policyVersion
    )
      fail("POLICY_MISMATCH", "Review differs from the immutable policy");
    if (this.state.decisions[review.checkpointSha256])
      fail("DECISION_FINAL", "Checkpoint decisions are terminal and cannot be overwritten");
    const result = document(review);
    this.state = commit(
      this.db,
      { ...this.state, decisions: { ...this.state.decisions, [review.checkpointSha256]: review } },
      auth,
      review.decision === "approved" ? "CHECKPOINT_APPROVED" : "CHECKPOINT_REJECTED",
      result,
      result,
    );
    return result;
  }

  async getDecision(checkpointSha256: string): Promise<Document | null> {
    hash.parse(checkpointSha256);
    const decision = this.state.decisions[checkpointSha256];
    return decision ? document(decision) : null;
  }
  async getSnapshot(): Promise<Document> {
    return document({
      campaignId: this.state.campaignId,
      objectiveSha256: this.state.objectiveSha256,
      policyVersion: this.state.policyVersion,
      decisions: this.state.decisions,
      version: this.state.version,
    });
  }
  async getEvents(after: number, limit: number): Promise<Document> {
    return events(this.db, this.state.version, after, limit);
  }
}
