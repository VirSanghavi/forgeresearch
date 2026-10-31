import { Actor, Atomic, Persisted } from "durable-actors";
import { z } from "zod";
import { hash, integer, type Document, type Journal, type SignedRequest } from "./contracts.js";
import {
  actorRead,
  authorize,
  commit,
  document,
  duplicate,
  events,
  fail,
  journal,
} from "./core.js";

const actorName = "FinalEvaluationActor";
interface Frozen {
  experimentId: string;
  checkpointSha256: string;
  evaluationReportSha256: string;
  developmentCorrect: number;
  developmentExamples: number;
  objectiveSha256: string;
  governanceEvidence: string[];
  frozenAtMs: number;
}
interface Release {
  releaseSha256: string;
  checkpointSha256: string;
  developmentHarnessSha256: string;
  boundAtMs: number;
}
interface FinalState extends Journal {
  campaignId: string | null;
  frozen: Frozen | null;
  release: Release | null;
  final: Document | null;
}

const releaseSchema = z.strictObject({
  releaseSha256: hash,
  checkpointSha256: hash,
  developmentHarnessSha256: hash,
});
const pair = z.strictObject({ baselineCorrect: integer, candidateCorrect: integer });
const finalSchema = z
  .strictObject({
    releaseSha256: hash,
    checkpointSha256: hash,
    datasetSha256: hash,
    harness: z.strictObject({
      name: z.string().min(1).max(100),
      version: z.string().min(1).max(50),
      task: z.string().min(1).max(100),
    }),
    protocolSha256: hash,
    numExamples: integer.min(1).max(1_000_000),
    strict: pair,
    flexible: pair,
    finalImprovementSupported: z.boolean(),
    resultBundleSha256: hash,
  })
  .refine(
    (v) =>
      [v.strict, v.flexible].every(
        (p) => p.baselineCorrect <= v.numExamples && p.candidateCorrect <= v.numExamples,
      ),
    "Invalid final counts",
  );

/**
 * Phase 8 authority for the sealed final evaluation. The owner freezes the campaign's
 * promoted checkpoint once the campaign has stopped, so no later selection is possible,
 * then binds the one release bundle packaged from it. Only the separately keyed final
 * evaluator records one result, for exactly that release, once. Research roles can neither
 * freeze, bind nor record.
 */
export class FinalEvaluationActor extends Actor {
  @Persisted private state: FinalState = {
    ...journal(),
    campaignId: null,
    frozen: null,
    release: null,
    final: null,
  };

  @Atomic
  async freeze(request: SignedRequest): Promise<Document> {
    const auth = authorize(request, actorName, this.id, "freeze", ["owner"], z.strictObject({}));
    if (request.campaignId !== this.id)
      fail("CAMPAIGN_SCOPE", "Use the campaign ID as the actor ID");
    const previous = duplicate(this.db, auth);
    if (previous) return previous;
    if (this.state.frozen) fail("FREEZE_FINAL", "This campaign is already frozen");
    const campaign = (await actorRead("ResearchCampaignActor", this.id, "getSnapshot")) as Document;
    if (campaign.status !== "stopped")
      fail("CAMPAIGN_OPEN", "Stop the campaign before freezing; selection must be over");
    const best = campaign.best as Document | null;
    if (!best) fail("NO_PROMOTED_CHECKPOINT", "The campaign promoted no checkpoint to evaluate");
    const experiment = (await actorRead(
      "ExperimentActor",
      String(best.experimentId),
      "getSnapshot",
    )) as Document;
    const training = experiment.training as Document | null;
    const evaluation = experiment.evaluation as Document | null;
    if (!training || !evaluation || training.checkpointSha256 !== best.checkpointSha256)
      fail("FREEZE_EVIDENCE", "Promoted checkpoint does not match its experiment record");
    const decision = (await actorRead("GovernanceActor", this.id, "getDecision", [
      best.checkpointSha256,
    ])) as Document | null;
    if (decision?.decision !== "approved")
      fail("FREEZE_EVIDENCE", "Promoted checkpoint lacks an approved governance decision");
    const frozen: Frozen = {
      experimentId: String(best.experimentId),
      checkpointSha256: String(best.checkpointSha256),
      evaluationReportSha256: String(evaluation.reportSha256),
      developmentCorrect: Number(best.candidateCorrect),
      developmentExamples: Number(best.numExamples),
      objectiveSha256: String(campaign.objectiveSha256),
      governanceEvidence: (decision.evidenceRefs as string[]) ?? [],
      frozenAtMs: Date.now(),
    };
    const result = document(frozen);
    this.state = commit(
      this.db,
      { ...this.state, campaignId: this.id, frozen },
      auth,
      "CHECKPOINT_FROZEN",
      result,
      result,
    );
    return result;
  }

  /** The release bundle (training-core `forge-harness freeze`) of the frozen checkpoint. */
  @Atomic
  async bindRelease(request: SignedRequest): Promise<Document> {
    const auth = authorize(request, actorName, this.id, "bindRelease", ["owner"], releaseSchema);
    if (request.campaignId !== this.id)
      fail("CAMPAIGN_SCOPE", "Use the campaign ID as the actor ID");
    const previous = duplicate(this.db, auth);
    if (previous) return previous;
    const frozen = this.state.frozen;
    if (!frozen) fail("NOT_FROZEN", "Freeze a checkpoint before binding its release");
    if (auth.payload.checkpointSha256 !== frozen.checkpointSha256)
      fail("NOT_FROZEN_CHECKPOINT", "A release must package the frozen checkpoint");
    if (this.state.release) fail("RELEASE_FINAL", "This campaign's release is already bound");
    const release: Release = { ...auth.payload, boundAtMs: Date.now() };
    const result = document(release);
    this.state = commit(this.db, { ...this.state, release }, auth, "RELEASE_BOUND", result, result);
    return result;
  }

  @Atomic
  async recordFinal(request: SignedRequest): Promise<Document> {
    const auth = authorize(
      request,
      actorName,
      this.id,
      "recordFinal",
      ["final_evaluator"],
      finalSchema,
    );
    if (request.campaignId !== this.id)
      fail("CAMPAIGN_SCOPE", "Use the campaign ID as the actor ID");
    const previous = duplicate(this.db, auth);
    if (previous) return previous;
    const { frozen, release } = this.state;
    if (!frozen || !release) fail("NOT_RELEASED", "Freeze and bind a release before final");
    if (
      auth.payload.releaseSha256 !== release.releaseSha256 ||
      auth.payload.checkpointSha256 !== frozen.checkpointSha256
    )
      fail("NOT_FROZEN_RELEASE", "Only the bound release can be finally evaluated");
    if (this.state.final) fail("FINAL_ONCE", "The sealed final evaluation runs once");
    const final = document({
      ...auth.payload,
      evaluatorKeyId: auth.keyId,
      recordedAtMs: Date.now(),
    });
    this.state = commit(this.db, { ...this.state, final }, auth, "FINAL_RECORDED", final, {
      releaseSha256: release.releaseSha256,
      resultBundleSha256: auth.payload.resultBundleSha256,
    });
    return final;
  }

  async getSnapshot(): Promise<Document> {
    return document({
      campaignId: this.state.campaignId,
      frozen: this.state.frozen,
      release: this.state.release,
      final: this.state.final,
      version: this.state.version,
    });
  }
  async getEvents(after: number, limit: number): Promise<Document> {
    return events(this.db, this.state.version, after, limit);
  }
}
