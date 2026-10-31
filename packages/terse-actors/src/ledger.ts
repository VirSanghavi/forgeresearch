import { Actor, Atomic, Persisted } from "durable-actors";
import { z } from "zod";
import { hash, id, type Document, type Journal, type SignedRequest } from "./contracts.js";
import {
  actorRead,
  authorize,
  commit,
  document,
  duplicate,
  events,
  fail,
  journal,
  requireScopedId,
} from "./core.js";

const evidenceSchema = z.discriminatedUnion("source", [
  z.strictObject({ source: z.literal("baseline"), reportSha256: hash }),
  z.strictObject({ source: z.literal("evaluation"), experimentId: id, reportSha256: hash }),
  z.strictObject({ source: z.literal("failure"), experimentId: id }),
]);
const rejectedSchema = z.array(
  z.strictObject({ option: z.string().min(1).max(200), reason: z.string().min(1).max(1000) }),
);
const launchSchema = z
  .array(z.looseObject({ experimentId: id, label: id, recipe: z.record(z.string(), z.json()) }))
  .max(4);
const entrySchema = z.strictObject({
  entryId: id,
  kind: z.enum(["diagnosis", "hypothesis", "decision", "stop"]),
  summary: z.string().min(1).max(2000),
  parents: z.array(id).max(16),
  evidence: z.array(evidenceSchema).max(32),
  body: z.record(z.string(), z.json()),
});
type Entry = z.infer<typeof entrySchema> & { sequence: number; authorKeyId: string };
interface LedgerState extends Journal {
  entries: Record<string, Entry>;
  launches: Record<string, string>;
  closedBy: string | null;
}
const actorName = "ResearchLedgerActor";

/**
 * Append-only research record. Decisions must cite measurements this actor can verify in
 * authoritative campaign and experiment state, and name the alternatives they rejected.
 * Recording a decision authorizes nothing: reservations and preflight review still apply.
 */
export class ResearchLedgerActor extends Actor {
  @Persisted private state: LedgerState = {
    ...journal(),
    entries: {},
    launches: {},
    closedBy: null,
  };

  @Atomic
  async record(request: SignedRequest): Promise<Document> {
    const auth = authorize(request, actorName, this.id, "record", ["director"], entrySchema);
    if (request.campaignId !== this.id)
      fail("CAMPAIGN_SCOPE", "Ledger belongs to another campaign");
    const previous = duplicate(this.db, auth);
    if (previous) return previous;
    const entry = auth.payload;
    if (this.state.closedBy) fail("LEDGER_CLOSED", "A stop decision already closed this ledger");
    if (this.state.entries[entry.entryId]) fail("ENTRY_IMMUTABLE", "Entry ID already exists");
    if (Object.keys(this.state.entries).length >= 1000)
      fail("LEDGER_LIMIT", "Create a new campaign before reaching capacity");
    for (const parent of entry.parents)
      if (!this.state.entries[parent]) fail("UNKNOWN_PARENT", "Parent entries must exist first");
    if (entry.kind === "hypothesis" ? !entry.parents.length : !entry.evidence.length)
      fail("EVIDENCE_REQUIRED", "Hypotheses need a parent; other entries must cite evidence");
    await this.verifyEvidence(entry.evidence);
    let launches = this.state.launches;
    if (entry.kind === "decision") {
      const body = z
        .looseObject({ rejected: rejectedSchema, launch: launchSchema })
        .safeParse(entry.body);
      if (!body.success)
        fail("DECISION_SHAPE", "Decisions list launched experiments and rejected alternatives");
      for (const launch of body.data.launch) {
        requireScopedId("exp", this.id, launch.experimentId);
        if (launches[launch.experimentId])
          fail("LAUNCH_EXISTS", "Another decision already launched this experiment");
        launches = { ...launches, [launch.experimentId]: entry.entryId };
      }
    }
    if (entry.kind === "stop" && !z.looseObject({ rule: id }).safeParse(entry.body).success)
      fail("STOP_RULE_REQUIRED", "Stop entries name the rule that ended the campaign");
    const stored: Entry = {
      ...entry,
      body: document(entry.body),
      sequence: this.state.version + 1,
      authorKeyId: auth.keyId,
    };
    const result = document({ entryId: entry.entryId, sequence: stored.sequence });
    this.state = commit(
      this.db,
      {
        ...this.state,
        entries: { ...this.state.entries, [entry.entryId]: stored },
        launches,
        closedBy: entry.kind === "stop" ? entry.entryId : null,
      },
      auth,
      "LEDGER_RECORDED",
      result,
      { entryId: entry.entryId, kind: entry.kind },
    );
    return result;
  }

  private async verifyEvidence(evidence: z.infer<typeof evidenceSchema>[]): Promise<void> {
    for (const item of evidence) {
      if (item.source === "baseline") {
        const campaign = z
          .object({ spec: z.object({ baseline: z.object({ reportSha256: hash }) }) })
          .safeParse(await actorRead("ResearchCampaignActor", this.id, "getSnapshot"));
        if (!campaign.success || campaign.data.spec.baseline.reportSha256 !== item.reportSha256)
          fail("EVIDENCE_MISMATCH", "Baseline evidence differs from the frozen campaign baseline");
        continue;
      }
      requireScopedId("exp", this.id, item.experimentId);
      const experiment = z
        .object({
          campaignId: z.string().nullable(),
          status: z.string(),
          evaluation: z.object({ reportSha256: hash }).nullable(),
        })
        .safeParse(await actorRead("ExperimentActor", item.experimentId, "getSnapshot"));
      const ok =
        experiment.success &&
        experiment.data.campaignId === this.id &&
        (item.source === "failure"
          ? experiment.data.status === "failed"
          : experiment.data.status === "evaluated" &&
            experiment.data.evaluation?.reportSha256 === item.reportSha256);
      if (!ok) fail("EVIDENCE_MISMATCH", "Experiment evidence is not recorded in actor state");
    }
  }

  async getLaunch(experimentId: string): Promise<Document | null> {
    id.parse(experimentId);
    const entryId = this.state.launches[experimentId];
    if (!entryId) return null;
    const entry = this.state.entries[entryId];
    const launch = (entry.body.launch as Document[]).find((l) => l.experimentId === experimentId);
    return document({ entryId, authorKeyId: entry.authorKeyId, launch });
  }
  async getSnapshot(): Promise<Document> {
    return document({
      campaignId: this.id,
      entries: this.state.entries,
      launches: this.state.launches,
      closedBy: this.state.closedBy,
      version: this.state.version,
    });
  }
  async getEvents(after: number, limit: number): Promise<Document> {
    return events(this.db, this.state.version, after, limit);
  }
}
