import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import type { Document } from "../../../packages/terse-actors/src/contracts.js";
import { canonical } from "../../../packages/terse-actors/src/core.js";
import type { RelayDispatcher } from "./relay.js";
import { exists, fileHash } from "./worker.js";

export interface Reconciliation {
  unsent: string[];
  unacknowledged: string[];
  /** Experiments the actor records as trained, with no TRAINING_COMPLETED queued yet. */
  unannouncedCompletions: string[];
  /** Running experiments whose worker lease expired: a new lease or an abort is required. */
  expiredLeases: string[];
}

/** Reads only authoritative actor state; never infers progress from Relay traffic. */
export async function reconcile(
  dispatcher: RelayDispatcher,
  now = Date.now(),
): Promise<Reconciliation> {
  const { actors, campaignId } = dispatcher;
  const workflow = await actors.coordinator.read(
    "ResearchWorkflowActor",
    campaignId,
    "getSnapshot",
  );
  const deliveries = await dispatcher.deliveries();
  const report: Reconciliation = {
    ...(await dispatcher.pending()),
    unannouncedCompletions: [],
    expiredLeases: [],
  };
  for (const experimentId of Object.keys(workflow!.assignments as Document)) {
    const experiment = (await actors.coordinator.read(
      "ExperimentActor",
      experimentId,
      "getSnapshot",
    ))!;
    const announced = deliveries.some(
      (d) => d.message.kind === "TRAINING_COMPLETED" && d.message.experimentId === experimentId,
    );
    if (["completed", "evaluated"].includes(String(experiment.status)) && !announced)
      report.unannouncedCompletions.push(experimentId);
    const lease = experiment.lease as Document | null;
    if (experiment.status === "running" && lease && Number(lease.expiresAtMs) <= now)
      report.expiredLeases.push(experimentId);
  }
  return report;
}

/**
 * Local CPU runs have no provider bill. Settle at the reserved upper bound and label it
 * unmeasured, so released budget never depends on an invented cost.
 */
export async function settleAtReservedBound(
  dispatcher: RelayDispatcher,
  experimentId: string,
  evidencePath: string,
): Promise<Document> {
  const { actors, campaignId } = dispatcher;
  const campaign = await actors.coordinator.read(
    "ResearchCampaignActor",
    campaignId,
    "getSnapshot",
  );
  const reservation = (campaign!.reservations as Document)[experimentId] as Document;
  const claims = (reservation.grant as Document).claims as Document;
  const reservedMicroUsd = Number((claims.proposal as Document).maxCostMicroUsd);
  const evidence = canonical({
    campaignId,
    experimentId,
    grantId: claims.grantId,
    basis: "reserved-upper-bound",
    measured: false,
    provider: "local-process",
    chargedMicroUsd: reservedMicroUsd,
  });
  // A retry after a lost reply reuses identical evidence; anything else is a conflict.
  if (!(await exists(evidencePath))) {
    await mkdir(dirname(evidencePath), { recursive: true });
    await writeFile(evidencePath, evidence, { flag: "wx" });
  } else if ((await readFile(evidencePath, "utf8")) !== evidence)
    throw new Error("COST_EVIDENCE_CONFLICT: existing settlement evidence differs");
  return actors.coordinator.command(
    "ResearchCampaignActor",
    campaignId,
    campaignId,
    "settle",
    {
      experimentId,
      actualMicroUsd: reservedMicroUsd,
      costEvidenceSha256: await fileHash(evidencePath),
    },
    `settle-${experimentId}`,
  );
}
