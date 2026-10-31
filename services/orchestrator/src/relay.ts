import { AgentRelay, type RelayAgentClient } from "@agent-relay/sdk";
import type { ForgeActorClient } from "../../../packages/terse-actors/src/client.js";
import type { Document, Role } from "../../../packages/terse-actors/src/contracts.js";
import { canonical, digest } from "../../../packages/terse-actors/src/core.js";
import type { Delivery, WorkflowMessage } from "../../../packages/terse-actors/src/workflow.js";

export const participantRoles = [
  "director",
  "data",
  "reviewer",
  "coordinator",
  "worker",
  "evaluator",
] as const;
export type ParticipantRole = (typeof participantRoles)[number];
export type Participants = Record<ParticipantRole, RelayAgentClient>;

export async function registerParticipants(relay: AgentRelay): Promise<Participants> {
  const participants = {} as Participants;
  // Workflow messages are 1:1 handoffs, so each travels as a Relay direct message into the
  // recipient's durable inbox. Participants never open the SDK's node WebSocket: it acks
  // deliveries on receipt, and a Forge delivery is acked only after its handler has run.
  for (const role of participantRoles)
    participants[role] = await relay.workspace.register(
      { name: `forge-${role}`, type: "agent" },
      { strict: true },
    );
  return participants;
}

export interface ReceiveReport {
  /** Handled from a delivery in the recipient's durable Relay inbox. */
  viaRelay: number;
  /** Recorded as sent but absent from the inbox page; fetched by its Relay ID. */
  viaReconciliation: number;
  /** Extra verified copies of a pending message; handled once, acked in Relay. */
  duplicateCopies: number;
  /** Copies claiming a pending message ID whose sender, route or bytes do not match; failed in Relay. */
  rejectedCopies: number;
  /** Inbox deliveries of messages Forge already acknowledged (a lost Relay ack); acked, not re-run. */
  alreadyHandled: number;
}

type InboxItem = Awaited<ReturnType<RelayAgentClient["inbox"]["list"]>>["items"][number];

/** Reads a workflow message ID from untrusted inbox text; anything else is not workflow traffic. */
function workflowMessageId(text: unknown): string | undefined {
  if (typeof text !== "string") return undefined;
  try {
    const parsed = JSON.parse(text) as Document | null;
    return typeof parsed?.messageId === "string" ? parsed.messageId : undefined;
  } catch {
    return undefined;
  }
}

/** The persisted outbox is authoritative. SDK transport acceptance is not work completion. */
export class RelayDispatcher {
  constructor(
    readonly campaignId: string,
    readonly actors: Record<Role, ForgeActorClient>,
    readonly participants: Participants,
    /** Inbox page size; older pending messages are reconciled by their recorded Relay ID. */
    readonly historyLimit = 100,
  ) {}
  async deliveries(): Promise<Delivery[]> {
    const snapshot = await this.actors.coordinator.read(
      "ResearchWorkflowActor",
      this.campaignId,
      "getSnapshot",
    );
    const rows = Object.values(snapshot!.deliveries as unknown as Record<string, Delivery>);
    return rows.sort((a, b) => a.sequence - b.sequence);
  }
  async pending(): Promise<{ unsent: string[]; unacknowledged: string[] }> {
    const rows = await this.deliveries();
    return {
      unsent: rows.filter((d) => !d.relayMessageId).map((d) => d.message.messageId),
      unacknowledged: rows
        .filter((d) => d.relayMessageId && !d.acknowledged)
        .map((d) => d.message.messageId),
    };
  }
  async flush(afterSend?: (delivery: Delivery, relayMessageId: string) => Promise<void>) {
    let sent = 0;
    for (const delivery of await this.deliveries()) {
      if (delivery.acknowledged || delivery.relayMessageId) continue;
      const m = delivery.message;
      // The idempotency key lets a resend after a lost response return the original DM.
      const result = await this.participants[m.senderRole as ParticipantRole].sendMessage({
        to: `@forge-${m.recipientRole}`,
        text: canonical(m),
        idempotencyKey: digest({ campaignId: this.campaignId, messageId: m.messageId }),
      });
      // Failure injection exercises lost responses before recording the transport receipt.
      if (afterSend) await afterSend(delivery, result.messageId);
      await this.actors.coordinator.command(
        "ResearchWorkflowActor",
        this.campaignId,
        this.campaignId,
        "markSent",
        { messageId: m.messageId, relayMessageId: result.messageId },
        `sent-${m.messageId}`,
      );
      sent++;
    }
    return sent;
  }
  /**
   * Drains the recipient's durable Relay inbox, then reconciles anything the outbox recorded
   * as sent that the inbox page did not return. A delivery is acked in Relay only after its
   * handler ran and Forge recorded the acknowledgement; forged copies are failed in Relay.
   * Handlers receive the durable message, never transport text, and must be idempotent: a
   * crash before the Forge acknowledgement re-runs them.
   */
  async receive(
    role: ParticipantRole,
    handler: (message: WorkflowMessage) => Promise<void>,
  ): Promise<ReceiveReport> {
    const report: ReceiveReport = {
      viaRelay: 0,
      viaReconciliation: 0,
      duplicateCopies: 0,
      rejectedCopies: 0,
      alreadyHandled: 0,
    };
    const client = this.participants[role];
    const rows = await this.deliveries();
    const byId = new Map(
      rows
        .filter((d) => d.relayMessageId && d.message.recipientRole === role)
        .map((d) => [d.message.messageId, d]),
    );
    const { items } = await client.inbox.list({ limit: this.historyLimit });
    const genuine = (
      incoming: { from?: { name?: string }; text?: unknown },
      expected: WorkflowMessage,
    ) =>
      incoming.from?.name === `forge-${expected.senderRole}` &&
      incoming.text === canonical(expected);
    const copies = new Map<string, InboxItem[]>();
    for (const item of items) {
      const messageId = workflowMessageId(item.message?.text);
      const delivery = messageId ? byId.get(messageId) : undefined;
      if (!delivery) {
        // Not a workflow message for this campaign: consumed so it cannot fill the inbox page.
        await client.inbox.ack({ inboxItemId: item.id });
        continue;
      }
      if (delivery.acknowledged) {
        // Forge recorded the work before a crash lost the Relay ack; close the delivery only.
        if (item.metadata?.reason === "dm" && genuine(item.message, delivery.message)) {
          await client.inbox.ack({ inboxItemId: item.id });
          report.alreadyHandled++;
        } else {
          await client.inbox.fail({ inboxItemId: item.id, error: "RELAY_COPY_MISMATCH" });
          report.rejectedCopies++;
        }
        continue;
      }
      copies.set(messageId!, [...(copies.get(messageId!) ?? []), item]);
    }
    for (const delivery of rows) {
      if (delivery.acknowledged || !byId.has(delivery.message.messageId)) continue;
      const expected = delivery.message;
      const seen = copies.get(expected.messageId) ?? [];
      const verified = seen.filter(
        (item) => item.metadata?.reason === "dm" && genuine(item.message, expected),
      );
      const rejected = seen.filter((item) => !verified.includes(item));
      if (verified.length) report.viaRelay++;
      else {
        const recorded = await client.messages.get(delivery.relayMessageId!);
        if (!genuine(recorded, expected))
          throw new Error("RELAY_EVIDENCE_MISMATCH: delivery does not match the durable message");
        report.viaReconciliation++;
      }
      await handler(expected);
      await this.actors[role].command(
        "ResearchWorkflowActor",
        this.campaignId,
        this.campaignId,
        "acknowledge",
        { messageId: expected.messageId },
        `ack-${expected.messageId}`,
      );
      // Relay transitions follow the authoritative Forge acknowledgement.
      for (const item of verified) await client.inbox.ack({ inboxItemId: item.id });
      if (verified.length) await client.messages.markRead(verified[0].message.messageId);
      report.duplicateCopies += Math.max(0, verified.length - 1);
      for (const item of rejected)
        await client.inbox.fail({ inboxItemId: item.id, error: "RELAY_COPY_MISMATCH" });
      report.rejectedCopies += rejected.length;
    }
    return report;
  }
  async publish(
    role: ParticipantRole,
    messageId: string,
    experimentId: string,
    kind: string,
    body: Document,
  ): Promise<void> {
    await this.actors[role].command(
      "ResearchWorkflowActor",
      this.campaignId,
      this.campaignId,
      "publish",
      { messageId, experimentId, kind, body },
      `queue-${messageId}`,
    );
  }
}
