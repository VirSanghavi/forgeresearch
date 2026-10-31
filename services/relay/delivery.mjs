// Every Forge handoff is a durable Agent Relay message. 1:1 handoffs travel as direct
// messages, decisions and announcements as research-room posts, and governance verdicts
// as thread replies under the checkpoint they judge. A message counts as delivered only
// after each recipient drains it from its own Relay inbox, verifies sender, route and
// exact bytes, acks the inbox delivery and marks it read.
export const roles = [
  "director",
  "data",
  "trainer",
  "evaluator",
  "governance",
  "hephaestus",
];
export const reactions = ["white_check_mark", "x", "eyes"];

const ID = /^[a-zA-Z0-9_-]{1,100}$/;
const RELAY_ID = /^[a-zA-Z0-9_-]{1,64}$/;
/** How long a recipient waits for the push before draining its inbox anyway. */
const PUSH_WAIT_MS = Number(process.env.FORGE_RELAY_PUSH_WAIT_MS || 4000);

export function validate(m) {
  const route = m.route ?? "dm";
  if (
    !roles.includes(m.sender) ||
    !roles.includes(m.recipient) ||
    !["dm", "room"].includes(route) ||
    (route === "dm" && m.sender === m.recipient) ||
    (m.reply_to != null &&
      (route !== "room" ||
        typeof m.reply_to !== "string" ||
        !RELAY_ID.test(m.reply_to))) ||
    typeof m.id !== "string" ||
    !ID.test(m.id) ||
    typeof m.campaign_id !== "string" ||
    !ID.test(m.campaign_id) ||
    typeof m.category !== "string" ||
    !/^[A-Z_]{1,60}$/.test(m.category) ||
    (m.experiment_id != null &&
      (typeof m.experiment_id !== "string" || !ID.test(m.experiment_id))) ||
    typeof m.text !== "string" ||
    !m.text.trim() ||
    m.text.length > 4000
  )
    throw new Error("Invalid message contract");
  return route;
}

/** The structured half of the contract travels as Relay metadata; the text stays readable. */
export function envelope(m) {
  return {
    forge: {
      id: m.id,
      campaign_id: m.campaign_id,
      sender: m.sender,
      recipient: m.recipient,
      category: m.category,
      experiment_id: m.experiment_id ?? null,
    },
  };
}

function sameEnvelope(metadata, expected) {
  const forge = metadata?.forge;
  if (!forge || typeof forge !== "object") return false;
  return Object.entries(expected.forge).every(
    ([key, value]) => (forge[key] ?? null) === value,
  );
}

/**
 * Wakes recipients on Relay's push stream (an observer WebSocket), so nobody polls for
 * mail. A recipient that sees no push within PUSH_WAIT_MS still drains its inbox, and the
 * receipt says so: delivery never depends on the push arriving.
 */
export function pushWaiter() {
  const seen = new Map();
  const waiters = new Map();
  const forget = () => {
    const cutoff = Date.now() - 10 * 60 * 1000;
    for (const [id, at] of seen) if (at < cutoff) seen.delete(id);
  };
  return {
    observe(event) {
      const id = event?.message?.messageId ?? event?.message?.id;
      if (typeof id !== "string") return;
      seen.set(id, Date.now());
      for (const resolve of waiters.get(id) ?? []) resolve("push");
      waiters.delete(id);
      if (seen.size > 2000) forget();
    },
    wait(id, ms = PUSH_WAIT_MS) {
      if (seen.has(id)) return Promise.resolve("push");
      return new Promise((resolve) => {
        const list = waiters.get(id) ?? [];
        const timer = setTimeout(() => resolve("inbox-sweep"), ms);
        list.push((how) => {
          clearTimeout(timer);
          resolve(how);
        });
        waiters.set(id, list);
      });
    },
  };
}

export function dispatcher(agents, push = pushWaiter()) {
  const rooms = new Map();
  const pending = new Map();
  async function room(campaign) {
    if (!rooms.has(campaign)) {
      const ready = (async () => {
        const name = `research-${campaign}`;
        // List first: reconnecting must reuse the existing room after a restart.
        const existing = await agents.director.channels.list();
        if (!existing.some((channel) => channel.name === name)) {
          await agents.director.channels.create({
            name,
            topic: "Forge research room: decisions, checkpoints and verdicts",
          });
        }
        for (const role of roles) await agents[role].channels.join(name);
        const channel = await agents.director.channels.get(name);
        const members = await agents.director.channels.members(name);
        if (
          !roles.every((role) =>
            members.some((member) => member.agentName === role),
          )
        )
          throw new Error("Campaign room is missing a research participant");
        return { name, id: channel.id };
      })();
      rooms.set(campaign, ready);
      ready.catch(() => rooms.delete(campaign));
    }
    return rooms.get(campaign);
  }

  /**
   * One recipient takes the message from its durable inbox. A copy that does not match is
   * failed in Relay and stops the handoff; a retry after a lost response finds the
   * delivery already acked and proves receipt from the read receipt instead.
   */
  async function take(role, sent, verify, routed) {
    const agent = agents[role];
    const { items } = await agent.inbox.list({ limit: 200 });
    const item = items.find((row) => row.message?.messageId === sent.messageId);
    if (item) {
      if (!verify(item.message, item.metadata?.reason)) {
        await agent.inbox.fail({
          inboxItemId: item.id,
          error: "FORGE_COPY_MISMATCH",
        });
        throw new Error(
          `Relay copy for ${role} did not match the durable message`,
        );
      }
      const acked = await agent.inbox.ack({ inboxItemId: item.id });
      const read = await agent.messages.markRead(sent.messageId);
      return {
        role,
        delivery_id: item.id,
        state: acked.supported ? acked.state : "read",
        read_at: read.readAt ?? null,
        source: "inbox",
      };
    }
    // No queued delivery: either already acked by an earlier attempt, or a route the engine
    // does not queue (thread replies). Read the message itself and verify it.
    const copy = await agent.messages.get(sent.messageId);
    if (!verify(copy, null) || !(await routed(agent, copy)))
      throw new Error(`Relay readback for ${role} did not match`);
    const read = await agent.messages.markRead(sent.messageId);
    return {
      role,
      delivery_id: null,
      state: "read",
      read_at: read.readAt ?? null,
      source: "readback",
    };
  }

  async function deliver(m) {
    const route = validate(m);
    const bytes = JSON.stringify(m);
    const key = `${m.campaign_id}:${m.id}`;
    const prior = pending.get(key);
    if (prior) {
      if (prior.bytes !== bytes) throw new Error("Message ID conflict");
      return prior.promise;
    }
    const promise = (async () => {
      const channel = await room(m.campaign_id);
      const metadata = envelope(m);
      let sent;
      if (route === "dm") {
        sent = await agents[m.sender].sendMessage({
          to: `@${m.recipient}`,
          text: m.text,
          metadata,
          idempotencyKey: `${key}:dm`,
        });
      } else if (m.reply_to) {
        const parent = await agents[m.sender].messages.get(m.reply_to);
        if (parent.channel?.id !== channel.id)
          throw new Error("Thread parent is not in this research room");
        sent = await agents[m.sender].reply({
          messageId: m.reply_to,
          text: m.text,
          metadata,
          idempotencyKey: `${key}:thread`,
        });
      } else {
        sent = await agents[m.sender].sendMessage({
          to: `#${channel.name}`,
          text: m.text,
          metadata,
          idempotencyKey: `${key}:room`,
        });
      }
      const wake = await push.wait(sent.messageId);
      // Inbox rows say how they were routed; a readback proves the route from Relay itself.
      const verify = (copy, reason) =>
        copy?.text === m.text &&
        copy?.from?.name === m.sender &&
        (copy.metadata === undefined ||
          sameEnvelope(copy.metadata, metadata)) &&
        (reason === null ||
          (route === "dm"
            ? reason === "dm"
            : copy.channel?.id === undefined ||
              copy.channel.id === channel.id));
      const routed = async (agent, copy) => {
        if (route !== "dm") return copy.channel?.id === channel.id;
        if (!sent.conversationId) return false;
        const direct = await agent.messages.listDirect({
          conversationId: sent.conversationId,
          limit: 100,
        });
        return direct.some((row) => row.messageId === sent.messageId);
      };
      const readers =
        route === "dm"
          ? [m.recipient]
          : roles.filter((role) => role !== m.sender);
      const receipts = [];
      for (const role of readers)
        receipts.push(await take(role, sent, verify, routed));
      const recorded = await agents[m.sender].messages.get(sent.messageId);
      if (
        recorded.text !== m.text ||
        !sameEnvelope(recorded.metadata, metadata)
      )
        throw new Error("Relay record does not match the dispatched message");
      const readBy = await agents[m.sender].messages.readers(sent.messageId);
      const readNames = new Set(readBy.map((row) => row.agentName));
      if (!readers.every((role) => readNames.has(role)))
        throw new Error("Relay read receipts are missing a recipient");
      return {
        message_id: sent.messageId,
        delivered: true,
        recipient: m.recipient,
        transport: "Agent Relay",
        route: m.reply_to ? "thread" : route,
        kind: recorded.kind ?? null,
        conversation_id: sent.conversationId ?? recorded.conversationId ?? null,
        thread_parent_id: m.reply_to ?? null,
        woke_by: wake,
        recipient_readback_verified: true,
        deliveries: receipts,
        read_by: readBy.map((row) => ({
          name: row.agentName,
          read_at: row.readAt ?? null,
        })),
        room: { name: channel.name, id: channel.id },
      };
    })();
    pending.set(key, { bytes, promise });
    promise.catch(() => pending.delete(key));
    return promise;
  }

  /** A role reacts to a research-room message; the reaction is Relay state, not a copy. */
  async function react({ campaign_id, sender, message_id, emoji }) {
    if (
      !roles.includes(sender) ||
      typeof campaign_id !== "string" ||
      !ID.test(campaign_id) ||
      typeof message_id !== "string" ||
      !RELAY_ID.test(message_id) ||
      !reactions.includes(emoji)
    )
      throw new Error("Invalid reaction contract");
    const channel = await room(campaign_id);
    const target = await agents[sender].messages.get(message_id);
    if (target.channel?.id !== channel.id)
      throw new Error("Reaction target is not in this research room");
    await agents[sender].react({ messageId: message_id, emoji });
    const groups = await agents[sender].messages.reactions(message_id);
    const mine = groups.find((group) => group.emoji === emoji);
    if (!mine?.agents?.includes(sender))
      throw new Error("Relay did not record the reaction");
    return {
      message_id,
      emoji,
      by: sender,
      reactions: groups.map((group) => ({
        emoji: group.emoji,
        count: group.count,
        agents: group.agents,
      })),
    };
  }

  /** Relay's own full-text search over the campaign room. */
  async function search({ campaign_id, q }) {
    if (
      typeof campaign_id !== "string" ||
      !ID.test(campaign_id) ||
      typeof q !== "string" ||
      !q.trim() ||
      q.length > 200
    )
      throw new Error("Invalid search");
    const channel = await room(campaign_id);
    const rows = await agents.director.messages.search(q.trim(), {
      channel: channel.name,
      limit: 25,
    });
    return rows.map((row) => ({
      message_id: row.id,
      sender: row.agentName,
      text: row.text,
      created_at: row.createdAt ?? null,
    }));
  }

  return { deliver, react, search };
}
