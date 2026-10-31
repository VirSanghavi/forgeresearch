"use client";
import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { Agent, Campaign, Event, RelayReaction, agentNames } from "@/lib/types";
import { Icon } from "./icon";
import { HephaestusChat } from "./hephaestus-chat";

const members: Agent[] = [
  "director",
  "data",
  "trainer",
  "evaluator",
  "governance",
  "hephaestus",
];
const initials: Record<Agent, string> = {
  director: "RD",
  data: "DR",
  trainer: "TE",
  evaluator: "EV",
  governance: "GC",
  hephaestus: "HE",
};
const shortNames: Record<Agent, string> = {
  director: "Director",
  data: "Data",
  trainer: "Trainer",
  evaluator: "Evaluator",
  governance: "Governance",
  hephaestus: "Hephaestus",
};
/** The header cluster: the director in the middle, the team fanned out around it. */
const cluster: { agent: Agent; size: number }[] = [
  { agent: "evaluator", size: 38 },
  { agent: "data", size: 44 },
  { agent: "director", size: 54 },
  { agent: "trainer", size: 44 },
  { agent: "governance", size: 38 },
];
/** iMessage starts a new timestamp after a quiet gap of five minutes. */
const DIVIDER_MS = 5 * 60 * 1000;

type Route = "dm" | "room" | "thread";
type ChatMessage = {
  key: string;
  relayId?: string;
  sender: Agent;
  recipient?: Agent;
  category?: string;
  text: string;
  timestamp: string;
  experimentId?: string;
  route: Route;
  parentId?: string | null;
  receipt?: Event["receipt"];
  pending: boolean;
};
type Conversation = {
  id: string;
  kind: "room" | "dm";
  title: string;
  members: Agent[];
  /** Whose phone this is by default: the director in the room, the first member in a DM. */
  defaultViewer: Agent;
  messages: ChatMessage[];
};

const isAgent = (value: unknown): value is Agent =>
  typeof value === "string" && (members as string[]).includes(value);

const byTeamOrder = (p: Agent, q: Agent) =>
  members.indexOf(p) - members.indexOf(q);

function pairId(a: Agent, b: Agent) {
  const [x, y] = [a, b].sort(byTeamOrder);
  return `dm:${x}:${y}`;
}

/** Builds conversations from the durable event journal; tolerant of older receipts. */
function conversations(events: Event[]) {
  const delivered = events.filter(
    (e) => e.kind === "message" && e.relay_id && isAgent(e.agent),
  );
  const deliveredIds = new Set(delivered.map((e) => e.message_id));
  const messages: ChatMessage[] = delivered.map((e) => ({
    key: e.id,
    relayId: e.relay_id,
    sender: e.agent,
    recipient: isAgent(e.recipient) ? e.recipient : undefined,
    category: e.category,
    text: e.text,
    timestamp: e.timestamp,
    experimentId: e.experiment_id,
    // Earlier receipts posted every handoff to the room as well as the DM.
    route: e.route ?? e.receipt?.route ?? "room",
    parentId: e.thread_parent_id,
    receipt: e.receipt,
    pending: false,
  }));
  for (const e of events) {
    if (e.kind !== "message_pending" || !e.message?.id) continue;
    if (deliveredIds.has(e.message.id) || !isAgent(e.agent)) continue;
    const reply = (e.message as { reply_to?: string | null }).reply_to;
    messages.push({
      key: e.id,
      sender: e.agent,
      recipient: isAgent(e.message.recipient) ? e.message.recipient : undefined,
      category: e.message.category,
      text: e.text,
      timestamp: e.timestamp,
      route: reply ? "thread" : (e.message.route ?? "dm"),
      parentId: reply ?? null,
      pending: true,
    });
  }
  messages.sort((a, b) => a.timestamp.localeCompare(b.timestamp));
  const byRelay = new Map(
    messages.filter((m) => m.relayId).map((m) => [m.relayId!, m]),
  );
  const replies = new Map<string, ChatMessage[]>();
  const room: Conversation = {
    id: "room",
    kind: "room",
    title: "Research room",
    members,
    defaultViewer: "director",
    messages: [],
  };
  const dms = new Map<string, Conversation>();
  for (const m of messages) {
    if (m.route === "thread" && m.parentId && byRelay.has(m.parentId)) {
      replies.set(m.parentId, [...(replies.get(m.parentId) ?? []), m]);
      continue;
    }
    if (m.route !== "dm" || !m.recipient || m.recipient === m.sender) {
      room.messages.push(m);
      continue;
    }
    const id = pairId(m.sender, m.recipient);
    if (!dms.has(id)) {
      const pair = [m.sender, m.recipient].sort(byTeamOrder);
      dms.set(id, {
        id,
        kind: "dm",
        title: `${shortNames[pair[0]]} and ${shortNames[pair[1]]}`,
        members: pair,
        defaultViewer: pair[0],
        messages: [],
      });
    }
    dms.get(id)!.messages.push(m);
  }
  const reactions = new Map<string, RelayReaction[]>();
  for (const e of events)
    if (e.kind === "reaction" && e.relay_id && Array.isArray(e.reactions))
      reactions.set(e.relay_id, e.reactions);
  const list = [
    room,
    ...[...dms.values()].sort((a, b) =>
      last(b).timestamp.localeCompare(last(a).timestamp),
    ),
  ];
  return { list, replies, reactions };
}

const last = (c: Conversation) =>
  c.messages[c.messages.length - 1] ?? {
    timestamp: "",
    text: "",
    sender: "director" as Agent,
  };

/* Times follow the rest of Forge and read in UTC; the full instant is in each title. */
const DAY_MS = 24 * 60 * 60 * 1000;
const utcDay = (date: Date) => Math.floor(date.getTime() / DAY_MS);
function clock(iso: string) {
  return new Date(iso).toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    timeZone: "UTC",
  });
}
function dayLabel(iso: string) {
  const date = new Date(iso);
  const ago = utcDay(new Date()) - utcDay(date);
  if (ago === 0) return "Today";
  if (ago === 1) return "Yesterday";
  if (ago > 1 && ago < 7)
    return date.toLocaleDateString("en-US", {
      weekday: "long",
      timeZone: "UTC",
    });
  return date.toLocaleDateString("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
    ...(date.getUTCFullYear() !== new Date().getUTCFullYear()
      ? { year: "numeric" }
      : {}),
    timeZone: "UTC",
  });
}
function listTime(iso: string) {
  if (!iso) return "";
  const date = new Date(iso);
  const ago = utcDay(new Date()) - utcDay(date);
  if (ago === 0) return clock(iso);
  if (ago === 1) return "Yesterday";
  if (ago > 1 && ago < 7)
    return date.toLocaleDateString("en-US", {
      weekday: "long",
      timeZone: "UTC",
    });
  return date.toLocaleDateString("en-US", {
    month: "numeric",
    day: "numeric",
    year: "2-digit",
    timeZone: "UTC",
  });
}
const utcTitle = (iso: string) =>
  `${new Date(iso).toISOString().replace("T", " ").slice(0, 19)} UTC`;
// Categories whose plain lowercasing would mangle a name.
const CATEGORY_LABELS: Record<string, string> = {
  VM_RUNS: "Freestyle VM runs",
};
const humanize = (category?: string) =>
  category
    ? (CATEGORY_LABELS[category] ??
      category.charAt(0) + category.slice(1).replaceAll("_", " ").toLowerCase())
    : "";

/** The receipt under the newest outgoing bubble, from Relay's own read receipts. */
function receiptLabel(m: ChatMessage, kind: Conversation["kind"]) {
  if (m.pending) return { strong: "Sending", rest: "" };
  const readers = (m.receipt?.read_by ?? []).filter((r) => r.name !== m.sender);
  if (kind === "dm" || m.route === "dm") {
    const read = readers.find((r) => r.read_at) ?? readers[0];
    return read
      ? { strong: "Read", rest: clock(read.read_at ?? m.timestamp) }
      : { strong: "Delivered", rest: "" };
  }
  const count = readers.length || m.receipt?.room?.readers?.length || 0;
  return count
    ? { strong: "Read by", rest: String(count) }
    : { strong: "Delivered", rest: "" };
}

function Avatar({ agent, size = 30 }: { agent: Agent; size?: number }) {
  return (
    <span
      className="imsg-avatar"
      style={{ width: size, height: size, fontSize: Math.round(size * 0.4) }}
      aria-hidden="true"
    >
      {size >= 18 ? initials[agent] : null}
    </span>
  );
}

function ConversationAvatar({ c }: { c: Conversation }) {
  if (c.kind === "room")
    return (
      <span className="convo-avatar convo-avatar-group" aria-hidden="true">
        <Avatar agent="director" size={24} />
        <Avatar agent="data" size={21} />
        <Avatar agent="trainer" size={21} />
      </span>
    );
  return (
    <span className="convo-avatar convo-avatar-pair" aria-hidden="true">
      <Avatar agent={c.members[0]} size={27} />
      <Avatar agent={c.members[1]} size={27} />
    </span>
  );
}

/** The iMessage tail: a curved hook that flows out of the bubble's bottom corner. */
function Tail() {
  return (
    <svg
      className="bubble-tail"
      viewBox="0 0 16 20"
      width="16"
      height="20"
      aria-hidden="true"
    >
      <path d="M0 0H6C6 8.4 8.6 14.6 15.2 18.9C15.9 19.4 15.7 20 14.9 20C9.6 20.2 5.2 18.7 1.8 15.6L0 15.2Z" />
    </svg>
  );
}

function Tapback({
  reactions,
  viewer,
}: {
  reactions: RelayReaction[];
  viewer: Agent;
}) {
  const mine = reactions.some((r) => r.agents.includes(viewer));
  const label = reactions
    .map(
      (r) =>
        `${r.emoji === "white_check_mark" ? "Approved" : r.emoji === "x" ? "Rejected" : "Reacted"} by ${r.agents
          .map((a) => (isAgent(a) ? agentNames[a] : a))
          .join(", ")}`,
    )
    .join("; ");
  return (
    <span
      className={`tapback ${mine ? "is-mine" : ""}`}
      role="img"
      aria-label={label}
      title={label}
    >
      {reactions.map((r) => (
        <Icon
          key={r.emoji}
          name={
            r.emoji === "white_check_mark"
              ? "check"
              : r.emoji === "x"
                ? "close"
                : "message"
          }
          size={15}
        />
      ))}
    </span>
  );
}

function readSeen(campaignId: string): Record<string, number> {
  try {
    const raw = localStorage.getItem(`forge-chat-seen:${campaignId}`);
    const parsed = raw ? JSON.parse(raw) : {};
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

type SearchRow = { message_id: string; sender: string; text: string };

export function AgentChat({
  campaign,
  events,
  loading = false,
}: {
  campaign: Campaign | null;
  events: Event[];
  loading?: boolean;
}) {
  const roomName = campaign ? `research-${campaign.id}` : "research room";
  const { list, replies, reactions } = useMemo(
    () => conversations(events),
    [events],
  );
  const [activeId, setActiveId] = useState("room");
  const [private_, setPrivate] = useState(false);
  const [pane, setPane] = useState<"list" | "conversation">("conversation");
  const [thread, setThread] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [highlight, setHighlight] = useState<string | null>(null);
  const [follow, setFollow] = useState(true);
  const [details, setDetails] = useState(false);
  const [viewers, setViewers] = useState<Record<string, Agent>>({});
  // The component remounts per campaign, so the stored read state loads once.
  const [stored, setStored] = useState<Record<string, number>>(() =>
    campaign && typeof window !== "undefined" ? readSeen(campaign.id) : {},
  );
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchRow[] | null>(null);
  const [searchState, setSearchState] = useState<"idle" | "busy" | "error">(
    "idle",
  );
  const scroller = useRef<HTMLDivElement>(null);
  const active = list.find((c) => c.id === activeId) ?? list[0];
  const viewer =
    viewers[active.id] && active.members.includes(viewers[active.id])
      ? viewers[active.id]
      : active.defaultViewer;
  const other =
    active.kind === "dm"
      ? (active.members.find((a) => a !== viewer) ?? active.members[0])
      : undefined;
  const threadParent = thread
    ? active.messages.find((m) => m.relayId === thread)
    : undefined;
  const shown = threadParent
    ? [threadParent, ...(replies.get(threadParent.relayId!) ?? [])]
    : active.messages;
  const group = active.kind === "room";
  const lastMine = shown.findLastIndex((m) => m.sender === viewer);

  // Whatever is on screen counts as read for this viewer.
  const seen =
    pane === "list"
      ? stored
      : { ...stored, [active.id]: active.messages.length };
  const seenKey = JSON.stringify(seen);
  useEffect(() => {
    if (!campaign) return;
    try {
      localStorage.setItem(`forge-chat-seen:${campaign.id}`, seenKey);
    } catch {
      /* Unread state is a convenience; private windows may refuse it. */
    }
  }, [campaign, seenKey]);
  useEffect(() => {
    const el = scroller.current;
    if (!el) return;
    if (highlight) {
      document
        .getElementById(`msg-${highlight}`)
        ?.scrollIntoView({ block: "center" });
      return;
    }
    if (follow) el.scrollTop = el.scrollHeight;
  }, [shown.length, follow, activeId, thread, highlight, viewer]);

  useEffect(() => {
    if (selected)
      document
        .querySelector(".bubble-evidence")
        ?.scrollIntoView({ block: "nearest" });
  }, [selected]);

  const openConversation = (id: string, relayId?: string) => {
    setStored(seen);
    setActiveId(id);
    setThread(null);
    setDetails(false);
    setSelected(relayId ?? null);
    setHighlight(relayId ?? null);
    setFollow(!relayId);
    setPane("conversation");
  };
  const search = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!campaign || !query.trim()) return;
    setSearchState("busy");
    try {
      const r = await fetch(
        `/api/lab/campaigns/${campaign.id}/relay/search?q=${encodeURIComponent(query.trim())}`,
      );
      if (!r.ok) throw new Error(String(r.status));
      const rows = (await r.json()) as unknown;
      setResults(
        Array.isArray(rows)
          ? rows.filter(
              (row): row is SearchRow =>
                typeof row?.message_id === "string" &&
                typeof row?.text === "string",
            )
          : [],
      );
      setSearchState("idle");
    } catch {
      setResults(null);
      setSearchState("error");
    }
  };

  const unread = (c: Conversation) =>
    Math.max(0, c.messages.length - (seen[c.id] ?? 0));
  const threadCount = thread ? (replies.get(thread)?.length ?? 0) : 0;

  return (
    <section
      className="messenger"
      data-pane={pane}
      data-thread={thread ? "" : undefined}
      aria-label="Agent Relay conversations"
    >
      <aside className="convo-sidebar" aria-label="Conversations">
        <header className="convo-sidebar-head">
          <h2>Messages</h2>
          <p>
            <Icon name="connections" size={13} />
            Agent Relay
          </p>
        </header>
        <form className="convo-search" role="search" onSubmit={search}>
          <Icon name="search" size={15} />
          <input
            type="search"
            value={query}
            placeholder="Search the research room"
            aria-label="Search the research room with Agent Relay"
            disabled={!campaign}
            onChange={(e) => {
              setQuery(e.target.value);
              if (!e.target.value) {
                setResults(null);
                setSearchState("idle");
              }
            }}
          />
        </form>
        {searchState === "error" && (
          <p className="convo-search-note" role="status">
            Relay search is unavailable. Check that the bridge is running.
          </p>
        )}
        {results ? (
          <div className="convo-results" aria-live="polite">
            <div className="convo-results-head">
              <span>
                {results.length} {results.length === 1 ? "match" : "matches"} in
                Research room
              </span>
              <button
                className="text-button"
                onClick={() => {
                  setResults(null);
                  setQuery("");
                }}
              >
                Clear
              </button>
            </div>
            <ul>
              {results.map((row) => (
                <li key={row.message_id}>
                  <button
                    onClick={() => openConversation("room", row.message_id)}
                  >
                    <strong>
                      {isAgent(row.sender)
                        ? agentNames[row.sender]
                        : row.sender}
                    </strong>
                    <span>{row.text}</span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        ) : (
          <ul className="convo-list">
            <li>
              <button
                className={`convo-row ${private_ ? "active" : ""}`}
                aria-current={private_ ? "true" : undefined}
                onClick={() => {
                  setPrivate(true);
                  setPane("conversation");
                }}
              >
                <span className="convo-unread is-read" aria-hidden="true" />
                <span className="avatar-hephaestus" aria-hidden="true">
                  HE
                </span>
                <span className="convo-row-body">
                  <span className="convo-row-top">
                    <strong>Hephaestus</strong>
                  </span>
                  <span className="convo-row-preview">
                    Just you. Runs the team as admin.
                  </span>
                </span>
              </button>
            </li>
            {list.map((c) => {
              const tail = last(c);
              const count = campaign ? unread(c) : 0;
              const current = !private_ && c.id === active.id;
              return (
                <li key={c.id}>
                  <button
                    className={`convo-row ${current ? "active" : ""}`}
                    aria-current={current ? "true" : undefined}
                    aria-label={`${c.title}${count ? `, ${count} unread` : ""}`}
                    onClick={() => {
                      setPrivate(false);
                      openConversation(c.id);
                    }}
                  >
                    <span
                      className={`convo-unread ${count ? "" : "is-read"}`}
                      aria-hidden="true"
                    />
                    <ConversationAvatar c={c} />
                    <span className="convo-row-body">
                      <span className="convo-row-top">
                        <strong>{c.title}</strong>
                        <time
                          dateTime={tail.timestamp || undefined}
                          title={
                            tail.timestamp
                              ? utcTitle(tail.timestamp)
                              : undefined
                          }
                        >
                          {listTime(tail.timestamp)}
                        </time>
                      </span>
                      <span className="convo-row-preview">
                        {c.messages.length
                          ? c.kind === "room"
                            ? `${shortNames[tail.sender]}: ${tail.text}`
                            : tail.text
                          : c.kind === "room"
                            ? "Decisions, checkpoints and verdicts land here."
                            : "No messages yet"}
                      </span>
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </aside>

      {private_ && <HephaestusChat onBack={() => setPane("list")} />}
      <div className="convo-pane" hidden={private_}>
        <header className="convo-head">
          <div className="convo-head-start">
            <button
              className="convo-back"
              aria-label={
                thread ? `Back to ${active.title}` : "All conversations"
              }
              onClick={() => {
                if (thread) {
                  setThread(null);
                  setFollow(true);
                  return;
                }
                setStored(seen);
                setPane("list");
              }}
            >
              <Icon name="chevron" size={22} />
              {thread && <span className="convo-back-label">Back</span>}
            </button>
          </div>
          <div className="convo-head-center">
            {thread ? (
              <>
                <h3>Thread</h3>
                <p className="convo-head-sub">
                  {threadCount} {threadCount === 1 ? "reply" : "replies"} in{" "}
                  {active.title}
                </p>
              </>
            ) : (
              <>
                {group ? (
                  <span className="convo-cluster" aria-hidden="true">
                    {cluster.map(({ agent, size }) => (
                      <Avatar key={agent} agent={agent} size={size} />
                    ))}
                  </span>
                ) : (
                  <span className="convo-cluster" aria-hidden="true">
                    <Avatar agent={other!} size={48} />
                  </span>
                )}
                <h3>{group ? active.title : agentNames[other!]}</h3>
                <button
                  className="convo-head-sub convo-details-toggle"
                  aria-expanded={details}
                  aria-controls="convo-details"
                  onClick={() => setDetails(!details)}
                >
                  {group ? `${members.length} agents` : "Direct message"}
                  <Icon name="chevron" size={12} />
                </button>
              </>
            )}
          </div>
          <div className="convo-head-end">
            <label className="viewer-select">
              <span>Viewing as</span>
              <select
                value={viewer}
                onChange={(e) => {
                  const next = e.target.value as Agent;
                  setViewers((v) => ({ ...v, [active.id]: next }));
                  setFollow(true);
                }}
              >
                {active.members.map((a) => (
                  <option key={a} value={a}>
                    {agentNames[a]}
                  </option>
                ))}
              </select>
              <Icon name="down" size={14} />
            </label>
          </div>
        </header>
        {details && !thread && (
          <div
            className="convo-details"
            id="convo-details"
            aria-label="Conversation details"
          >
            <ul>
              {active.members.map((a) => (
                <li key={a}>
                  <Avatar agent={a} size={32} />
                  <span>
                    <strong>{agentNames[a]}</strong>
                    {a === viewer && <em>Viewing as</em>}
                  </span>
                </li>
              ))}
            </ul>
            <p className="mono">
              {group
                ? `#${roomName}`
                : (active.messages.find((m) => m.receipt?.conversation_id)
                    ?.receipt?.conversation_id ?? "Relay direct message")}
            </p>
          </div>
        )}
        <div
          className="convo-scroll"
          ref={scroller}
          tabIndex={0}
          aria-label={thread ? "Thread messages" : `${active.title} messages`}
          aria-busy={loading && !shown.length ? true : undefined}
          onScroll={() => {
            const el = scroller.current;
            if (el) {
              setFollow(el.scrollHeight - el.scrollTop - el.clientHeight < 80);
              if (highlight) setHighlight(null);
            }
          }}
        >
          {shown.length ? (
            <ol className="bubble-list">
              {shown.map((m, i) => {
                const prev = shown[i - 1];
                const next = shown[i + 1];
                const gap = (a?: ChatMessage, b?: ChatMessage) =>
                  a && b
                    ? Date.parse(b.timestamp) - Date.parse(a.timestamp)
                    : Infinity;
                const divider = !prev || gap(prev, m) >= DIVIDER_MS;
                // In a thread the original stands alone above its replies.
                const start =
                  divider || prev?.sender !== m.sender || (!!thread && i === 1);
                const end =
                  !next ||
                  next.sender !== m.sender ||
                  gap(m, next) >= DIVIDER_MS ||
                  (!!thread && i === 0);
                const mine = m.sender === viewer;
                const expanded = selected === m.key;
                const threadReplies =
                  !thread && m.relayId ? (replies.get(m.relayId) ?? []) : [];
                const tapbacks = m.relayId
                  ? reactions.get(m.relayId)
                  : undefined;
                const receipt =
                  (mine && i === lastMine) || m.pending
                    ? receiptLabel(m, active.kind)
                    : null;
                const meta = [
                  humanize(m.category),
                  group && m.recipient && m.recipient !== viewer
                    ? `to ${shortNames[m.recipient]}`
                    : group && m.recipient === viewer
                      ? "to you"
                      : "",
                ].filter(Boolean);
                return (
                  <li
                    key={m.key}
                    id={m.relayId ? `msg-${m.relayId}` : undefined}
                    className="bubble-item"
                  >
                    {divider && (
                      <p className="bubble-divider">
                        <time
                          dateTime={m.timestamp}
                          title={utcTitle(m.timestamp)}
                        >
                          <strong>{dayLabel(m.timestamp)}</strong>{" "}
                          {clock(m.timestamp)}
                        </time>
                      </p>
                    )}
                    {thread && i === 1 && (
                      <p className="bubble-divider thread-divider">
                        {threadCount} {threadCount === 1 ? "Reply" : "Replies"}
                      </p>
                    )}
                    <div
                      className={`bubble-row ${mine ? "mine" : "theirs"} ${start ? "start" : ""} ${end ? "end" : ""} ${group ? "in-group" : ""} ${highlight === m.relayId ? "found" : ""}`}
                    >
                      {!mine && group && (
                        <span className="bubble-gutter">
                          {end && <Avatar agent={m.sender} size={30} />}
                        </span>
                      )}
                      <div className="bubble-col">
                        {start && !mine && group && (
                          <span className="bubble-sender">
                            {agentNames[m.sender]}
                          </span>
                        )}
                        <div
                          className={`bubble-wrap ${tapbacks?.length ? "has-tapback" : ""}`}
                        >
                          <button
                            className={`bubble ${m.pending ? "pending" : ""}`}
                            aria-expanded={expanded}
                            aria-label={`${agentNames[m.sender]}${meta.length ? `, ${meta.join(", ")}` : ""}${m.experimentId ? `, ${m.experimentId}` : ""}: ${m.text}`}
                            onClick={() => setSelected(expanded ? null : m.key)}
                          >
                            {meta.length > 0 || m.experimentId ? (
                              <span className="bubble-meta" aria-hidden="true">
                                {[
                                  meta[0] && <span key="c">{meta[0]}</span>,
                                  m.experimentId && (
                                    <span key="e" className="mono">
                                      {m.experimentId}
                                    </span>
                                  ),
                                  meta[1] && <span key="r">{meta[1]}</span>,
                                ]
                                  .filter(Boolean)
                                  .map((part, n) => (
                                    <Fragment key={n}>
                                      {n > 0 && "\u00a0· "}
                                      {part}
                                    </Fragment>
                                  ))}
                              </span>
                            ) : null}
                            <span className="bubble-text" aria-hidden="true">
                              {m.text}
                            </span>
                            {end && <Tail />}
                          </button>
                          {tapbacks?.length ? (
                            <Tapback reactions={tapbacks} viewer={viewer} />
                          ) : null}
                        </div>
                      </div>
                    </div>
                    {(threadReplies.length > 0 || receipt || expanded) && (
                      <div
                        className={`bubble-after ${mine ? "mine" : "theirs"} ${group ? "in-group" : ""}`}
                      >
                        {threadReplies.length > 0 && (
                          <button
                            className="bubble-thread-link"
                            onClick={() => {
                              setThread(m.relayId!);
                              setSelected(null);
                              setDetails(false);
                              setFollow(true);
                            }}
                          >
                            {threadReplies.length}{" "}
                            {threadReplies.length === 1 ? "Reply" : "Replies"}
                          </button>
                        )}
                        {receipt && (
                          <span className="bubble-receipt">
                            <strong>{receipt.strong}</strong>
                            {receipt.rest && <> {receipt.rest}</>}
                          </span>
                        )}
                        {expanded && (
                          <dl className="bubble-evidence">
                            <dt>Sent</dt>
                            <dd>{utcTitle(m.timestamp)}</dd>
                            <dt>Relay message</dt>
                            <dd>{m.relayId ?? "Awaiting Relay receipt"}</dd>
                            <dt>Route</dt>
                            <dd>
                              {m.route === "dm"
                                ? `Direct message${m.receipt?.conversation_id ? `, ${m.receipt.conversation_id}` : ""}`
                                : m.route === "thread"
                                  ? "Thread reply in the research room"
                                  : `#${m.receipt?.room?.name ?? roomName}`}
                            </dd>
                            {m.receipt?.woke_by && (
                              <>
                                <dt>Recipient woken by</dt>
                                <dd>
                                  {m.receipt.woke_by === "push"
                                    ? "Relay push stream"
                                    : "Inbox sweep"}
                                </dd>
                              </>
                            )}
                            {m.receipt?.deliveries?.map((d) => (
                              <div key={d.role}>
                                <dt>{agentNames[d.role] ?? d.role}</dt>
                                <dd>
                                  {d.delivery_id
                                    ? `Inbox delivery ${d.delivery_id} acked, read ${d.read_at ? clock(d.read_at) : ""}`
                                    : `Read back from Relay ${d.read_at ? clock(d.read_at) : ""}`}
                                </dd>
                              </div>
                            ))}
                          </dl>
                        )}
                      </div>
                    )}
                  </li>
                );
              })}
            </ol>
          ) : loading ? (
            <div className="chat-skeleton" aria-label="Loading messages">
              {[
                "w-64 theirs",
                "w-48 theirs",
                "w-56 mine",
                "w-72 theirs",
                "w-40 mine",
              ].map((k) => (
                <span key={k} className={`chat-skeleton-bubble ${k}`} />
              ))}
            </div>
          ) : (
            <div className="chat-empty">
              <span className="convo-cluster" aria-hidden="true">
                {(active.kind === "room" ? cluster : cluster.slice(1, 4)).map(
                  ({ agent, size }) => (
                    <Avatar key={agent} agent={agent} size={size} />
                  ),
                )}
              </span>
              <h3>
                {!campaign
                  ? "No campaign selected"
                  : active.kind === "room"
                    ? "The research room is quiet"
                    : "No direct messages yet"}
              </h3>
              <p>
                {campaign
                  ? "Messages appear here once Agent Relay confirms the recipient read them. Handoffs between two agents arrive as direct messages."
                  : "Launch a campaign to bring the research team into the room."}
              </p>
            </div>
          )}
        </div>
        <footer className="composer">
          {!follow && !highlight && shown.length > 0 && (
            <button
              className="composer-latest"
              aria-label="Jump to the latest message"
              onClick={() => {
                setFollow(true);
                const el = scroller.current;
                if (el) el.scrollTop = el.scrollHeight;
              }}
            >
              <Icon name="down" size={18} />
            </button>
          )}
          <div className="composer-field">
            <span className="composer-placeholder">
              Only agents write here. Message Hephaestus to direct the team.
            </span>
            <button
              className="composer-send"
              disabled
              aria-label="Send. Only agents can send in this transcript."
            >
              <svg
                viewBox="0 0 24 24"
                width="16"
                height="16"
                aria-hidden="true"
              >
                <path d="M12 19V5M6 11l6-6 6 6" />
              </svg>
            </button>
          </div>
        </footer>
      </div>
    </section>
  );
}
