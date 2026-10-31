"use client";
import { FormEvent, useCallback, useEffect, useRef, useState } from "react";
import { Icon } from "./icon";
import { api } from "@/lib/packs";

type Line = {
  id: string;
  sender: "you" | "hephaestus";
  text: string;
  at: number;
};
type Desk = {
  messages: Line[];
  thinking: boolean;
  activity: string | null;
  model: {
    connected: boolean;
    provider: string | null;
    name: string | null;
    tokens_today: number;
  };
  mission: { goal: string; wakes: number; max_wakes: number } | null;
  waiting: string[];
};

/** Codes it asked the owner about that haven't been answered, newest message first. */
function openCode(messages: Line[], index: number) {
  const found = /Reply "confirm ([0-9a-f]{4})"/.exec(messages[index].text);
  if (!found || messages[index].sender !== "hephaestus") return null;
  const answered = messages
    .slice(index + 1)
    .some(
      (m) =>
        m.sender === "you" &&
        new RegExp(`^(confirm|decline) ${found[1]}$`, "i").test(m.text.trim()),
    );
  return answered ? null : found[1];
}

/** The private conversation in Agent chat: just the lab's owner and Hephaestus. */
export function HephaestusChat({ onBack }: { onBack: () => void }) {
  const [desk, setDesk] = useState<Desk | null>(null),
    [draft, setDraft] = useState(""),
    [error, setError] = useState(""),
    [away, setAway] = useState("");
  const scroller = useRef<HTMLDivElement>(null);

  const load = useCallback(async () => {
    try {
      setDesk(await api<Desk>("hephaestus/messages"));
      setAway("");
    } catch (e) {
      setAway(e instanceof Error ? e.message : "Hephaestus is unavailable.");
    }
  }, []);
  useEffect(() => {
    const first = setTimeout(load, 0);
    const timer = setInterval(load, 2000);
    return () => {
      clearTimeout(first);
      clearInterval(timer);
    };
  }, [load]);

  const count = desk?.messages.length ?? 0;
  const thinking = desk?.thinking ?? false;
  useEffect(() => {
    const el = scroller.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [count, thinking]);

  async function say(text: string) {
    setError("");
    try {
      await api("hephaestus/messages", { text });
      await load();
    } catch (failure) {
      setDraft(text);
      setError(failure instanceof Error ? failure.message : "Not sent.");
    }
  }

  function send(e: FormEvent) {
    e.preventDefault();
    const text = draft.trim();
    if (!text) return;
    setDraft("");
    void say(text);
  }

  return (
    <div className="convo-pane">
      <header className="convo-head">
        <div className="convo-head-start">
          <button
            className="convo-back"
            aria-label="Back to conversations"
            onClick={onBack}
          >
            <Icon name="chevron" size={22} />
          </button>
        </div>
        <div className="convo-head-center">
          <span className="avatar-hephaestus" aria-hidden="true">
            HE
          </span>
          <h3>Hephaestus</h3>
          <p className="convo-head-sub">
            {desk?.model.name
              ? `Just you · ${desk.model.name} · ${desk.model.tokens_today.toLocaleString()} tokens today`
              : "Just you · admin of the agent chat"}
          </p>
        </div>
        <div className="convo-head-end" />
      </header>
      {desk && !desk.model.connected && (
        <p className="hephaestus-note">
          No model connected, so it only understands commands. Say help to see
          them.
        </p>
      )}
      {desk?.mission && (
        <p className="hephaestus-note">
          On a mission ({desk.mission.wakes}/{desk.mission.max_wakes} wake-ups):{" "}
          {desk.mission.goal}
        </p>
      )}
      {away && <p className="hephaestus-note">{away}</p>}
      <div className="convo-scroll" ref={scroller} tabIndex={0}>
        <div className="bubble-list" aria-live="polite">
          {desk?.messages.map((m, index) => {
            const code = openCode(desk.messages, index);
            return (
              <div
                key={m.id}
                className={`bubble-row start end ${m.sender === "you" ? "mine" : ""}`}
              >
                <div className="bubble-col">
                  <div className="bubble-wrap">
                    <div className="bubble">
                      <span className="bubble-text">{m.text}</span>
                    </div>
                  </div>
                  {code && desk.waiting.length > 0 && (
                    <div className="hephaestus-approval">
                      <button
                        className="button primary"
                        onClick={() => void say(`confirm ${code}`)}
                      >
                        Approve
                      </button>
                      <button
                        className="button"
                        onClick={() => void say(`decline ${code}`)}
                      >
                        Decline
                      </button>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
          {thinking && (
            <div className="bubble-row start end">
              <div className="bubble-col">
                <div className="bubble-wrap">
                  <div className="bubble pending">
                    <span className="bubble-text">
                      {desk?.activity
                        ? `Working on it: ${desk.activity.replaceAll("_", " ")}…`
                        : "Working on it…"}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
        {desk && !count && !thinking && (
          <p className="hephaestus-note">
            Tell it what to train, or say help. It commands the research team:
            their orders, models and next experiment, and when they hold or
            finish. It can keep working on a goal between your messages. It
            can&apos;t override governance, budgets or test data.
          </p>
        )}
      </div>
      <footer className="composer">
        {error && <p className="hephaestus-note">{error}</p>}
        <form className="composer-field" onSubmit={send}>
          <input
            className="composer-input"
            aria-label="Message Hephaestus"
            placeholder="Message Hephaestus"
            maxLength={2000}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
          />
          <button
            className="composer-send"
            disabled={!draft.trim()}
            aria-label="Send"
          >
            <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
              <path d="M12 19V5M6 11l6-6 6 6" />
            </svg>
          </button>
        </form>
      </footer>
    </div>
  );
}
