"use client";
import { useCallback, useEffect, useState } from "react";
import { Icon } from "./icon";
import { ConnectionsPanel } from "./overseer-connections";
import { api, OverseerSummary, Pack, QueueItem } from "@/lib/packs";
import { percent } from "@/lib/types";

const ms = (n: number | null) =>
  n === null
    ? "not logged"
    : n >= 1000
      ? `${(n / 1000).toFixed(1)}s`
      : `${Math.round(n)}ms`;

export function OverseerView({ onSent }: { onSent: () => void }) {
  const [packs, setPacks] = useState<Pack[]>([]),
    [packId, setPackId] = useState("gsm8k"),
    [summary, setSummary] = useState<OverseerSummary | null>(null),
    [queue, setQueue] = useState<QueueItem[]>([]),
    [answers, setAnswers] = useState<Record<string, string>>({}),
    [busy, setBusy] = useState(""),
    [notice, setNotice] = useState(""),
    [error, setError] = useState("");

  const load = useCallback(async () => {
    try {
      const [s, q] = await Promise.all([
        api<OverseerSummary>(`overseer/summary?task_pack=${packId}`),
        api<QueueItem[]>(`overseer/queue?task_pack=${packId}`),
      ]);
      setSummary(s);
      setQueue(q);
      setError("");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Overseer unavailable.");
    }
  }, [packId]);
  useEffect(() => {
    api<Pack[]>("packs")
      .then(setPacks)
      .catch(() => undefined);
  }, []);
  useEffect(() => {
    const first = setTimeout(load, 0);
    const timer = setInterval(load, 15000);
    return () => {
      clearTimeout(first);
      clearInterval(timer);
    };
  }, [load]);

  async function act(label: string, fn: () => Promise<unknown>) {
    setBusy(label);
    setError("");
    try {
      await fn();
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed.");
    } finally {
      setBusy("");
    }
  }

  const pack = packs.find((p) => p.id === packId);
  const judged = summary?.judged;
  return (
    <div className="production-view">
      <div className="table-tools">
        <select
          value={packId}
          onChange={(e) => setPackId(e.target.value)}
          aria-label="Task"
        >
          {(packs.length
            ? packs
            : [
                {
                  id: "gsm8k",
                  display: { name: "Mathematical reasoning" },
                } as Pack,
              ]
          ).map((p) => (
            <option key={p.id} value={p.id}>
              {p.display.name}
            </option>
          ))}
        </select>
        <button
          className="button small-button"
          disabled={!!busy}
          onClick={() => act("Checking…", () => api("overseer/cycle", {}))}
        >
          <Icon name="refresh" size={14} />
          {busy === "Checking…" ? "Checking…" : "Check now"}
        </button>
      </div>
      {error && <p className="error-banner">{error}</p>}
      <ConnectionsPanel packId={packId} />
      {summary && (
        <>
          <section
            className="production-summary"
            aria-label="Production health"
          >
            <div>
              <span>Traces watched</span>
              <strong>{summary.traces}</strong>
              <p>
                {summary.sealed_overlap
                  ? `${summary.sealed_overlap} sealed-test prompts kept as flags only`
                  : "Free checks run on every trace"}
              </p>
            </div>
            <div>
              <span>Flagged</span>
              <strong className={summary.flagged ? "accent-text" : ""}>
                {summary.flagged}
              </strong>
              <p>
                {summary.traces
                  ? `${percent(summary.flagged / Math.max(1, summary.traces - summary.sealed_overlap))} of live traces`
                  : "No traffic yet"}
              </p>
            </div>
            <div>
              <span>Judged failure rate</span>
              <strong>
                {judged?.sample_failure_rate === null || !judged
                  ? "Not judged"
                  : percent(judged.sample_failure_rate)}
              </strong>
              <p>
                {judged?.random_sample
                  ? `Random sample of ${judged.random_sample}, 95% ${percent(judged.sample_ci95?.[0])} to ${percent(judged.sample_ci95?.[1])}`
                  : summary.judge.ready
                    ? "No sampled traces judged yet"
                    : (summary.judge.reason ?? "Judge off")}
              </p>
            </div>
            <div>
              <span>Latency p95</span>
              <strong>
                {summary.latency_ms.p95 !== null
                  ? ms(summary.latency_ms.p95)
                  : summary.traces
                    ? "Not logged"
                    : "No traffic"}
              </strong>
              <p>
                p50 {ms(summary.latency_ms.p50)} · judge spend $
                {summary.spend_today.usd.toFixed(2)} of $
                {summary.budget.daily_usd.toFixed(2)} today
              </p>
            </div>
          </section>
          {summary.alerts.length > 0 && (
            <section className="panel overseer-panel" aria-label="Alerts">
              <h3>Alerts</h3>
              <ul className="overseer-alerts">
                {summary.alerts.map((a) => {
                  const d = JSON.parse(a.detail);
                  return (
                    <li key={a.id}>
                      <Icon name="alert" size={14} />
                      <strong>{a.kind.replace(/_/g, " ")}</strong>
                      <span className="muted small">
                        {"connection" in d
                          ? `${d.connection}: ${d.reason}`
                          : "today" in d
                            ? `${percent(d.today)} today vs ${percent(d.baseline)} over the previous week (${d.n} traces)`
                            : "model_version" in d
                              ? `first traffic from ${d.model_version}`
                              : `p95 ${ms(d.today_ms)} vs ${ms(d.baseline_ms)}`}
                      </span>
                    </li>
                  );
                })}
              </ul>
            </section>
          )}
          {summary.signals.length > 0 && (
            <section className="panel overseer-panel" aria-label="Signals">
              <h3>What the checks found</h3>
              <ul className="overseer-signals">
                {summary.signals.map((s) => (
                  <li key={s.kind}>
                    <span>{s.label}</span>
                    <span className="overseer-bar" aria-hidden>
                      <span
                        style={{ width: `${Math.min(100, s.rate * 100)}%` }}
                      />
                    </span>
                    <span className="mono small">
                      {s.count} · {percent(s.rate)}
                    </span>
                  </li>
                ))}
              </ul>
            </section>
          )}
        </>
      )}
      <section
        className="panel production-findings overseer-panel"
        aria-label="Review queue"
      >
        <header className="overseer-queue-head">
          <div>
            <h3>Review queue</h3>
            <p className="muted small">
              Flagged and sampled answers. Only your label becomes a training
              answer; user corrections and judge notes are shown for context
              only.
            </p>
          </div>
          <button
            className="button primary"
            disabled={!!busy || !summary?.labels}
            onClick={() =>
              act("Sending…", async () => {
                const scan = await api<{ id: string; flagged: number }>(
                  "overseer/export",
                  {
                    task_pack: packId,
                    name: `Overseer · ${pack?.display.name ?? packId}`,
                  },
                );
                setNotice(
                  `Sent as production scan ${scan.id}. Open Production failures to train on it.`,
                );
                onSent();
              })
            }
          >
            Send {summary?.labels ?? 0} labeled to the lab
            <Icon name="arrow" size={16} />
          </button>
        </header>
        {notice && <p className="production-note">{notice}</p>}
        {!queue.length && <p className="muted">Nothing waiting for review.</p>}
        <ul className="finding-list">
          {queue.map((item) => (
            <li key={item.key} className="finding">
              <div className="finding-tags">
                {item.signals.map((s) => (
                  <span key={s.kind} className="tag">
                    {s.kind.replace(/_/g, " ")}
                  </span>
                ))}
                {item.sampled && <span className="tag">random sample</span>}
                {item.judgment && (
                  <span className="tag">judge: {item.judgment.verdict}</span>
                )}
                <span className="muted small mono">
                  {item.model_version ?? "unknown model"} ·{" "}
                  {ms(item.latency_ms)}
                </span>
              </div>
              <p className="finding-prompt">{item.trace.prompt}</p>
              <pre className="prediction-output finding-output">
                {item.trace.output}
              </pre>
              {item.judgment?.rationale && (
                <p className="muted small">
                  Judge note: {item.judgment.rationale}
                </p>
              )}
              {item.trace.feedback?.correction && (
                <p className="finding-correction">
                  User correction (not used as a label):{" "}
                  {item.trace.feedback.correction}
                </p>
              )}
              <div className="field-row">
                <input
                  aria-label="Correct answer"
                  placeholder={
                    pack?.id === "gsm8k"
                      ? "Correct answer, ending with #### <number>"
                      : "Correct answer"
                  }
                  value={answers[item.key] ?? ""}
                  onChange={(e) =>
                    setAnswers((a) => ({ ...a, [item.key]: e.target.value }))
                  }
                />
                <button
                  className="button small-button"
                  disabled={!!busy || !answers[item.key]}
                  onClick={() =>
                    act("Saving…", () =>
                      api("overseer/labels", {
                        key: item.key,
                        answer: answers[item.key],
                      }),
                    )
                  }
                >
                  Save label
                </button>
              </div>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
