"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { Icon } from "./icon";

type Owner = { epoch: number; state_version: number } | null;
type ActorProof = {
  actor: string;
  id: string;
  before: string;
  after: string | null;
  identical: boolean;
  owner_before: Owner;
  owner_after: Owner;
};
type Incident = {
  id: string;
  campaign_id: string;
  phase: "killed" | "fenced" | "restored" | "failed";
  kill: { pid: number; signal: string; signum: number; at: number };
  restart?: {
    pid: number;
    old_pid: number;
    signal: string | null;
    listening_at: number;
    restart_seconds: number;
  };
  restored_at?: number;
  downtime_seconds?: number;
  process_seconds?: number;
  fence_seconds?: number;
  error?: string;
  campaign_status?: string | null;
  resumed?: { at: number; text: string } | null;
  retries: { method: string; actor: string; attempts: number }[];
  proof?: {
    actors: ActorProof[];
    identical: number;
    total: number;
    hash_before: string;
    hash_after: string;
    budget: {
      consumed_after: number;
      settled_sum: number;
      double_counted: boolean;
    };
    leases: { after: number; held: number; unchanged: boolean };
    governance: {
      reviews_before: number;
      reviews_after: number;
      retained: boolean;
    };
  };
};
type ChaosStatus = {
  enabled: boolean;
  supervised: boolean;
  runtime: { pid: number; restarts: number } | null;
  incidents: Incident[];
};

const seconds = (value: number) => `${value.toFixed(1)}s`;
const clock = (unix: number) =>
  `${new Date(unix * 1000).toLocaleTimeString("en-GB", { hour12: false, timeZone: "UTC" })} UTC`;
const short = (hash: string | null | undefined) =>
  hash ? hash.slice(0, 8) : "";
const restartCount = (n: number) =>
  n === 0
    ? "No restarts yet."
    : n === 1
      ? "Restarted once."
      : `Restarted ${n} times.`;
const ACTIVE = new Set(["killed", "fenced"]);

export function CrashTest({
  campaignId,
  onRecovered,
}: {
  campaignId: string;
  onRecovered: () => void;
}) {
  const [status, setStatus] = useState<ChaosStatus | null>(null);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");
  const [now, setNow] = useState(() => Date.now() / 1000);
  const phase = useRef<string | undefined>(undefined);
  const latest = status?.incidents[0];
  const active = Boolean(latest && ACTIVE.has(latest.phase));

  const read = useCallback(async () => {
    try {
      const response = await fetch("/api/lab/chaos/actors", {
        cache: "no-store",
      });
      if (!response.ok) return;
      const data: ChaosStatus = await response.json();
      setStatus(data);
      setNow(Date.now() / 1000);
      const current = data.incidents[0]?.phase;
      if (
        phase.current &&
        phase.current !== "restored" &&
        current === "restored"
      )
        onRecovered();
      phase.current = current;
    } catch {
      // The API itself is up during a runtime crash; a failed poll is retried next tick.
    }
  }, [onRecovered]);

  useEffect(() => {
    const first = setTimeout(() => void read(), 0);
    const timer = setInterval(() => void read(), active ? 250 : 1000);
    return () => {
      clearTimeout(first);
      clearInterval(timer);
    };
  }, [read, active]);

  async function kill() {
    setSending(true);
    setError("");
    try {
      const response = await fetch("/api/lab/chaos/actors/kill", {
        method: "POST",
        body: JSON.stringify({ campaign_id: campaignId }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "The kill was refused.");
      setStatus(data);
      phase.current = data.incidents[0]?.phase;
    } catch (e) {
      setError(e instanceof Error ? e.message : "The kill was refused.");
    } finally {
      setSending(false);
    }
  }

  if (!status || (!status.enabled && !status.incidents.length)) return null;
  const incident = latest?.campaign_id === campaignId ? latest : undefined;
  const ready = status.enabled && status.supervised && !active;
  const reason = active
    ? "Recovery in progress. The supervisor restarts the runtime on the same data directory."
    : !status.enabled
      ? "Crash tests are off. Start the lab with FORGE_CHAOS=1."
      : !status.supervised
        ? "The runtime is not supervised by this lab, so nothing would restart it."
        : `Runtime pid ${status.runtime?.pid}, supervised. ${restartCount(status.runtime?.restarts ?? 0)}`;

  return (
    <section className="crash-panel" aria-labelledby="crash-title">
      <div className="crash-head">
        <div>
          <span className="eyebrow">Crash test</span>
          <h2 id="crash-title">
            Kill the runtime in the middle of the campaign.
          </h2>
          <p>
            Sends SIGKILL to the Terse process group. The lab supervisor
            restarts it on the same data directory, and the worker retries each
            interrupted call under its original request ID, so nothing runs
            twice.
          </p>
        </div>
        <div className="crash-action">
          <button
            className="button danger crash-button"
            onClick={() => void kill()}
            disabled={!ready || sending}
          >
            <Icon name="alert" size={17} />
            {sending ? "Sending SIGKILL" : "Kill the actor runtime"}
          </button>
          <small>{reason}</small>
        </div>
      </div>
      {error && (
        <p className="crash-error" role="alert">
          {error}
        </p>
      )}
      {incident ? <Timeline incident={incident} now={now} /> : null}
    </section>
  );
}

function Step({
  done,
  tone,
  label,
  figure,
  note,
}: {
  done: boolean;
  tone?: "danger";
  label: string;
  figure: string;
  note: React.ReactNode;
}) {
  return (
    <li className={`crash-step${done ? " done" : ""}${tone ? ` ${tone}` : ""}`}>
      <span className="crash-label">
        {done && <Icon name={tone ? "alert" : "check"} size={14} />}
        {label}
      </span>
      <strong className="crash-figure">{figure}</strong>
      <span className="crash-note">{note}</span>
    </li>
  );
}

function Timeline({ incident, now }: { incident: Incident; now: number }) {
  const { kill, restart, proof } = incident;
  const restored = incident.phase === "restored";
  const elapsed = Math.max(0, now - kill.at);
  const resumed = incident.resumed;
  const completed = incident.campaign_status === "completed";
  const identical = proof && proof.identical === proof.total;
  return (
    <>
      <ol className="crash-steps" aria-live="polite">
        <Step
          done
          tone="danger"
          label={kill.signal}
          figure={`pid ${kill.pid}`}
          note={`Signal ${kill.signum} to the process group at ${clock(kill.at)}`}
        />
        <Step
          done={restored}
          label="Runtime back in"
          figure={
            restored
              ? seconds(incident.downtime_seconds ?? 0)
              : seconds(elapsed)
          }
          note={
            restored && restart
              ? `Restarted as pid ${restart.pid} in ${seconds(incident.process_seconds ?? 0)}. Terse fenced writes for ${seconds(incident.fence_seconds ?? 0)} until the killed owner's lease expired.`
              : restart
                ? `Restarted as pid ${restart.pid}. Waiting for the killed owner's lease to expire.`
                : incident.phase === "failed"
                  ? incident.error
                  : "Waiting for the supervisor to restart the process."
          }
        />
        <Step
          done={Boolean(restored && identical)}
          label="Actors restored"
          figure={proof ? `${proof.identical} of ${proof.total}` : "Pending"}
          note={
            proof ? (
              identical ? (
                <>
                  State identical ·{" "}
                  <span className="mono">{short(proof.hash_after)}</span>
                </>
              ) : (
                `${proof.total - proof.identical} advanced by the resumed campaign`
              )
            ) : (
              "Compared after the first successful read."
            )
          }
        />
        <Step
          done={Boolean(resumed)}
          label="Campaign resumed"
          figure={
            resumed && incident.restored_at
              ? `+${seconds(Math.max(0, resumed.at - incident.restored_at))}`
              : "Pending"
          }
          note={resumed ? resumed.text : "Worker calls are retrying."}
        />
        <Step
          done={completed}
          label="Campaign"
          figure={
            incident.campaign_status
              ? incident.campaign_status[0].toUpperCase() +
                incident.campaign_status.slice(1)
              : "Pending"
          }
          note={
            completed
              ? "Completed normally after the crash."
              : "Running on the restored state."
          }
        />
      </ol>
      {proof && <Proof proof={proof} />}
    </>
  );
}

function Proof({ proof }: { proof: NonNullable<Incident["proof"]> }) {
  const { budget, leases, governance } = proof;
  return (
    <div className="crash-proof">
      <div className="crash-facts">
        <div>
          <span>Budget</span>
          <strong>{seconds(budget.consumed_after)} settled</strong>
          <small>
            {budget.double_counted
              ? "Does not match the sum of settled leases"
              : `Equals the sum of settled leases, ${seconds(budget.settled_sum)}. No double count.`}
          </small>
        </div>
        <div>
          <span>Leases</span>
          <strong>
            {leases.after} recorded, {leases.held} held
          </strong>
          <small>
            {leases.unchanged
              ? "Every lease from before the kill is byte for byte unchanged."
              : "A lease changed across the kill."}
          </small>
        </div>
        <div>
          <span>Governance</span>
          <strong>
            {governance.reviews_after} of {governance.reviews_before} decisions
          </strong>
          <small>
            {governance.retained
              ? "All review decisions retained with their evidence."
              : "A review decision changed."}
          </small>
        </div>
      </div>
      <div
        className="crash-table"
        role="table"
        aria-label="Actor state before and after the kill"
      >
        <div role="row" className="crash-row crash-row-head">
          <span role="columnheader">Actor</span>
          <span role="columnheader">Committed version</span>
          <span role="columnheader">Owner epoch</span>
          <span role="columnheader">SHA-256 before</span>
          <span role="columnheader">SHA-256 after</span>
          <span role="columnheader">Result</span>
        </div>
        {proof.actors.map((a) => (
          <div role="row" className="crash-row" key={`${a.actor}/${a.id}`}>
            <span role="cell" className="mono crash-actor">
              {a.actor}
            </span>
            <span role="cell" className="mono" data-label="Committed version">
              {a.owner_before && a.owner_after
                ? `${a.owner_before.state_version} → ${a.owner_after.state_version}`
                : "Not recorded"}
            </span>
            <span role="cell" className="mono" data-label="Owner epoch">
              {a.owner_before && a.owner_after
                ? `${a.owner_before.epoch} → ${a.owner_after.epoch}`
                : "Not recorded"}
            </span>
            <span role="cell" className="mono" data-label="Before">
              {short(a.before)}
            </span>
            <span role="cell" className="mono" data-label="After">
              {short(a.after)}
            </span>
            <span
              role="cell"
              className={a.identical ? "crash-result same" : "crash-result"}
              data-label="Result"
            >
              {a.identical ? "Identical" : "Advanced"}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
