"use client";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  CampaignEvidence,
  CheckState,
  Claim,
  Count,
  agentNames,
  duration,
  modelLabel,
} from "@/lib/types";
import { Icon } from "./icon";
import { Empty, PanelHeader } from "./views";

const CLAIM: Record<Claim, { label: string; tone: string }> = {
  validated: { label: "Validated", tone: "pass" },
  uncertain: { label: "Uncertain", tone: "warn" },
  none: { label: "No improvement", tone: "neutral" },
};
const STATE: Record<CheckState, { label: string; icon: string }> = {
  pass: { label: "Passed", icon: "check" },
  fail: { label: "Failed", icon: "close" },
  partial: { label: "Partial", icon: "alert" },
  not_recorded: { label: "Not recorded", icon: "info" },
  not_applicable: { label: "Does not apply", icon: "info" },
};
// What each check proves, so a reader can judge it without opening the code.
const MEANING: Record<string, string> = {
  sealed_once:
    "The test questions are scored once per model. Agents cannot iterate against them.",
  selection_frozen:
    "The winning checkpoint is chosen on development data and locked before the test runs.",
  reload:
    "Scores come from weights loaded off disk into a fresh model, not the training process.",
  sha256:
    "Every saved file is hashed again now and compared with the hash recorded when it was written.",
  governance:
    "A separate council approves or rejects each gate, with machine-readable reasons.",
  no_sealed_access:
    "Governance records whether any training step touched sealed data.",
  pack_pinned:
    "Every gate evaluates the same task pack, identified by its SHA-256.",
};

const signed = (pp: number) =>
  `${pp > 0 ? "+" : pp < 0 ? "−" : ""}${Math.abs(pp).toFixed(1)}`;
const count = (c: Count | null | undefined) =>
  c ? `${c.correct} of ${c.n}` : "Not recorded";
const number = (n: number) => n.toLocaleString("en-US");
const megabytes = (b: number) => `${(b / 1e6).toFixed(1)} MB`;
const when = (iso: string | null) =>
  iso
    ? new Date(iso).toLocaleString("en-US", {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      })
    : "";
const hardware = (c: CampaignEvidence) =>
  c.hardware?.name
    ? c.hardware.backend === "modal"
      ? `${c.hardware.name} on Modal`
      : c.hardware.name
    : "hardware not recorded";

function ClaimTag({ claim }: { claim: Claim | null | undefined }) {
  if (!claim) return <span className="ev-claim muted">No claim recorded</span>;
  const c = CLAIM[claim];
  return <span className={`ev-claim ${c.tone}`}>{c.label}</span>;
}

/** One shared axis for every row, so intervals compare at a glance. */
function useScale(rows: CampaignEvidence[]) {
  return useMemo(() => {
    const values = rows.flatMap((r) => [
      r.sealed?.improvement_pp ?? 0,
      ...(r.sealed?.interval_pp ?? []),
    ]);
    const lo = Math.floor(Math.min(0, ...values) / 10) * 10;
    const hi = Math.max(10, Math.ceil(Math.max(0, ...values) / 10) * 10);
    // Ticks are multiples of the step, so zero is always labeled.
    const step = hi - lo > 60 ? 20 : 10,
      ticks = [];
    for (let t = Math.ceil(lo / step) * step; t <= hi; t += step) ticks.push(t);
    return { lo, hi, ticks, x: (v: number) => ((v - lo) / (hi - lo)) * 100 };
  }, [rows]);
}

function Interval({
  pp,
  interval,
  scale,
}: {
  pp: number | null;
  interval: [number, number] | null;
  scale: ReturnType<typeof useScale>;
}) {
  if (pp === null) return null;
  const label = interval
    ? `Change ${signed(pp)} points, 95% paired interval ${signed(interval[0])} to ${signed(interval[1])}`
    : `Change ${signed(pp)} points, no interval recorded`;
  return (
    <svg
      className="ev-plot"
      viewBox="0 0 100 16"
      preserveAspectRatio="none"
      role="img"
      aria-label={label}
    >
      <line
        className="ev-zero"
        x1={scale.x(0)}
        x2={scale.x(0)}
        y1="0"
        y2="16"
      />
      {interval && (
        <line
          className="ev-range"
          x1={scale.x(interval[0])}
          x2={scale.x(interval[1])}
          y1="8"
          y2="8"
        />
      )}
      <line
        className={interval ? "ev-point" : "ev-point hollow"}
        x1={scale.x(pp)}
        x2={scale.x(pp)}
        y1="3"
        y2="13"
      />
    </svg>
  );
}

function Ledger({ rows }: { rows: CampaignEvidence[] }) {
  const finals = rows.filter((r) => r.sealed);
  const by = (claim: Claim | null) =>
    finals.filter((r) => (r.sealed?.claim ?? null) === claim).length;
  const files = rows.reduce(
    (a, r) => ({
      count: a.count + r.artifacts.count,
      verified: a.verified + r.artifacts.verified,
      mismatched: a.mismatched + r.artifacts.mismatched,
    }),
    { count: 0, verified: 0, mismatched: 0 },
  );
  const relay = rows.reduce(
    (a, r) => ({
      sent: a.sent + r.relay.sent,
      delivered: a.delivered + r.relay.delivered,
    }),
    { sent: 0, delivered: 0 },
  );
  const withCalls = rows.filter((r) => r.agents);
  const tokens = withCalls.reduce(
    (a, r) => a + r.agents!.input_tokens + r.agents!.output_tokens,
    0,
  );
  const legacy = by(null);
  return (
    <section className="ev-ledger" aria-label="Summary of recorded evidence">
      <div>
        <span>Sealed finals</span>
        <strong className="mono">{finals.length}</strong>
        <p>
          {by("validated")} validated, {by("uncertain")} uncertain, {by("none")}{" "}
          without improvement
          {legacy ? `, ${legacy} recorded before claims existed` : ""}. Out of{" "}
          {rows.length} campaigns.
        </p>
      </div>
      <div>
        <span>Artifacts re-hashed</span>
        <strong className="mono">
          {number(files.verified)}
          <small> of {number(files.count)}</small>
        </strong>
        <p>
          {files.mismatched
            ? `${files.mismatched} differ from their recorded SHA-256.`
            : "Every file present matches its recorded SHA-256."}
        </p>
      </div>
      <div>
        <span>Relay messages delivered</span>
        <strong className="mono">
          {number(relay.delivered)}
          <small> of {number(relay.sent)}</small>
        </strong>
        <p>Counted only with a delivery receipt from Agent Relay.</p>
      </div>
      <div>
        <span>Agent tokens</span>
        <strong className="mono">{number(tokens)}</strong>
        <p>
          Input and output, from the {withCalls.length} campaigns that record
          model usage.
        </p>
      </div>
    </section>
  );
}

function Results({
  rows,
  selected,
  onSelect,
}: {
  rows: CampaignEvidence[];
  selected: string;
  onSelect: (id: string) => void;
}) {
  const scale = useScale(rows);
  return (
    <section className="panel">
      <PanelHeader
        title="Sealed test results"
        icon="results"
        subtitle="Original model and selected checkpoint, each scored once on the same sealed questions. A claim is validated only when the whole 95% paired bootstrap interval is above zero."
      />
      <div className="table-scroll">
        <table className="ev-table">
          <thead>
            <tr>
              <th scope="col">Campaign</th>
              <th scope="col">Sealed test</th>
              <th scope="col" className="ev-axis-head">
                <span className="sr-only">Change in points with interval</span>
                <span className="ev-axis" aria-hidden="true">
                  {scale.ticks.map((t) => (
                    <i key={t} style={{ left: `${scale.x(t)}%` }}>
                      {t === 0 ? "0" : signed(t).replace(".0", "")}
                    </i>
                  ))}
                </span>
              </th>
              <th scope="col">Claim</th>
              <th scope="col">Production held-out</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const s = r.sealed!;
              return (
                <tr
                  key={r.id}
                  className={r.id === selected ? "selected" : undefined}
                  onClick={() => onSelect(r.id)}
                >
                  <td>
                    <button
                      className="ev-row-button"
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelect(r.id);
                      }}
                      aria-pressed={r.id === selected}
                    >
                      <strong>{r.name || r.id}</strong>
                      <small>
                        {modelLabel(r.model_id ?? undefined)} · {hardware(r)}
                        <br />
                        <span className="mono">{r.id}</span> ·{" "}
                        {when(r.created_at)}
                      </small>
                    </button>
                  </td>
                  <td>
                    <span className="mono ev-counts">
                      {count(s.baseline)}
                      <Icon name="arrow" size={12} />
                      {count(s.checkpoint)}
                    </span>
                    {s.kept_original && <small>Kept the original model</small>}
                  </td>
                  <td className="ev-change">
                    <span className="mono">
                      {s.improvement_pp === null
                        ? "Not recorded"
                        : `${signed(s.improvement_pp)} pts`}
                    </span>
                    <Interval
                      pp={s.improvement_pp}
                      interval={s.interval_pp}
                      scale={scale}
                    />
                    <small className="mono">
                      {s.interval_pp
                        ? `95% CI ${signed(s.interval_pp[0])} to ${signed(s.interval_pp[1])}`
                        : "No interval recorded"}
                    </small>
                  </td>
                  <td>
                    <ClaimTag claim={s.claim} />
                    {s.improved !== null && s.regressed !== null && (
                      <small>
                        {s.improved} fixed, {s.regressed} broken
                      </small>
                    )}
                  </td>
                  <td>
                    {r.production ? (
                      <>
                        <span className="mono ev-counts">
                          {count(r.production.baseline)}
                          <Icon name="arrow" size={12} />
                          {count(r.production.checkpoint)}
                        </span>
                        <ClaimTag claim={r.production.claim} />
                      </>
                    ) : (
                      <span className="ev-faint">
                        Not part of this campaign
                      </span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Checks({
  campaign,
  all,
}: {
  campaign: CampaignEvidence;
  all: CampaignEvidence[];
}) {
  return (
    <section className="panel">
      <PanelHeader
        title="Why the result can be trusted"
        icon="governance"
        subtitle={`Checks for ${campaign.id}, derived from its saved state, events and files. The right column counts the same check across every campaign where it applies.`}
      />
      <ol className="ev-checks">
        {campaign.checks.map((c) => {
          const applicable = all
            .map((r) => r.checks.find((x) => x.id === c.id))
            .filter((x) => x && x.state !== "not_applicable");
          const passed = applicable.filter((x) => x!.state === "pass").length;
          return (
            <li key={c.id} className={`ev-check ${c.state}`}>
              <span className="ev-check-icon" aria-hidden="true">
                <Icon name={STATE[c.state].icon} size={14} />
              </span>
              <div>
                <strong>{c.label}</strong>
                <p>{c.detail}</p>
                <p className="ev-faint">{MEANING[c.id]}</p>
              </div>
              <div className="ev-check-state">
                <span>{STATE[c.state].label}</span>
                <small className="mono">
                  {passed} of {applicable.length} campaigns
                </small>
              </div>
            </li>
          );
        })}
      </ol>
    </section>
  );
}

function Observability({ campaign: c }: { campaign: CampaignEvidence }) {
  const facts: [string, string][] = [
    ["Hardware", c.hardware?.name ? hardware(c) : "Not recorded"],
    [
      "Wall time",
      c.elapsed_seconds === null ? "Not recorded" : duration(c.elapsed_seconds),
    ],
    [
      "Compute time",
      c.compute_seconds === null ? "Not recorded" : duration(c.compute_seconds),
    ],
    ["Experiments", String(c.experiments)],
    [
      "Relay messages",
      `${c.relay.delivered} of ${c.relay.sent} delivered, ${c.relay.readback_verified} read back by the recipient`,
    ],
    [
      "Terse state",
      c.terse?.state_version
        ? `Version ${c.terse.state_version}, ${c.terse.decisions} saved decisions`
        : "Not recorded",
    ],
    [
      "Artifacts",
      `${c.artifacts.verified} of ${c.artifacts.count} verified, ${megabytes(c.artifacts.bytes)}`,
    ],
    ["Director", c.director || "Not recorded"],
    ...(c.vms
      ? ([
          [
            "Freestyle VMs",
            `${c.vms.count} used for ${c.vms.runs} runs over ${duration(c.vms.seconds)}, ${c.vms.deleted} of ${c.vms.count} deleted`,
          ],
        ] as [string, string][])
      : []),
  ];
  const vmColumn = !!c.agents?.vm_runs;
  return (
    <section className="panel">
      <PanelHeader
        title="What ran"
        icon="chart"
        subtitle="Compute, coordination and model usage recorded for this campaign."
      />
      <dl className="ev-facts">
        {facts.map(([k, v]) => (
          <div key={k}>
            <dt>{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
      {c.agents ? (
        <div className="table-scroll">
          <table className="ev-agents">
            <thead>
              <tr>
                <th scope="col">Role</th>
                <th scope="col" className="numeric">
                  Calls
                </th>
                <th scope="col" className="numeric">
                  Subagents
                </th>
                {vmColumn && (
                  <th scope="col" className="numeric">
                    VM runs
                  </th>
                )}
                <th scope="col" className="numeric">
                  Tokens in
                </th>
                <th scope="col" className="numeric">
                  Tokens out
                </th>
              </tr>
            </thead>
            <tbody>
              {c.agents.by_role.map((r) => (
                <tr key={r.role}>
                  <td>
                    {agentNames[r.role as keyof typeof agentNames] ?? r.role}
                    <small>{r.model}</small>
                  </td>
                  <td className="numeric mono">{r.calls}</td>
                  <td className="numeric mono">{r.subagents}</td>
                  {vmColumn && (
                    <td className="numeric mono">{r.vm_runs ?? 0}</td>
                  )}
                  <td className="numeric mono">{number(r.input_tokens)}</td>
                  <td className="numeric mono">{number(r.output_tokens)}</td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr>
                <td>Total</td>
                <td className="numeric mono">{c.agents.calls}</td>
                <td className="numeric mono">{c.agents.subagent_runs}</td>
                {vmColumn && (
                  <td className="numeric mono">{c.agents.vm_runs}</td>
                )}
                <td className="numeric mono">
                  {number(c.agents.input_tokens)}
                </td>
                <td className="numeric mono">
                  {number(c.agents.output_tokens)}
                </td>
              </tr>
            </tfoot>
          </table>
        </div>
      ) : (
        <p className="ev-note">
          This campaign predates per-role usage recording, so model calls and
          tokens were not saved.
        </p>
      )}
    </section>
  );
}

export function EvidenceView({ onOpen }: { onOpen: (id: string) => void }) {
  const [rows, setRows] = useState<CampaignEvidence[] | null>(null),
    [error, setError] = useState(""),
    [selected, setSelected] = useState("");
  const load = useCallback(async () => {
    setError("");
    try {
      const r = await fetch("/api/lab/evidence", { cache: "no-store" });
      const data = await r.json();
      if (!r.ok || !Array.isArray(data))
        throw new Error(
          typeof data?.detail === "string"
            ? data.detail
            : "The evidence could not be loaded.",
        );
      setRows(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);
  useEffect(() => {
    // Initial load from the API; state updates land after the fetch resolves.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void load();
  }, [load]);

  const finals = useMemo(() => (rows ?? []).filter((r) => r.sealed), [rows]);
  const others = useMemo(() => (rows ?? []).filter((r) => !r.sealed), [rows]);
  const current =
    rows?.find((r) => r.id === selected) ?? finals[0] ?? rows?.[0] ?? null;

  if (error)
    return (
      <Empty
        icon="alert"
        title="The evidence could not be loaded"
        description={error}
        action={
          <button className="button" onClick={() => void load()}>
            <Icon name="refresh" size={15} />
            Try again
          </button>
        }
      />
    );
  if (!rows)
    return (
      <div className="ev" aria-busy="true" aria-label="Loading evidence">
        <div className="ev-ledger skeleton-block" />
        <div className="panel skeleton-block tall" />
      </div>
    );
  if (!rows.length)
    return (
      <Empty
        icon="results"
        title="No campaigns yet"
        description="Results, trust checks and usage appear here after the first campaign saves its evidence."
      />
    );

  return (
    <div className="ev">
      <Ledger rows={rows} />
      {finals.length ? (
        <Results
          rows={finals}
          selected={current?.id ?? ""}
          onSelect={setSelected}
        />
      ) : (
        <Empty
          icon="results"
          title="No sealed result yet"
          description="A campaign reports a sealed result after it freezes its selection and runs the final test once."
        />
      )}
      {others.length > 0 && (
        <p className="ev-note">
          {others.length} campaign{others.length === 1 ? "" : "s"} without a
          sealed final:{" "}
          {others.map((r, i) => (
            <span key={r.id}>
              {i ? ", " : ""}
              <button className="ev-inline" onClick={() => setSelected(r.id)}>
                {r.id}
              </button>{" "}
              ({r.status.replaceAll("_", " ")})
            </span>
          ))}
          .
        </p>
      )}
      {current && (
        <>
          <div className="ev-selected">
            <h2>
              {current.name || current.id}
              <span className="mono">{current.id}</span>
            </h2>
            <button className="button" onClick={() => onOpen(current.id)}>
              Open campaign
              <Icon name="arrow" size={14} />
            </button>
          </div>
          <div className="ev-detail">
            <Checks campaign={current} all={rows} />
            <Observability campaign={current} />
          </div>
        </>
      )}
    </div>
  );
}
