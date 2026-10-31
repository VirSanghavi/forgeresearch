"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  Campaign,
  DetectorAudit,
  DetectorKind,
  ProductionScan,
  ProductionScanSummary,
  ProductionSignal,
  percent,
} from "@/lib/types";
import { Dialog } from "./dialog";
import { Icon } from "./icon";
import { Empty, PanelHeader } from "./views";

// Measured on scan-d2f134cd: 24 training steps left every held-out production
// failure unfixed (forge-f530d7fa); 96 steps fixed 11 of 33 (forge-ad0318ee).
const PRODUCTION_CAMPAIGN = {
  objective:
    "Fix the failures found in production traffic without benchmark leakage, and measure the fix on held-out production failures.",
  strategies: ["targeted_sft", "broad_sft"],
  max_experiments: 2,
  train_steps: 96,
  dev_samples: 24,
  final_samples: 24,
  max_seconds: 5400,
};

const PAGE = 6;
const TRACES_LIMIT = 8_000_000;
const LABELS_LIMIT = 2_000_000;
const detectorName: Record<DetectorKind, string> = {
  arithmetic_slip: "Arithmetic slip",
  missing_final_answer: "Missing final answer",
  truncated: "Truncated",
  user_reported: "User reported",
};

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const r = await fetch("/api/lab/" + path, { ...options, cache: "no-store" });
  const data = await r.json().catch(() => ({}));
  if (!r.ok)
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : "The request could not be completed.",
    );
  return data;
}
const message = (e: unknown, fallback: string) =>
  e instanceof Error ? e.message : fallback;
const when = (iso: string) =>
  new Date(iso).toLocaleString("en-GB", {
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "UTC",
  }) + " UTC";
const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

export function ProductionView({
  campaignRunning,
  onCampaign,
}: {
  campaignRunning: boolean;
  onCampaign: (campaign: Campaign) => void;
}) {
  const [scans, setScans] = useState<ProductionScanSummary[] | null>(null),
    [listError, setListError] = useState(""),
    [selected, setSelected] = useState(""),
    [scan, setScan] = useState<ProductionScan | null>(null),
    [scanError, setScanError] = useState(""),
    [attempt, setAttempt] = useState(0),
    [uploading, setUploading] = useState(false);
  const loadScans = useCallback(async () => {
    try {
      setScans(await api<ProductionScanSummary[]>("production/scans"));
      setListError("");
    } catch (e) {
      setListError(message(e, "Production scans could not be loaded."));
    }
  }, []);
  useEffect(() => {
    const initial = setTimeout(() => void loadScans(), 0);
    const timer = setInterval(() => void loadScans(), 15000);
    return () => {
      clearTimeout(initial);
      clearInterval(timer);
    };
  }, [loadScans]);
  const active = selected || scans?.[0]?.id || "";
  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    const timer = setTimeout(async () => {
      try {
        const report = await api<ProductionScan>(`production/scans/${active}`);
        if (!cancelled) {
          setScan(report);
          setScanError("");
        }
      } catch (e) {
        if (!cancelled)
          setScanError(message(e, "This scan could not be loaded."));
      }
    }, 0);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [active, attempt]);

  const upload = uploading && (
    <UploadDialog
      onClose={() => setUploading(false)}
      onCreated={(created) => {
        setUploading(false);
        setScans((list) => [
          created,
          ...(list || []).filter((s) => s.id !== created.id),
        ]);
        setSelected(created.id);
      }}
    />
  );

  if (!scans)
    return listError ? (
      <ErrorState
        text={listError}
        onRetry={() => {
          setListError("");
          void loadScans();
        }}
      />
    ) : (
      <ProductionSkeleton />
    );
  if (!scans.length)
    return (
      <section className="panel">
        <div className="empty production-empty">
          <div className="empty-symbol">
            <Icon name="production" size={26} />
          </div>
          <h3>No production traffic scanned yet</h3>
          <p>
            This page finds wrong answers in production traffic without an
            answer key, audits the detectors against reviewer labels and turns
            labeled failures into training data. Upload model prompts and
            outputs as JSONL, or generate simulated traffic with{" "}
            <code className="mono">
              .venv/bin/python scripts/simulate-production.py
            </code>
          </p>
          <button className="button primary" onClick={() => setUploading(true)}>
            <Icon name="plus" size={16} />
            Upload traffic
          </button>
        </div>
        {upload}
      </section>
    );

  const summary = scans.find((s) => s.id === active);
  const ready = scan && scan.id === active ? scan : null;
  return (
    <div className="production-view">
      <section className="campaign-bar" aria-label="Production scan">
        <div className="campaign-identity">
          <div className="campaign-glyph">
            <Icon name="production" size={18} />
          </div>
          <div>
            <label htmlFor="scan-select">Production scan</label>
            <select
              id="scan-select"
              value={active}
              onChange={(e) => {
                setSelected(e.target.value);
                setScanError("");
              }}
            >
              {scans.map((s) => (
                <option value={s.id} key={s.id}>
                  {s.name}
                </option>
              ))}
            </select>
          </div>
        </div>
        {summary && (
          <div className="campaign-meta">
            <span>
              {summary.source === "simulated"
                ? "Simulated traffic"
                : "Uploaded traffic"}
            </span>
            <span className="meta-divider" />
            <span>{when(summary.created_at)}</span>
          </div>
        )}
        <button
          className="button small-button"
          onClick={() => setUploading(true)}
        >
          <Icon name="plus" size={14} />
          Upload traffic
        </button>
      </section>
      {listError && (
        <div className="error-banner" role="alert">
          <Icon name="alert" />
          <span>{listError}</span>
          <button
            className="button small-button"
            onClick={() => void loadScans()}
          >
            Retry
          </button>
        </div>
      )}
      {scanError ? (
        <ErrorState
          text={scanError}
          onRetry={() => {
            setScanError("");
            setAttempt((n) => n + 1);
          }}
        />
      ) : ready ? (
        <ScanReport
          key={ready.id}
          scan={ready}
          campaignRunning={campaignRunning}
          onCampaign={onCampaign}
        />
      ) : (
        <ProductionSkeleton report />
      )}
      {upload}
    </div>
  );
}

function ErrorState({ text, onRetry }: { text: string; onRetry: () => void }) {
  return (
    <div className="error-banner" role="alert">
      <Icon name="alert" />
      <span>{text}</span>
      <button className="button small-button" onClick={onRetry}>
        Retry
      </button>
    </div>
  );
}

function ProductionSkeleton({ report = false }: { report?: boolean }) {
  return (
    <div
      className="production-view production-skeleton"
      aria-busy="true"
      aria-label="Loading production scan"
    >
      {!report && <div className="skeleton skeleton-bar" />}
      <div className="production-summary">
        {[0, 1, 2, 3].map((i) => (
          <div key={i}>
            <i className="skeleton-line short" />
            <i className="skeleton-line number" />
            <i className="skeleton-line" />
          </div>
        ))}
      </div>
      <div className="skeleton skeleton-action" />
      <div className="production-desks">
        <div className="skeleton skeleton-desk" />
        <div className="skeleton skeleton-desk" />
      </div>
      <div className="skeleton skeleton-findings" />
      <span className="loading-note">Reading the scan report…</span>
    </div>
  );
}

function ScanReport({
  scan,
  campaignRunning,
  onCampaign,
}: {
  scan: ProductionScan;
  campaignRunning: boolean;
  onCampaign: (campaign: Campaign) => void;
}) {
  const [filter, setFilter] = useState<DetectorKind | "all">("all"),
    [page, setPage] = useState(0),
    [starting, setStarting] = useState(false),
    [trainError, setTrainError] = useState("");
  const list = useRef<HTMLElement>(null);
  const findings = scan.findings.filter(
    (f) => filter === "all" || f.signals.some((s) => s.kind === filter),
  );
  const pages = Math.max(1, Math.ceil(findings.length / PAGE));
  const shown = findings.slice(page * PAGE, page * PAGE + PAGE);
  const audit = scan.audit;
  const blocked = !audit
    ? "Training needs reviewer labels. Upload this traffic again with a labels file so each failure has a reference answer."
    : campaignRunning
      ? "A campaign already owns the local GPU. Train after it finishes."
      : "";
  async function train() {
    setStarting(true);
    setTrainError("");
    try {
      const campaign = await api<Campaign>("campaigns", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: `Production failures · ${scan.name}`.slice(0, 80),
          production_scan: scan.id,
          ...PRODUCTION_CAMPAIGN,
        }),
      });
      onCampaign(campaign);
    } catch (e) {
      setTrainError(message(e, "The campaign could not start."));
    } finally {
      setStarting(false);
    }
  }
  function turn(next: number) {
    setPage(next);
    list.current?.scrollIntoView({ block: "start", behavior: "instant" });
  }
  const flaggedShare = scan.traces ? scan.flagged / scan.traces : 0;
  return (
    <>
      <section className="production-summary" aria-label="Scan summary">
        <div>
          <span>Traces scanned</span>
          <strong>{scan.traces}</strong>
          <p>Production answers read without an answer key</p>
        </div>
        <div>
          <span>Flagged</span>
          <strong className={scan.flagged ? "accent-text" : ""}>
            {scan.flagged}
          </strong>
          <p>
            {scan.traces
              ? `${percent(flaggedShare)} show at least one failure signal`
              : "No readable traces in this scan"}
          </p>
        </div>
        <div>
          <span>Malformed rows skipped</span>
          <strong>{scan.malformed.length}</strong>
          <p>
            {scan.malformed.length
              ? "Unreadable rows, reported and left out"
              : "Every row was readable"}
          </p>
        </div>
        <div>
          <span>Sealed-test overlaps excluded</span>
          <strong>{scan.excluded_sealed_overlap.length}</strong>
          <p>Prompts matching the sealed GSM8K test are never scanned</p>
        </div>
      </section>
      {scan.source === "simulated" && (
        <p className="production-note">
          <Icon name="info" size={16} />
          <span>
            Simulated traffic: real Qwen2.5-0.5B answers to held-aside GSM8K
            train-split questions. No user feedback was invented.
          </span>
        </p>
      )}
      {(scan.malformed.length > 0 ||
        scan.excluded_sealed_overlap.length > 0) && (
        <details className="production-skipped">
          <summary>
            What was left out
            <Icon name="down" size={14} />
          </summary>
          <ul>
            {scan.malformed.map((m) => (
              <li key={`line-${m.line}`}>
                <span className="mono">Line {m.line}</span>
                <span>{m.reason}</span>
              </li>
            ))}
            {scan.excluded_sealed_overlap.map((id) => (
              <li key={`sealed-${id}`}>
                <span className="mono">{id}</span>
                <span>Matches a sealed test question</span>
              </li>
            ))}
          </ul>
        </details>
      )}
      <section
        className="briefing-final production-train panel"
        aria-label="Train on these failures"
      >
        <div>
          <span>Turn failures into training data</span>
          <h2>
            {audit
              ? `${plural(audit.true_positives, "flagged answer")} confirmed wrong`
              : "Reviewer labels required"}
          </h2>
        </div>
        <div className="production-train-action">
          <p>
            {audit
              ? "A campaign trains on half of the labeled failures and measures the other, held-out half before and after training."
              : "Labels give each failure a reference answer. Without them there is nothing to train on or measure."}
          </p>
          <button
            className="button primary"
            onClick={() => void train()}
            disabled={!!blocked || starting}
            aria-describedby={blocked ? "train-blocked" : undefined}
          >
            {starting ? "Starting campaign…" : "Train on these failures"}
            <Icon name="arrow" size={16} />
          </button>
          {blocked && (
            <p id="train-blocked" className="production-blocked">
              <Icon name="lock" size={14} />
              {blocked}
            </p>
          )}
          {trainError && (
            <p className="form-error" role="alert">
              {trainError}
            </p>
          )}
        </div>
      </section>
      <div className="production-desks">
        <AuditPanel audit={audit} />
        <section className="panel">
          <PanelHeader
            title="What the detectors found"
            icon="production"
            right={<span className="tag">Answers flagged per detector</span>}
          />
          <ul className="detector-list">
            {scan.categories.map((c) => (
              <li key={c.kind}>
                <div>
                  <strong>{detectorName[c.kind] || c.kind}</strong>
                  <span>{c.label}</span>
                </div>
                <b className="mono">{c.count}</b>
                <i aria-hidden="true">
                  <i
                    style={{
                      width: `${scan.traces ? (c.count / scan.traces) * 100 : 0}%`,
                    }}
                  />
                </i>
              </li>
            ))}
          </ul>
        </section>
      </div>
      <section
        className="panel production-findings"
        ref={list}
        aria-label="Flagged answers"
      >
        <PanelHeader
          title="Flagged answers"
          icon="alert"
          subtitle="Model output as served. Wrong equations are marked."
        />
        {scan.findings.length > 0 && (
          <div className="table-tools">
            <label className="filter-label">
              Signal
              <select
                value={filter}
                onChange={(e) => {
                  setFilter(e.target.value as DetectorKind | "all");
                  setPage(0);
                }}
              >
                <option value="all">All signals</option>
                {scan.categories.map((c) => (
                  <option key={c.kind} value={c.kind}>
                    {detectorName[c.kind] || c.kind} ({c.count})
                  </option>
                ))}
              </select>
            </label>
            <span className="mono muted">
              {plural(findings.length, "answer")}
            </span>
          </div>
        )}
        {shown.length ? (
          <ol className="finding-list">
            {shown.map((f) => (
              <Finding key={f.trace.id} finding={f} />
            ))}
          </ol>
        ) : (
          <Empty
            icon="check"
            title={
              scan.findings.length
                ? "No answers match this signal"
                : "No failures detected"
            }
            description={
              scan.findings.length
                ? "Choose another signal to see its flagged answers."
                : "None of the four detectors fired on this traffic. Detectors can miss failures; reviewer labels measure how many."
            }
          />
        )}
        {pages > 1 && (
          <nav className="finding-pager" aria-label="Flagged answer pages">
            <span className="mono">
              {page * PAGE + 1}–{Math.min(findings.length, (page + 1) * PAGE)}{" "}
              of {findings.length}
            </span>
            <div>
              <button
                className="button small-button"
                onClick={() => turn(page - 1)}
                disabled={page === 0}
              >
                Previous
              </button>
              <button
                className="button small-button"
                onClick={() => turn(page + 1)}
                disabled={page >= pages - 1}
              >
                Next
              </button>
            </div>
          </nav>
        )}
      </section>
      <p className="briefing-timestamp mono">
        Scanned {when(scan.created_at)} · {scan.id} · sha256{" "}
        {scan.source_sha256.slice(0, 12)}
      </p>
    </>
  );
}

function AuditPanel({ audit }: { audit: DetectorAudit | null }) {
  return (
    <section className="panel">
      <PanelHeader
        title="Detector audit"
        icon="governance"
        right={<span className="tag">Reviewer labels</span>}
      />
      {audit ? (
        <div className="audit-body">
          <div className="audit-metrics">
            <div>
              <span>Precision</span>
              <strong>
                {audit.precision === null ? "n/a" : percent(audit.precision)}
              </strong>
              <p>
                {audit.true_positives} of{" "}
                {audit.true_positives + audit.false_positives} flagged answers
                were actually wrong
              </p>
            </div>
            <div>
              <span>Recall</span>
              <strong>
                {audit.recall === null ? "n/a" : percent(audit.recall)}
              </strong>
              <p>
                {audit.true_positives} of {audit.actually_wrong} wrong answers
                were flagged
              </p>
            </div>
          </div>
          <dl className="audit-counts">
            <div>
              <dt>Labeled traces</dt>
              <dd className="mono">{audit.labeled}</dd>
            </div>
            <div>
              <dt>Actually wrong</dt>
              <dd className="mono">{audit.actually_wrong}</dd>
            </div>
            <div>
              <dt>False alarms</dt>
              <dd className="mono">{audit.false_positives}</dd>
            </div>
            <div>
              <dt>Missed</dt>
              <dd className="mono">{audit.missed}</dd>
            </div>
          </dl>
          <p className="audit-scope">
            Measured against reviewer labels. Labels are used only to audit the
            detectors and as training references, never to detect.
          </p>
        </div>
      ) : (
        <Empty
          icon="governance"
          title="No reviewer labels were provided"
          description="Precision and recall need a reference answer per trace. Upload a labels file with the traffic to audit the detectors."
        />
      )}
    </section>
  );
}

function Finding({
  finding: { trace, signals },
}: {
  finding: ProductionScan["findings"][number];
}) {
  const kinds = [...new Set(signals.map((s) => s.kind))];
  return (
    <li className="finding">
      <header>
        <span className="mono muted">{trace.id}</span>
        <span className="finding-tags">
          {kinds.map((k) => (
            <span className="tag" key={k}>
              {detectorName[k] || k}
            </span>
          ))}
        </span>
      </header>
      <p className="finding-prompt" title={trace.prompt}>
        {trace.prompt}
      </p>
      <MarkedOutput output={trace.output} signals={signals} />
      <ul className="finding-evidence">
        {signals.map((s, i) => (
          <Evidence key={`${s.kind}-${i}`} signal={s} />
        ))}
      </ul>
    </li>
  );
}

function Evidence({ signal: s }: { signal: ProductionSignal }) {
  if (s.kind === "arithmetic_slip" && s.evidence.equation) {
    const sides = s.evidence.equation.split("=");
    const wrote = sides.at(-1)?.trim();
    const left = sides.slice(0, -1).join("=").trim();
    const plain = /^\$?[\d,]+(\.\d+)?$/.test(wrote || "");
    return (
      <li>
        <Icon name="alert" size={14} />
        <span>
          Wrote <b className="mono">{wrote}</b>
          {plain ? ", actually " : `, but ${left} is `}
          <b className="mono">{s.evidence.computed}</b>
          {plain && (
            <code className="mono slip-equation">{s.evidence.equation}</code>
          )}
        </span>
      </li>
    );
  }
  if (s.kind === "missing_final_answer")
    return (
      <li>
        <Icon name="alert" size={14} />
        <span>
          No final <code className="mono">#### &lt;number&gt;</code> line, so
          the answer cannot be used
        </span>
      </li>
    );
  if (s.kind === "truncated")
    return (
      <li>
        <Icon name="alert" size={14} />
        <span>Generation stopped at the length limit</span>
      </li>
    );
  if (s.kind === "user_reported")
    return (
      <li>
        <Icon name="message" size={14} />
        <span>
          {s.evidence.correction ? (
            <>
              User correction:{" "}
              <q className="finding-correction">{s.evidence.correction}</q>
            </>
          ) : (
            "The user marked this answer wrong"
          )}
        </span>
      </li>
    );
  return null;
}

/**
 * The detector reads LaTeX-normalized text ("\\times" becomes "*"), so its
 * evidence is not always a verbatim substring. Match it back tolerantly.
 */
function equationPattern(equation: string) {
  const tokens = equation.match(/\$?\d[\d,]*(?:\.\d+)?|[-+*/=()]/g);
  if (!tokens) return null;
  const gap = String.raw`(?:\s|\\[\[\]()]|\\[,;!])*`;
  const token = (t: string) =>
    t === "*"
      ? String.raw`(?:\*|\\times|\\cdot|×|·|x)`
      : t === "/"
        ? String.raw`(?:/|\\div|÷)`
        : t.replace(/[.*+?^${}()|[\]\\/-]/g, "\\$&");
  return new RegExp(tokens.map(token).join(gap), "g");
}

/** The output as served, with each wrong equation marked where it was written. */
function MarkedOutput({
  output,
  signals,
}: {
  output: string;
  signals: ProductionSignal[];
}) {
  const box = useRef<HTMLPreElement>(null);
  const ranges: [number, number][] = [];
  for (const s of signals) {
    const eq = s.kind === "arithmetic_slip" ? s.evidence.equation : undefined;
    const pattern = eq && equationPattern(eq);
    if (!pattern) continue;
    for (const m of output.matchAll(pattern)) {
      const span: [number, number] = [m.index, m.index + m[0].length];
      if (!ranges.some(([a, b]) => span[0] < b && span[1] > a)) {
        ranges.push(span);
        break;
      }
    }
  }
  ranges.sort((a, b) => a[0] - b[0]);
  const parts: React.ReactNode[] = [];
  let cursor = 0;
  ranges.forEach(([a, b]) => {
    if (a > cursor) parts.push(output.slice(cursor, a));
    parts.push(
      <mark key={a} className="slip-mark">
        {output.slice(a, b)}
      </mark>,
    );
    cursor = b;
  });
  parts.push(output.slice(cursor));
  useEffect(() => {
    const el = box.current;
    const mark = el?.querySelector<HTMLElement>("mark");
    if (el && mark && mark.offsetTop + mark.offsetHeight > el.clientHeight)
      el.scrollTop = mark.offsetTop - el.clientHeight / 3;
  }, [output]);
  return (
    <pre
      ref={box}
      className="prediction-output finding-output"
      tabIndex={0}
      aria-label="Model output"
    >
      {parts}
    </pre>
  );
}

function UploadDialog({
  onClose,
  onCreated,
}: {
  onClose: () => void;
  onCreated: (scan: ProductionScanSummary) => void;
}) {
  const [busy, setBusy] = useState(false),
    [errors, setErrors] = useState<
      Partial<Record<"name" | "traces" | "labels" | "form", string>>
    >({});
  async function submit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const f = new FormData(e.currentTarget);
    const name = String(f.get("name") || "").trim();
    const traces = f.get("traces");
    const labels = f.get("labels");
    const found: typeof errors = {};
    if (!name) found.name = "Name this scan so you can find it later.";
    let tracesText = "",
      labelsText = "";
    if (!(traces instanceof File) || !traces.size)
      found.traces = "Choose a traces .jsonl file.";
    else if (traces.size > TRACES_LIMIT)
      found.traces = "The traces file is larger than 8 MB. Split it first.";
    else {
      tracesText = await traces.text();
      const lines = tracesText.split("\n").filter((l) => l.trim());
      const readable = lines.filter((l) => {
        try {
          const row = JSON.parse(l);
          return row && typeof row === "object" && !Array.isArray(row);
        } catch {
          return false;
        }
      });
      if (!readable.length)
        found.traces =
          "No line in this file is a JSON object. Each line needs id, prompt and output.";
    }
    if (labels instanceof File && labels.size) {
      if (labels.size > LABELS_LIMIT)
        found.labels = "The labels file is larger than 2 MB.";
      else {
        labelsText = await labels.text();
        const bad = labelsText.split("\n").findIndex((l) => {
          if (!l.trim()) return false;
          try {
            const row = JSON.parse(l);
            return (
              typeof row?.id !== "string" || typeof row?.answer !== "string"
            );
          } catch {
            return true;
          }
        });
        if (bad !== -1)
          found.labels = `Line ${bad + 1} needs a JSON object with string id and answer.`;
      }
    }
    setErrors(found);
    if (Object.keys(found).length) return;
    setBusy(true);
    try {
      onCreated(
        await api<ProductionScanSummary>("production/scans", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name,
            traces: tracesText,
            ...(labelsText.trim() ? { labels: labelsText } : {}),
          }),
        }),
      );
    } catch (error) {
      setErrors({ form: message(error, "The scan could not be created.") });
    } finally {
      setBusy(false);
    }
  }
  const field = (key: "name" | "traces" | "labels") =>
    errors[key]
      ? { "aria-invalid": true, "aria-describedby": `${key}-error` }
      : {};
  return (
    <Dialog
      title="Scan production traffic"
      subtitle="Detectors read only what a deployed model leaves behind: its output, the stop reason and user feedback."
      onClose={onClose}
    >
      <form onSubmit={submit} className="campaign-form" noValidate>
        <label>
          Scan name
          <input
            name="name"
            maxLength={80}
            placeholder="Math assistant traffic, week 44"
            data-autofocus
            {...field("name")}
          />
          {errors.name && (
            <span id="name-error" className="field-error">
              {errors.name}
            </span>
          )}
        </label>
        <label>
          Traces
          <input
            type="file"
            name="traces"
            accept=".jsonl,.ndjson,.json,.txt"
            {...field("traces")}
          />
          {errors.traces ? (
            <span id="traces-error" className="field-error">
              {errors.traces}
            </span>
          ) : (
            <span className="field-help">
              JSONL, one trace per line: <code>id</code>, <code>prompt</code>,{" "}
              <code>output</code>, optional <code>finish_reason</code> and{" "}
              <code>feedback</code>. Unreadable rows are reported and skipped.
            </span>
          )}
        </label>
        <label>
          Reviewer labels (optional)
          <input
            type="file"
            name="labels"
            accept=".jsonl,.ndjson,.json,.txt"
            {...field("labels")}
          />
          {errors.labels ? (
            <span id="labels-error" className="field-error">
              {errors.labels}
            </span>
          ) : (
            <span className="field-help">
              JSONL with <code>id</code> and an <code>answer</code> ending in{" "}
              <code>#### &lt;number&gt;</code>. Used to audit the detectors and
              to train, never to detect.
            </span>
          )}
        </label>
        {errors.form && (
          <p className="form-error" role="alert">
            {errors.form}
          </p>
        )}
        <footer className="dialog-actions">
          <span className="muted small">
            Prompts matching the sealed test set are excluded
          </span>
          <button type="submit" className="button primary" disabled={busy}>
            {busy ? "Scanning traffic…" : "Scan traffic"}
            <Icon name="arrow" size={16} />
          </button>
        </footer>
      </form>
    </Dialog>
  );
}
