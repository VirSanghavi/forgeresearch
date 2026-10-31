import { useId } from "react";
import { Campaign, Agent, Metric, percent, agentNames } from "@/lib/types";
import { Icon } from "./icon";

export function LossChart({
  metrics,
  compact = false,
}: {
  metrics: Metric[];
  compact?: boolean;
}) {
  const id = useId().replace(/:/g, "");
  const w = 640,
    h = compact ? 150 : 210,
    left = 38,
    right = 16,
    top = 18,
    bottom = 30;
  if (!metrics.length)
    return (
      <div className="chart-empty">
        <Icon name="chart" size={26} />
        <span>Loss telemetry appears when optimization starts.</span>
        <small>Reported directly by the training worker</small>
      </div>
    );
  const low = Math.min(0, ...metrics.map((m) => m.loss)),
    high = Math.max(0, ...metrics.map((m) => m.loss));
  const padding = Math.max(0.01, (high - low) * 0.12);
  const max = high + padding,
    min = low < 0 ? low - padding : 0;
  const x = (i: number) =>
    left + (i / Math.max(1, metrics.length - 1)) * (w - left - right);
  const y = (v: number) =>
    top + (1 - (v - min) / (max - min || 1)) * (h - top - bottom);
  const points = metrics.map((m, i) => `${x(i)},${y(m.loss)}`).join(" ");
  return (
    <svg
      className="loss-chart"
      viewBox={`0 0 ${w} ${h}`}
      role="img"
      aria-label={`Training loss over ${metrics.length} steps. First ${metrics[0].loss.toFixed(3)}, latest ${metrics.at(-1)!.loss.toFixed(3)}.`}
    >
      <defs>
        <linearGradient id={`fill-${id}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="var(--accent)" stopOpacity=".13" />
          <stop offset="1" stopColor="var(--accent)" stopOpacity="0" />
        </linearGradient>
      </defs>
      {[0, 1, 2, 3].map((i) => {
        const v = min + ((max - min) * i) / 3;
        return (
          <g key={i}>
            <line
              x1={left}
              x2={w - right}
              y1={y(v)}
              y2={y(v)}
              className="chart-grid"
            />
            <text x={left - 10} y={y(v) + 4} textAnchor="end">
              {v.toFixed(1)}
            </text>
          </g>
        );
      })}
      <polygon
        points={`${left},${h - bottom} ${points} ${x(metrics.length - 1)},${h - bottom}`}
        fill={`url(#fill-${id})`}
      />
      <polyline
        points={points}
        fill="none"
        stroke="var(--accent)"
        strokeWidth="2"
        strokeLinejoin="round"
      />
      <circle
        cx={x(metrics.length - 1)}
        cy={y(metrics.at(-1)!.loss)}
        r="3.5"
        fill="var(--accent)"
      />
      {[0, Math.floor((metrics.length - 1) / 2), metrics.length - 1].map(
        (i, j) => (
          <text
            key={j}
            x={x(i)}
            y={h - 6}
            textAnchor={j === 0 ? "start" : j === 2 ? "end" : "middle"}
          >
            {metrics[i]?.step}
          </text>
        ),
      )}
    </svg>
  );
}

export function Benchmark({ campaign }: { campaign: Campaign | null }) {
  const base = campaign?.final?.baseline || campaign?.baseline;
  const best = campaign?.final?.checkpoint || campaign?.best?.evaluation;
  const delta = base && best ? (best.score - base.score) * 100 : undefined;
  return (
    <div className="benchmark-inner">
      <div className="benchmark-numbers">
        <div>
          <span className="eyebrow">Original model</span>
          <strong>{percent(base?.score)}</strong>
          <small>
            {base ? `${base.correct} / ${base.n} correct` : "Awaiting baseline"}
          </small>
        </div>
        <div className="benchmark-divider">
          <Icon name="arrow" />
        </div>
        <div>
          <span className="eyebrow">
            {campaign?.final ? "Final checkpoint" : "Best approved"}
          </span>
          <strong className={best ? "accent-text" : ""}>
            {percent(best?.score)}
          </strong>
          <small>
            {best
              ? `${best.correct} / ${best.n} correct`
              : "No approved improvement yet"}
          </small>
        </div>
      </div>
      <div
        className="comparison-bars"
        role="group"
        aria-label="Baseline and checkpoint accuracy"
      >
        <div>
          <span>Baseline</span>
          <div className="bar-track">
            <div
              className="bar-base"
              style={{ width: base ? `${base.score * 100}%` : "0%" }}
            />
          </div>
          <b>{percent(base?.score)}</b>
        </div>
        <div>
          <span>Checkpoint</span>
          <div className="bar-track">
            <div
              className="bar-best"
              style={{
                width: best ? `${best.score * 100}%` : "0%",
                background:
                  delta !== undefined && delta < 0
                    ? "var(--danger)"
                    : undefined,
              }}
            />
          </div>
          <b>{percent(best?.score)}</b>
        </div>
      </div>
      <div className="benchmark-foot">
        <span
          className={delta !== undefined && delta > 0 ? "accent-text" : "muted"}
        >
          {delta === undefined
            ? "An improvement must earn approval."
            : `${delta > 0 ? "+" : ""}${delta.toFixed(1)} percentage points`}
        </span>
        <span className="mono">
          {campaign?.final ? "Sealed final" : "Development"}
        </span>
      </div>
      <p className="chart-note">
        {base
          ? `${base.n} GSM8K problems · greedy decoding · numeric exact match`
          : "GSM8K subset · identical evaluation settings"}
        <br />
        {base
          ? "Small sample. Uncertainty and individual predictions are in the report."
          : "Measured results only. No benchmark has run yet."}
      </p>
    </div>
  );
}

const graphNodes: {
  id: Agent;
  x: number;
  y: number;
  label: string;
  detail: string;
  icon: string;
}[] = [
  {
    id: "director",
    x: 245,
    y: 32,
    label: "Research director",
    detail: "Strategy & allocation",
    icon: "target",
  },
  {
    id: "data",
    x: 66,
    y: 157,
    label: "Data researcher",
    detail: "Provenance & curation",
    icon: "artifacts",
  },
  {
    id: "trainer",
    x: 425,
    y: 157,
    label: "Training engineer",
    detail: "Supervised LoRA",
    icon: "chip",
  },
  {
    id: "evaluator",
    x: 245,
    y: 280,
    label: "Experiment evaluator",
    detail: "Independent evaluation",
    icon: "chart",
  },
  {
    id: "governance",
    x: 245,
    y: 414,
    label: "Governance council",
    detail: "Independent veto authority",
    icon: "governance",
  },
];
export function ResearchGraph({
  campaign,
  onAgent,
}: {
  campaign: Campaign | null;
  onAgent: (agent: Agent) => void;
}) {
  const status = campaign?.status || "idle";
  const active: Agent =
    status === "training"
      ? "trainer"
      : ["baseline", "evaluating", "finalizing"].includes(status)
        ? "evaluator"
        : status === "reviewing"
          ? "governance"
          : "director";
  const running =
    !!campaign &&
    !["completed", "failed", "cancelled", "interrupted"].includes(status);
  return (
    <div className="research-graph">
      <div className="graph-paper" />
      <svg
        className="graph-edges"
        viewBox="0 0 710 510"
        preserveAspectRatio="none"
        aria-hidden="true"
      >
        <defs>
          <marker
            id="arrowhead"
            markerWidth="5"
            markerHeight="5"
            refX="4"
            refY="2.5"
            orient="auto"
          >
            <path d="M0 0l5 2.5L0 5" fill="var(--line-strong)" />
          </marker>
        </defs>
        <path
          d="M335 103v22H156v28M335 125h180v28M156 227v29h179v20M515 227v29H335M335 350v59"
          fill="none"
          stroke="var(--line-strong)"
          strokeWidth="1.5"
          markerEnd="url(#arrowhead)"
        />
        <path
          d="M425 449h215V67H426"
          fill="none"
          stroke="var(--line-strong)"
          strokeDasharray="4 5"
          strokeWidth="1.2"
        />
        <text
          x="653"
          y="275"
          transform="rotate(-90 653 275)"
          className="edge-label"
        >
          Independent review
        </text>
      </svg>
      <div className="graph-node-layer">
        {graphNodes.map((n) => (
          <button
            key={n.id}
            className={`graph-node ${running && active === n.id ? "node-active" : ""} ${n.id === "governance" ? "node-governance" : ""}`}
            style={{
              left: `${(n.x / 710) * 100}%`,
              top: `${(n.y / 510) * 100}%`,
            }}
            onClick={() => onAgent(n.id)}
            aria-label={`Inspect ${agentNames[n.id]}`}
          >
            <span className="node-icon">
              <Icon name={n.icon} size={18} />
            </span>
            <span>
              <b>{n.label}</b>
              <small>
                {running && active === n.id && campaign?.progress
                  ? `${campaign.progress.done} / ${campaign.progress.total} ${n.id === "trainer" ? "optimizer steps" : "evaluated"}`
                  : n.id === "trainer" &&
                      campaign?.experiments.at(-1)?.strategy === "grpo"
                    ? "Terse-backed GRPO"
                    : n.detail}
              </small>
            </span>
            <span
              className={`node-dot ${running && active === n.id ? "active" : ""}`}
            />
          </button>
        ))}
      </div>
      <span className="graph-route route-left">verified data</span>
      <span className="graph-route route-right">model weights</span>
      <div className="graph-caption">
        <span>
          <i className="static-dot" />{" "}
          {running
            ? "Evidence-driven execution"
            : campaign
              ? "Persistent research trace"
              : "Ready for a research objective"}
        </span>
        <span className="mono">TERSE × AGENT RELAY</span>
      </div>
    </div>
  );
}
