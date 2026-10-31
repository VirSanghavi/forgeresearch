"use client";
import { useState } from "react";
import {
  Artifact,
  Campaign,
  Environment,
  Event,
  Experiment,
  Health,
  percent,
  strategyName,
  agentNames,
  duration,
} from "@/lib/types";
import { Icon } from "./icon";
import { Dialog } from "./dialog";

const ALGORITHMS: Record<string, string> = {
  broad_sft: "Supervised fine-tuning with LoRA",
  targeted_sft: "Supervised fine-tuning with LoRA, production failures first",
  distill_sft: "Supervised fine-tuning with LoRA on verified teacher solutions",
  hard_sft:
    "Supervised fine-tuning with LoRA on the questions the base model misses",
  grpo: "TRL GRPO with Terse world rewards",
  dpo: "TRL DPO preference alignment with LoRA",
};
import { LossChart } from "./charts";

export function Status({ value }: { value: string }) {
  const tone = [
    "completed",
    "approved",
    "saved",
    "connected",
    "correct",
  ].includes(value)
    ? "success"
    : ["failed", "rejected", "interrupted", "incorrect"].includes(value)
      ? "danger"
      : [
            "training",
            "evaluating",
            "baseline",
            "finalizing",
            "recording",
            "reviewing",
            "diagnosing",
          ].includes(value)
        ? "active"
        : "neutral";
  return (
    <span className={`status ${tone}`}>
      <i />
      {value.replaceAll("_", " ")}
    </span>
  );
}
export function Empty({
  icon = "experiments",
  title,
  description,
  action,
}: {
  icon?: string;
  title: string;
  description: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="empty">
      <div className="empty-symbol">
        <Icon name={icon} size={26} />
      </div>
      <h3>{title}</h3>
      <p>{description}</p>
      {action}
    </div>
  );
}
export function PanelHeader({
  title,
  icon,
  right,
  subtitle,
}: {
  title: string;
  icon?: string;
  right?: React.ReactNode;
  subtitle?: string;
}) {
  return (
    <header className="panel-header">
      <div>
        {icon && <Icon name={icon} size={17} />}
        <div>
          <h2>{title}</h2>
          {subtitle && <p>{subtitle}</p>}
        </div>
      </div>
      {right}
    </header>
  );
}
export function ExperimentTable({
  campaign,
  onSelect,
  compact = false,
}: {
  campaign: Campaign | null;
  onSelect: (e: Experiment) => void;
  compact?: boolean;
}) {
  const [query, setQuery] = useState("");
  const exps = (campaign?.experiments || []).filter((e) =>
    `${e.id} ${e.strategy} ${e.hypothesis}`
      .toLowerCase()
      .includes(query.toLowerCase()),
  );
  return (
    <div className="experiment-ledger">
      {!compact && (
        <div className="table-tools">
          <label className="search-field">
            <Icon name="search" size={17} />
            <input
              placeholder="Find an experiment…"
              aria-label="Search experiments"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </label>
          <span className="mono muted">
            {exps.length} experiment{exps.length === 1 ? "" : "s"}
          </span>
        </div>
      )}
      {!exps.length ? (
        <Empty
          title={
            query ? "No matching experiments" : "Every hypothesis starts here"
          }
          description={
            query
              ? "Try a different experiment ID or strategy."
              : "The research director will propose experiments after measuring the baseline."
          }
        />
      ) : (
        <div className="table-scroll">
          <table className="experiment-compare">
            <thead>
              <tr>
                <th scope="col">Experiment</th>
                <th scope="col">Method</th>
                <th scope="col">Status</th>
                <th scope="col" className="numeric">
                  Development
                </th>
                {!compact && (
                  <>
                    <th scope="col" className="numeric">
                      vs original
                    </th>
                    <th scope="col" className="numeric">
                      Production held-out
                    </th>
                    <th scope="col" className="numeric">
                      Final loss
                    </th>
                    <th scope="col" className="numeric">
                      Training
                    </th>
                    <th scope="col">Governance</th>
                  </>
                )}
                <th scope="col">
                  <span className="sr-only">Details</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {exps.map((e) => {
                const base = campaign?.baseline;
                const delta =
                  e.evaluation && base
                    ? (e.evaluation.score - base.score) * 100
                    : null;
                const selected = campaign?.best?.id === e.id;
                return (
                  <tr key={e.id} onClick={() => onSelect(e)}>
                    <td>
                      <button
                        className="table-link"
                        onClick={(event) => {
                          event.stopPropagation();
                          onSelect(e);
                        }}
                      >
                        <span className="mono">{e.id}</span>
                        <small>
                          {selected
                            ? "Selected on development data"
                            : e.proposal.focus.replaceAll("_", " ")}
                        </small>
                      </button>
                    </td>
                    <td>
                      {strategyName(e.strategy)}
                      <small>
                        LoRA rank {e.proposal.lora_rank} · lr{" "}
                        {e.proposal.learning_rate}
                      </small>
                    </td>
                    <td>
                      <Status value={e.status} />
                    </td>
                    <td className="numeric">
                      {e.evaluation
                        ? `${e.evaluation.correct} of ${e.evaluation.n}`
                        : "Not yet"}
                    </td>
                    {!compact && (
                      <>
                        <td
                          className={`numeric ${delta === null || delta === 0 ? "" : delta > 0 ? "accent-text" : "warning-text"}`}
                        >
                          {delta === null
                            ? "Not yet"
                            : `${delta > 0 ? "+" : delta < 0 ? "−" : ""}${Math.abs(delta).toFixed(1)} pts`}
                        </td>
                        <td className="numeric">
                          {e.production
                            ? `${e.production.correct} of ${e.production.n}`
                            : "Not measured"}
                        </td>
                        <td className="numeric">
                          {e.metrics.at(-1)?.loss.toFixed(3) ?? "Not yet"}
                        </td>
                        <td className="numeric">
                          {e.training
                            ? duration(e.training.seconds)
                            : "Not yet"}
                        </td>
                        <td>
                          {e.review ? (
                            <Status value={e.review.decision} />
                          ) : (
                            <span className="muted">Pending</span>
                          )}
                        </td>
                      </>
                    )}
                    <td>
                      <Icon name="chevron" size={15} />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export function ExperimentDetail({
  experiment: e,
  onClose,
  campaign,
}: {
  experiment: Experiment;
  onClose: () => void;
  campaign: Campaign | null;
}) {
  const [tab, setTab] = useState("overview");
  return (
    <Dialog
      title={e.id}
      subtitle={strategyName(e.strategy) + " · " + e.proposal.director}
      onClose={onClose}
      wide
      drawer
    >
      <div className="detail-title">
        <Status value={e.status} />
        <span className="mono muted">
          {new Date(e.created_at).toLocaleString()}
        </span>
      </div>
      <div className="detail-tabs">
        {["overview", "predictions", "configuration"].map((t) => (
          <button
            key={t}
            className={tab === t ? "selected" : ""}
            onClick={() => setTab(t)}
          >
            {t[0].toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>
      {tab === "overview" ? (
        <>
          <div className="detail-section">
            <span className="eyebrow">Research hypothesis</span>
            <p className="hypothesis">{e.hypothesis}</p>
          </div>
          <div className="detail-stats">
            <div>
              <span>Development accuracy</span>
              <strong>{percent(e.evaluation?.score)}</strong>
            </div>
            <div>
              <span>Optimization steps</span>
              <strong>
                {e.metrics.length}
                <small> / {e.max_steps}</small>
              </strong>
            </div>
            <div>
              <span>Latest loss</span>
              <strong>{e.metrics.at(-1)?.loss.toFixed(3) || "Not yet"}</strong>
            </div>
          </div>
          <div className="detail-section">
            <span className="eyebrow">Training loss</span>
            <LossChart metrics={e.metrics} />
          </div>
          <div className="detail-section">
            <span className="eyebrow">Independent review</span>
            {e.review ? (
              <div className="review-detail">
                <Status value={e.review.decision} />
                <p>
                  {e.review.reason_codes.length
                    ? e.review.reason_codes.join(", ").replaceAll("_", " ")
                    : "Saved weights changed, checkpoint reloaded independently, provenance verified, and declared regression threshold satisfied."}
                </p>
              </div>
            ) : (
              <p className="muted">
                Review begins after training and evaluation complete.
              </p>
            )}
          </div>
          {e.training && (
            <div className="detail-section">
              <span className="eyebrow">Checkpoint identity</span>
              <p className="hash">{e.training.sha256}</p>
              <p className="small accent-text">
                <Icon name="check" size={14} /> Trainable weights changed:{" "}
                {e.training.weights_changed ? "confirmed" : "not confirmed"}
              </p>
            </div>
          )}
        </>
      ) : tab === "predictions" ? (
        <div className="predictions">
          {e.evaluation?.predictions.length ? (
            e.evaluation.predictions.map((p) => (
              <article key={p.id}>
                <div>
                  <span className="mono muted">{p.id}</span>
                  <Status value={p.correct ? "correct" : "incorrect"} />
                </div>
                <h3>{p.question}</h3>
                <div className="prediction-output">{p.output}</div>
                <footer>
                  Expected <b className="mono">{p.expected}</b> · Predicted{" "}
                  <b className="mono">{p.answer || "No numeric answer"}</b>
                </footer>
              </article>
            ))
          ) : (
            <Empty
              icon="chart"
              title="Evaluation has not finished"
              description="Sample-level predictions appear after the saved checkpoint is reloaded and evaluated."
            />
          )}
        </div>
      ) : (
        <>
          <div className="config-list">
            {Object.entries({
              Algorithm: ALGORITHMS[e.strategy] ?? ALGORITHMS.broad_sft,
              "Base model":
                campaign?.config.model_id ?? "Qwen/Qwen2.5-0.5B-Instruct",
              "Learning rate": e.proposal.learning_rate,
              "LoRA rank": e.proposal.lora_rank,
              "LoRA targets": (
                e.training?.lora_targets ?? ["q_proj", "v_proj"]
              ).join(", "),
              "Maximum sequence length": e.training?.max_sequence_length ?? 768,
              Seed: campaign?.config.seed,
              Dataset:
                !campaign?.config.task_pack ||
                campaign.config.task_pack === "gsm8k"
                  ? "openai/gsm8k, disjoint train slice"
                  : `Task pack ${campaign.config.task_pack}, training split`,
              "Reward version":
                (e.review?.evidence as { reward_version?: string } | undefined)
                  ?.reward_version ?? "exact-answer-v1",
              Precision: `${e.training?.dtype ?? "float16"} on ${e.training?.device?.name ?? "Metal or CUDA"}`,
            }).map(([k, v]) => (
              <div key={k}>
                <span>{k}</span>
                <strong className="mono">{v}</strong>
              </div>
            ))}
          </div>
          <p className="chart-note">
            The complete optimizer configuration and hashes are available in the
            training artifact.
          </p>
        </>
      )}
    </Dialog>
  );
}

export function Communications({
  events,
  compact = false,
}: {
  events: Event[];
  compact?: boolean;
}) {
  const [filter, setFilter] = useState("all");
  const rows = events
    .filter(
      (e) => e.kind === "message" && (filter === "all" || e.agent === filter),
    )
    .slice(compact ? -5 : -40)
    .reverse();
  return (
    <div className="communications">
      {!compact && (
        <div className="table-tools">
          <label className="filter-label">
            Agent
            <select value={filter} onChange={(e) => setFilter(e.target.value)}>
              <option value="all">All researchers</option>
              {Object.entries(agentNames).map(([id, name]) => (
                <option key={id} value={id}>
                  {name}
                </option>
              ))}
            </select>
          </label>
          <span className="tag">
            <Icon name="connections" size={13} />
            Agent Relay
          </span>
        </div>
      )}
      {!rows.length ? (
        <Empty
          icon="message"
          title="A channel for consequential work"
          description="Real research assignments, findings, and reviews appear here when a campaign begins."
        />
      ) : (
        <div
          className="message-feed"
          role="log"
          aria-label="Research communications"
        >
          {rows.map((e) => (
            <article className="message-row" key={e.id}>
              <div className={`agent-avatar avatar-${e.agent}`}>
                {e.agent.slice(0, 1).toUpperCase()}
              </div>
              <div className="message-content">
                <header>
                  <b>{agentNames[e.agent] || e.agent}</b>
                  {e.recipient && (
                    <>
                      <Icon name="arrow" size={12} />
                      <span>{e.recipient}</span>
                    </>
                  )}
                  <time dateTime={e.timestamp}>
                    {new Date(e.timestamp).toLocaleTimeString([], {
                      hour: "2-digit",
                      minute: "2-digit",
                      hour12: false,
                    })}
                  </time>
                </header>
                <p>{e.text}</p>
                <footer>
                  <span>{e.category?.toLowerCase().replaceAll("_", " ")}</span>
                  <span className="mono">relay/{e.relay_id?.slice(-8)}</span>
                </footer>
              </div>
            </article>
          ))}
        </div>
      )}
    </div>
  );
}

export function EnvironmentsView({
  campaign,
  onSelect,
}: {
  campaign: Campaign | null;
  onSelect: (e: Environment) => void;
}) {
  const [filter, setFilter] = useState("all");
  const envs = (campaign?.environments || []).filter(
    (e) =>
      filter === "all" ||
      (filter === "correct" ? e.reward === 1 : e.reward === 0),
  );
  return (
    <section className="panel">
      <PanelHeader
        title="Environment fleet"
        subtitle="Real, isolated mathematical tasks persisted through Terse."
        right={<span className="tag">Terse Durable Actors</span>}
      />
      <div className="table-tools">
        <label className="filter-label">
          Outcome
          <select value={filter} onChange={(e) => setFilter(e.target.value)}>
            <option value="all">All environments</option>
            <option value="correct">Correct answer</option>
            <option value="incorrect">Incorrect answer</option>
          </select>
        </label>
        <span className="mono muted">{envs.length} ACTORS</span>
      </div>
      <div className="environment-grid">
        {envs.map((e, i) => (
          <button
            className="environment-card"
            key={e.id}
            onClick={() => onSelect(e)}
          >
            <header>
              <span className="mono">
                ENV / {String(i + 1).padStart(3, "0")}
              </span>
              <span className={e.reward === 1 ? "accent-text" : "warning-text"}>
                {e.reward === 1 ? (
                  <Icon name="check" size={17} />
                ) : (
                  <Icon name="close" size={17} />
                )}
              </span>
            </header>
            <p>{e.task}</p>
            <footer>
              <span>
                <i className="static-dot" />{" "}
                {e.terminated ? "Terminated" : "Ready"}
              </span>
              <span className="mono">reward {e.reward ?? "none"}</span>
            </footer>
          </button>
        ))}
      </div>
      {!envs.length && (
        <Empty
          icon="environments"
          title="Worlds with a memory"
          description="Each development prediction is submitted to its own Terse actor. Open a world to inspect its task, action, reward and state checksum."
        />
      )}
      <div className="section-note">
        <Icon name="info" size={16} />
        <span>
          These worlds score real model predictions. The current training path
          includes supervised LoRA and optional GRPO. GRPO rewards come directly
          from these actors.
        </span>
      </div>
    </section>
  );
}
export function EnvironmentDetail({
  environment: e,
  onClose,
}: {
  environment: Environment;
  onClose: () => void;
}) {
  const [step, setStep] = useState(e.actions.length);
  return (
    <Dialog
      title="Environment inspector"
      subtitle="Terse RLWorldActor · persisted state"
      onClose={onClose}
      drawer
    >
      <div className="detail-title">
        <span className="mono hash">{e.id}</span>
        <Status value={step ? "completed" : "initialized"} />
      </div>
      <div className="detail-section">
        <span className="eyebrow">Task</span>
        <p className="hypothesis">{e.task}</p>
      </div>
      <div className="detail-stats">
        <div>
          <span>Version</span>
          <strong>{step ? e.version : 0}</strong>
        </div>
        <div>
          <span>Reward</span>
          <strong className={e.reward === 1 ? "accent-text" : ""}>
            {step ? e.reward : "Not yet"}
          </strong>
        </div>
        <div>
          <span>Seed</span>
          <strong>{e.seed}</strong>
        </div>
      </div>
      <div className="detail-section">
        <span className="eyebrow">Trajectory replay</span>
        <p className="small muted">
          Inspect recorded transitions. Replaying does not mutate the actor.
        </p>
        <div className="replay-controls">
          <button
            className="button"
            onClick={() => setStep(0)}
            disabled={!step}
          >
            <Icon name="refresh" size={15} />
            Reset view
          </button>
          <span className="mono">
            {step} / {e.actions.length}
          </span>
          <button
            className="button"
            onClick={() => setStep(Math.min(e.actions.length, step + 1))}
            disabled={step === e.actions.length}
          >
            Next action
            <Icon name="arrow" size={15} />
          </button>
        </div>
        {step ? (
          e.actions.slice(0, step).map((a) => (
            <div className="trajectory-action" key={a.action_id}>
              <span className="mono">submit_answer</span>
              <p>
                Model submitted <b>{a.answer || "no numeric answer"}</b>
              </p>
              <footer>
                Deterministic reward: <b>{a.reward}</b> · Episode terminated
              </footer>
            </div>
          ))
        ) : (
          <div className="trajectory-action muted">
            Initial observation. The hidden answer is not included in actor
            observations.
          </div>
        )}
      </div>
      <div className="config-list">
        <div>
          <span>Reward function</span>
          <strong className="mono">{e.reward_version}</strong>
        </div>
        <div>
          <span>Termination</span>
          <strong>{step ? "True" : "False"}</strong>
        </div>
        <div>
          <span>Duplicate action protection</span>
          <strong>Enforced</strong>
        </div>
      </div>
      <div className="detail-section">
        <span className="eyebrow">Committed state checksum</span>
        <p className="hash mono">{e.checksum}</p>
      </div>
    </Dialog>
  );
}

export function GovernanceView({
  campaign,
  onProbe,
  probing,
}: {
  campaign: Campaign | null;
  onProbe: () => void;
  probing: boolean;
}) {
  const reviews = campaign?.reviews || [];
  const policies = [
    [
      "Immutable objective",
      "Research agents cannot rewrite the original campaign objective.",
    ],
    [
      "Sealed final evaluation",
      "Checkpoint selection freezes before test questions are accessed.",
    ],
    [
      "Fixed reward function",
      "Only exact-answer-v1 is permitted. Modified rewards are rejected.",
    ],
    [
      "Checkpoint integrity",
      "Changed weights, a verified hash and an independent reload are required.",
    ],
    [
      "Regression boundary",
      "A drop greater than 25 percentage points on the declared math suite blocks promotion.",
    ],
    [
      "Serialized promotion",
      "A rejected or unreviewed checkpoint cannot pass the governance actor.",
    ],
  ];
  return (
    <div className="governance-layout">
      <section className="panel">
        <PanelHeader
          title="Research boundaries"
          icon="governance"
          right={<span className="tag success-tag">Policy v1 · immutable</span>}
        />
        <div className="policies">
          {policies.map(([name, text], i) => (
            <div key={name}>
              <span className="policy-index mono">0{i + 1}</span>
              <div>
                <strong>{name}</strong>
                <p>{text}</p>
              </div>
              <Icon name="check" size={17} />
            </div>
          ))}
        </div>
        <div className="section-note">
          <Icon name="info" size={16} />
          <span>
            Defined controls over specific failure modes. These checks do not
            establish general model safety or solve alignment.
          </span>
        </div>
      </section>
      <section className="panel">
        <PanelHeader title="Put the gate to the test" icon="target" />
        <div className="probe-panel">
          <h3>Watch the gate reject a bad proposal</h3>
          <p>
            Inject a clearly labeled proposal that changes the reward and
            requests sealed data. The real governance actor must reject it and
            block promotion.
          </p>
          <button
            className="button"
            onClick={onProbe}
            disabled={!campaign || probing}
          >
            {probing ? "Checking enforcement…" : "Run integrity probe"}
            <Icon name="arrow" size={16} />
          </button>
          <small>
            Controlled adversarial fixture. No real experiment is altered.
          </small>
        </div>
      </section>
      <section className="panel governance-history">
        <PanelHeader
          title="Decision ledger"
          right={<span className="mono muted">{reviews.length} REVIEWS</span>}
        />
        {reviews.length ? (
          <div className="review-list">
            {[...reviews].reverse().map((r) => (
              <article key={r.id}>
                <div
                  className={
                    r.decision === "approved"
                      ? "review-icon approved"
                      : "review-icon rejected"
                  }
                >
                  <Icon
                    name={r.decision === "approved" ? "check" : "close"}
                    size={18}
                  />
                </div>
                <div>
                  <header>
                    <strong>
                      {r.evidence.controlled_probe
                        ? "Controlled integrity probe"
                        : r.evidence.phase === "checkpoint"
                          ? "Checkpoint review"
                          : "Experiment preflight"}
                    </strong>
                    <Status value={r.decision} />
                  </header>
                  <p>
                    {r.reason_codes.length
                      ? r.reason_codes
                          .join(" · ")
                          .toLowerCase()
                          .replaceAll("_", " ")
                      : "All required evidence and policy checks passed."}
                  </p>
                  <footer className="mono">
                    {r.id} · {r.policy_version}
                  </footer>
                </div>
              </article>
            ))}
          </div>
        ) : (
          <Empty
            icon="governance"
            title="Independent by design"
            description="Every preflight and checkpoint decision is preserved here with its evidence and reason codes."
          />
        )}
      </section>
    </div>
  );
}

export function ArtifactsView({
  campaign,
  onArtifact,
}: {
  campaign: Campaign | null;
  onArtifact: (a: Artifact) => void;
}) {
  const [filter, setFilter] = useState("all");
  const rows = (campaign?.artifacts || []).filter(
    (a) => filter === "all" || a.kind === filter,
  );
  return (
    <section className="panel">
      <PanelHeader
        title="Evidence registry"
        subtitle="Immutable references. Downloads are checked against their SHA-256 hash."
        right={
          <span className="tag">
            <Icon name="governance" size={13} />
            Integrity checked
          </span>
        }
      />
      <div className="table-tools">
        <label className="filter-label">
          Artifact type
          <select value={filter} onChange={(e) => setFilter(e.target.value)}>
            <option value="all">All artifacts</option>
            <option value="checkpoint_zip">Checkpoint bundles</option>
            <option value="evaluation">Evaluations</option>
            <option value="training">Training configurations</option>
            <option value="dataset">Datasets</option>
            <option value="report">Reports</option>
            <option value="recording">Recordings</option>
            <option value="trajectory">Environment trajectories</option>
            <option value="trace">Research traces</option>
          </select>
        </label>
        {campaign && (
          <a
            className="button small-button"
            href={`/api/lab/campaigns/${campaign.id}/trace`}
          >
            <Icon name="download" size={15} />
            Export trace
          </a>
        )}
      </div>
      {rows.length ? (
        <div className="artifact-list">
          {rows.map((a) => (
            <article key={a.id}>
              <span className="artifact-icon">
                <Icon
                  name={
                    a.kind.startsWith("checkpoint")
                      ? "chip"
                      : a.kind === "evaluation"
                        ? "chart"
                        : a.kind === "dataset"
                          ? "artifacts"
                          : "code"
                  }
                  size={20}
                />
              </span>
              <button onClick={() => onArtifact(a)} className="artifact-name">
                <strong>{a.name}</strong>
                <span className="mono">{a.path}</span>
              </button>
              <span className="artifact-size mono">
                {a.bytes > 1048576
                  ? `${(a.bytes / 1048576).toFixed(1)} MB`
                  : `${(a.bytes / 1024).toFixed(1)} KB`}
              </span>
              <a
                className="icon-button"
                aria-label={`Download ${a.name}`}
                href={`/api/lab/campaigns/${campaign!.id}/artifacts/${a.id}`}
              >
                <Icon name="download" />
              </a>
            </article>
          ))}
        </div>
      ) : (
        <Empty
          icon="artifacts"
          title="Evidence, preserved"
          description="Checkpoint weights, predictions, training configurations and dataset manifests are registered as the research proceeds."
        />
      )}
      <div className="section-note">
        <Icon name="info" size={16} />
        <span>
          LoRA adapters require the pinned base model. Each bundle includes its
          adapter configuration. See reproduction instructions for loading.
        </span>
      </div>
    </section>
  );
}

export function ConnectionsView({ health }: { health: Health | null }) {
  const connections = [
    {
      id: "terse",
      name: "Terse Durable Actors",
      icon: "environments",
      description:
        "Authoritative campaigns, atomic budgets, isolated worlds, governance and artifact references.",
      version: "0.14.0",
      endpoint: "127.0.0.1:7100",
    },
    {
      id: "relay",
      name: "Agent Relay",
      icon: "message",
      description:
        "Real assignments, findings and corrections between five registered research identities.",
      version:
        health?.services.relay?.transport || "SDK 14.0.0 · Engine 12.0.0",
      endpoint: health?.services.relay?.endpoint || "127.0.0.1:7528",
    },
    {
      id: "director",
      name: "Research agents",
      icon: "target",
      description:
        "All five roles reason with this model and can start subagents for focused analysis. Credentials stay on the server.",
      version: health?.services.director?.model || "Not configured",
      endpoint: health?.services.director?.mode || "Azure-hosted deployment",
    },
    {
      id: "storage",
      name: "Supabase",
      icon: "artifacts",
      description:
        "Private, immutable evidence archive. Local artifacts remain available if cloud upload fails.",
      version: "Private Storage + Postgres",
      endpoint: "forge-research-lab",
    },
  ];
  return (
    <>
      <div className="connection-grid">
        {connections.map((c) => (
          <section className="panel connection-panel" key={c.id}>
            <header>
              <span className="connection-icon">
                <Icon name={c.icon} size={27} />
              </span>
              <Status
                value={
                  health?.services[c.id]?.connected
                    ? "connected"
                    : "unavailable"
                }
              />
            </header>
            <h2>{c.name}</h2>
            <p>{c.description}</p>
            <dl>
              <div>
                <dt>Runtime</dt>
                <dd>{c.version}</dd>
              </div>
              <div>
                <dt>Connection</dt>
                <dd className="mono">{c.endpoint}</dd>
              </div>
              <div>
                <dt>Mode</dt>
                <dd>{health?.services[c.id]?.mode || "Checking connection"}</dd>
              </div>
            </dl>
          </section>
        ))}
      </div>
      <section className="panel worker-panel">
        <Icon name="chip" size={28} />
        <div>
          <h2>{health?.worker || "Local compute worker"}</h2>
          <p>
            PyTorch performs actual optimization. Terse coordinates the work
            without blocking on model inference.
          </p>
        </div>
        <span className="tag">Bounded execution</span>
      </section>
    </>
  );
}

export function ArtifactDetail({
  artifact: a,
  campaign,
  onClose,
}: {
  artifact: Artifact;
  campaign: Campaign;
  onClose: () => void;
}) {
  const [copied, setCopied] = useState(false);
  return (
    <Dialog
      title={a.name}
      subtitle="Immutable artifact registry"
      onClose={onClose}
    >
      <div className="config-list">
        <div>
          <span>Type</span>
          <strong>{a.kind.replaceAll("_", " ")}</strong>
        </div>
        <div>
          <span>Size</span>
          <strong>{a.bytes.toLocaleString()} bytes</strong>
        </div>
        <div>
          <span>Created</span>
          <strong>{new Date(a.created_at).toLocaleString()}</strong>
        </div>
        <div>
          <span>Storage path</span>
          <strong className="mono hash">{a.path}</strong>
        </div>
      </div>
      <div className="detail-section">
        <span className="eyebrow">SHA-256 checksum</span>
        <p className="hash mono">{a.sha256}</p>
        <button
          className="button small-button"
          onClick={async () => {
            try {
              await navigator.clipboard.writeText(a.sha256);
              setCopied(true);
            } catch {
              setCopied(false);
            }
          }}
        >
          <Icon name={copied ? "check" : "copy"} size={15} />
          {copied ? "Copied" : "Copy checksum"}
        </button>
      </div>
      <a
        href={`/api/lab/campaigns/${campaign.id}/artifacts/${a.id}`}
        className="button primary full-button"
      >
        <Icon name="download" size={17} />
        Download verified artifact
      </a>
    </Dialog>
  );
}
