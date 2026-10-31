"use client";
import {
  Campaign,
  Event,
  Experiment,
  TERMINAL,
  agentNames,
  duration,
  hardwareLabel,
  trainingSecondsLeft,
} from "@/lib/types";
import { Icon } from "./icon";
import { Empty, PanelHeader, Status } from "./views";

const method = (value: string | null) =>
  value === "grpo"
    ? "Graded practice · GRPO"
    : value === "targeted_sft"
      ? "Focused examples · SFT"
      : value === "broad_sft"
        ? "Broad examples · SFT"
        : "Measuring the original model";
const reason: Record<string, string> = {
  REWARD_FUNCTION_MODIFIED: "Reward modification rejected",
  SEALED_DATA_ACCESS: "Protected test access denied",
  REGRESSION_THRESHOLD_EXCEEDED: "Regression limit exceeded",
  NO_WEIGHT_UPDATE: "No model weight change",
  ARTIFACT_INTEGRITY_FAILURE: "Artifact integrity check failed",
  CHECKPOINT_NOT_RELOADED: "Independent reload required",
};
const count = (value?: { correct: number; n: number }) =>
  value ? `${value.correct} / ${value.n}` : "Awaiting";
const claim = {
  validated: "Validated improvement: the 95% paired interval is above zero.",
  uncertain:
    "A higher score, but within uncertainty. No validated improvement is claimed.",
  none: "No improvement on these failures. No validated improvement is claimed.",
};
const signed = (n: number) => `${n > 0 ? "+" : ""}${n.toFixed(1)}`;

function ProductionCheck({ campaign: c }: { campaign: Campaign }) {
  const p = c.production;
  if (!p) return null;
  const verdict = c.final?.production;
  const base = c.production_baseline;
  const best = c.best?.production || undefined;
  const n = p.held_out.length;
  const stopped = TERMINAL.includes(c.status) && !verdict;
  const before = verdict?.baseline ?? base;
  const after = verdict?.checkpoint ?? best;
  const fixed = verdict
    ? verdict.fixed_ids.length
    : base && best
      ? best.predictions.filter(
          (a) =>
            a.correct &&
            base.predictions.some((b) => b.id === a.id && !b.correct),
        ).length
      : null;
  return (
    <section
      className="briefing-final briefing-production panel"
      aria-label="Held-out production failures"
    >
      <div>
        <span>
          Production failures · {p.train.length} trained on ·{" "}
          {p.held_out.length} held out
        </span>
        <h2>
          {!n ? (
            "No held-out failures to measure"
          ) : before && after ? (
            <>
              Original {count(before)} <Icon name="arrow" size={20} /> Candidate{" "}
              {count(after)}
            </>
          ) : before ? (
            <>Before {count(before)}</>
          ) : stopped ? (
            "Held-out failures were not measured"
          ) : (
            `Measuring the ${n} held-out failures before training`
          )}
        </h2>
      </div>
      <p>
        {!n ? (
          `None of the ${p.flagged} flagged answers had a reference answer to hold out, so before and after cannot be compared. Source: ${p.name}.`
        ) : verdict ? (
          <>
            <strong>
              {verdict.fixed_ids.length} of {verdict.baseline.n} held-out
              failures fixed.
            </strong>{" "}
            {claim[verdict.improvement_claim]} 95% paired interval{" "}
            {signed(verdict.paired_bootstrap_95_pp[0])} to{" "}
            {signed(verdict.paired_bootstrap_95_pp[1])} percentage points.
          </>
        ) : fixed !== null ? (
          <>
            <strong>
              {fixed} of {best?.n} fixed by {c.best?.id}.
            </strong>{" "}
            A development-stage reading. The verdict comes with the final
            evaluation.
          </>
        ) : before && !stopped ? (
          `Candidate not measured yet: each approved checkpoint is measured on the same ${n} failures. Source: ${p.name}.`
        ) : (
          `${n} held-out production failures, never trained on, measured before and after. Source: ${p.name}.`
        )}
      </p>
    </section>
  );
}

export function Briefing({
  campaign: c,
  events,
  onInspect,
}: {
  campaign: Campaign | null;
  events: Event[];
  onInspect: (experiment: Experiment) => void;
}) {
  const b = c?.briefing;
  if (!c || !b)
    return (
      <section className="panel">
        <Empty
          icon="target"
          title="Give the lab a task. Keep the decisions in view."
          description="Choose the verified mathematical reasoning template, define your goal and set a budget. Forge measures the baseline, selects experiments and reports the evidence here."
        />
      </section>
    );
  const experiment = c.experiments.find((e) => e.id === b.experiment_id);
  const latest = events.findLast((e) => e.kind === "message" && e.relay_id);
  const progress = c.progress;
  const left = trainingSecondsLeft(c);
  const oversight = b.oversight;
  const policy = c.actor?.selection_frozen
    ? "Selection frozen before final evaluation"
    : "Final questions withheld from the research director";
  return (
    <div className="briefing-view">
      <section
        className="briefing-summary"
        aria-label="Research status at a glance"
      >
        <div>
          <span>Training run</span>
          <strong>
            {b.experiment_number || "Baseline"}
            {b.experiment_number > 0 && <small>of {b.experiment_limit}</small>}
          </strong>
          <p>{method(b.strategy)}</p>
        </div>
        <div>
          <span>Before training · development</span>
          <strong>{count(c.baseline)}</strong>
          <p>
            {c.baseline
              ? "Correct answers on the fixed development set"
              : "Baseline measurement in progress"}
          </p>
        </div>
        <div>
          <span>Best approved · development</span>
          <strong className={c.best ? "accent-text" : ""}>
            {c.best || !TERMINAL.includes(c.status)
              ? count(c.best?.evaluation)
              : c.final
                ? "Original"
                : "None approved"}
          </strong>
          <p>
            {c.best
              ? `${c.best.id} · development selection only`
              : c.final
                ? "No checkpoint beat the original, so it was kept"
                : TERMINAL.includes(c.status)
                  ? "The campaign ended before a checkpoint was approved"
                  : "A candidate needs evidence and approval"}
          </p>
        </div>
      </section>
      <section className="briefing-now" aria-label="Current operation">
        <div>
          <span className="static-dot" />
          <strong>
            {c.status === "completed"
              ? "Research complete"
              : c.status === "training"
                ? "Updating model weights"
                : c.status === "baseline"
                  ? "Measuring before training"
                  : c.status === "finalizing"
                    ? "Independent final evaluation"
                    : c.status === "evaluating"
                      ? "Testing a fresh checkpoint reload"
                      : c.status === "diagnosing"
                        ? "Choosing the next experiment"
                        : c.status === "reviewing"
                          ? "Reviewing checkpoint evidence"
                          : c.status.replaceAll("_", " ")}
          </strong>
        </div>
        {progress ? (
          <span className="mono">
            {progress.done} / {progress.total}{" "}
            {c.status === "training" ? "optimizer steps" : "questions measured"}
            {left !== null && (
              <span className="briefing-eta">
                {" "}
                · about {duration(Math.max(1, left))} left at this pace
              </span>
            )}
          </span>
        ) : (
          <Status value={c.status} />
        )}
        {progress && (
          <div
            className="briefing-progress"
            aria-label={`${progress.done} of ${progress.total} completed`}
          >
            <i
              style={{
                width: `${Math.min(100, (progress.done / progress.total) * 100)}%`,
              }}
            />
          </div>
        )}
      </section>
      {c.final && (
        <section
          className="briefing-final panel"
          aria-label="Final held-out check"
        >
          <div>
            <span>Final held-out check</span>
            <h2>
              Original {count(c.final.baseline)} <Icon name="arrow" size={20} />{" "}
              Candidate {count(c.final.checkpoint)}
            </h2>
          </div>
          <p>
            <strong>
              {c.final.improvement_pp > 0 ? "+" : ""}
              {c.final.improvement_pp.toFixed(1)} percentage points.
            </strong>{" "}
            A subset estimate, with uncertainty in the report.{" "}
            {c.final.valid_improvement
              ? "An observed gain on these questions."
              : "Retain the original model; no validated improvement was found."}
          </p>
        </section>
      )}
      <ProductionCheck campaign={c} />
      <div className="briefing-desks">
        <section className="panel briefing-desk">
          <PanelHeader
            title="What the research director sees"
            icon="target"
            right={<span className="tag">Development evidence</span>}
          />
          <div className="briefing-desk-body">
            <div className="briefing-goal">
              <span>Your task</span>
              <p>{c.config.objective}</p>
            </div>
            <div className="briefing-hypothesis">
              <span>Why this experiment</span>
              <p>
                {b.hypothesis ||
                  "The director waits for measured baseline failures before choosing a strategy."}
              </p>
            </div>
            <div className="briefing-inputs">
              <Icon name="artifacts" size={16} />
              <p>
                {b.director.input_recorded ? (
                  <>
                    <strong>
                      {b.director.failure_examples} failure examples
                    </strong>{" "}
                    and{" "}
                    <strong>
                      {b.director.previous_results} prior experiment results
                    </strong>{" "}
                    were supplied to the director. The exact input is archived.
                  </>
                ) : c.experiments.length ? (
                  "This run predates input archiving. Its hypotheses and measurements remain in the experiment trace."
                ) : (
                  "The director will receive measured baseline failures before choosing the first experiment."
                )}
              </p>
            </div>
            <div className="briefing-boundary">
              <Icon name="governance" size={16} />
              <span>{policy}</span>
            </div>
            {experiment && (
              <button
                className="text-button briefing-inspect"
                onClick={() => onInspect(experiment)}
              >
                Inspect the experiment and its evidence{" "}
                <Icon name="arrow" size={15} />
              </button>
            )}
          </div>
        </section>
        <section className="panel briefing-desk">
          <PanelHeader
            title="What the overseer sees"
            icon="governance"
            right={<Status value={oversight.decision} />}
          />
          <div className="briefing-desk-body">
            <p className="briefing-review-scope">
              Current decision covers{" "}
              <strong>{oversight.decision_scope}</strong>.
            </p>
            <ul className="briefing-checks">
              {oversight.checks.map((check) => (
                <li key={check.label}>
                  <Icon
                    name={
                      ["approved", "verified"].includes(check.state)
                        ? "check"
                        : check.state === "rejected"
                          ? "close"
                          : "clock"
                    }
                    size={15}
                  />
                  <span>{check.label}</span>
                  <strong data-state={check.state}>
                    {check.state === "pending"
                      ? "Awaiting evidence"
                      : check.state[0].toUpperCase() + check.state.slice(1)}
                  </strong>
                </li>
              ))}
            </ul>
            {oversight.last_block && (
              <div className="briefing-veto">
                <Icon name="alert" size={18} />
                <div>
                  <strong>
                    {oversight.blocked_proposals} proposal
                    {oversight.blocked_proposals === 1 ? "" : "s"} blocked
                  </strong>
                  <p>
                    {oversight.last_block.evidence.controlled_probe
                      ? "Controlled integrity test. "
                      : ""}
                    {oversight.last_block.reason_codes
                      .map(
                        (code) =>
                          reason[code] ||
                          code.replaceAll("_", " ").toLowerCase(),
                      )
                      .join(". ")}
                    .
                  </p>
                </div>
              </div>
            )}
            <p className="briefing-scope">{oversight.regression_scope}</p>
          </div>
        </section>
      </div>
      <section
        className="briefing-execution"
        aria-label="What the execution framework sees"
      >
        <span>
          <Icon name="environments" size={16} />
          <strong>{b.execution.terse_worlds}</strong> persisted Terse worlds
        </span>
        <span>
          <Icon name="message" size={16} />
          <strong>{b.execution.relay_deliveries}</strong> confirmed Relay
          deliveries
        </span>
        <span>
          <Icon name="chip" size={16} />
          <strong>{b.execution.rl_rollouts}</strong> real RL rollouts
        </span>
        <span>
          <Icon name="clock" size={16} />
          {duration(b.execution.settled_compute_seconds)} settled /{" "}
          {duration(b.execution.time_limit_seconds)} limit
        </span>
        {hardwareLabel(b.execution.hardware) && (
          <span>
            <Icon name="chip" size={16} />
            <strong>{hardwareLabel(b.execution.hardware)}</strong>
          </span>
        )}
      </section>
      {latest && (
        <section className="panel briefing-handoff">
          <div>
            <Icon name="message" size={18} />
            <strong>
              {agentNames[latest.agent]} <span>to</span>{" "}
              {agentNames[latest.recipient || "director"]}
            </strong>
            <time className="mono">
              {new Date(latest.timestamp).toLocaleTimeString("en-GB", {
                timeZone: "UTC",
              })}{" "}
              UTC
            </time>
          </div>
          <p>{latest.text}</p>
          <small>
            Actual Agent Relay message · recipient readback verified
          </small>
        </section>
      )}
      <p className="briefing-timestamp mono">
        Evidence as of{" "}
        {new Date(b.as_of).toLocaleTimeString("en-GB", { timeZone: "UTC" })} UTC
        · {c.id}
      </p>
    </div>
  );
}
