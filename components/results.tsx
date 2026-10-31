"use client";
import { useState } from "react";
import { Campaign, ResultFilter, duration, percent } from "@/lib/types";
import { Icon } from "./icon";
import { Empty, PanelHeader, Status } from "./views";
export function Results({ campaign }: { campaign: Campaign | null }) {
  const [filter, setFilter] = useState<ResultFilter>("all"),
    [limit, setLimit] = useState(6);
  if (!campaign?.final)
    return (
      <section className="panel">
        <Empty
          icon="chart"
          title="No sealed result yet"
          description="Final results appear only after checkpoint selection freezes and independent evaluation finishes."
        />
      </section>
    );
  const f = campaign.final;
  const cases = f.baseline.predictions
    .map((before) => ({
      before,
      after: f.checkpoint.predictions.find((p) => p.id === before.id)!,
    }))
    .filter((p) => p.after);
  const selected = cases.filter(
    (p) =>
      filter === "all" ||
      (filter === "improved"
        ? !p.before.correct && p.after.correct
        : p.before.correct && !p.after.correct),
  );
  const report = campaign.artifacts.find((a) => a.kind === "report");
  const checkpoint = campaign.artifacts.find(
    (a) =>
      a.kind === "checkpoint_zip" &&
      a.path.startsWith(campaign.best?.id || "no-candidate"),
  );
  return (
    <div className="results-view">
      <section className="panel result-hero">
        <div className="result-eyebrow">
          <Icon name="governance" size={16} />
          Selection frozen · sealed evaluation
        </div>
        <h2>
          {f.valid_improvement
            ? "A measurable step forward."
            : "A clear result. An honest conclusion."}
        </h2>
        <p>
          {f.valid_improvement
            ? "A new checkpoint, tested against the original model on held-out questions."
            : "The campaign found no validated improvement. The evidence and candidate artifacts remain available."}
        </p>
        <div className="result-scores">
          <div>
            <span>Original model</span>
            <strong>{percent(f.baseline.score)}</strong>
            <small>
              {f.baseline.correct} / {f.baseline.n} correct
            </small>
          </div>
          <Icon name="arrow" size={25} />
          <div>
            <span>Frozen candidate</span>
            <strong
              className={
                f.improvement_pp > 0
                  ? "accent-text"
                  : f.improvement_pp < 0
                    ? "warning-text"
                    : ""
              }
            >
              {percent(f.checkpoint.score)}
            </strong>
            <small>
              {f.checkpoint.correct} / {f.checkpoint.n} correct
            </small>
          </div>
          <div className="result-delta">
            <span>Absolute change</span>
            <strong>
              {f.improvement_pp > 0 ? "+" : ""}
              {f.improvement_pp.toFixed(1)}
              <small>pp</small>
            </strong>
            <small>Identical evaluation settings</small>
          </div>
        </div>
        <div className="result-actions">
          {report && (
            <a
              className="button primary"
              href={`/api/lab/campaigns/${campaign.id}/artifacts/${report.id}`}
            >
              <Icon name="download" size={16} />
              Download report
            </a>
          )}
          {checkpoint && (
            <a
              className="button"
              href={`/api/lab/campaigns/${campaign.id}/artifacts/${checkpoint.id}`}
            >
              <Icon name="chip" size={16} />
              {f.valid_improvement
                ? "Download checkpoint"
                : "Download development candidate"}
            </a>
          )}
          <a
            className="button"
            href={`/api/lab/campaigns/${campaign.id}/trace`}
          >
            <Icon name="code" size={16} />
            Full research trace
          </a>
        </div>
      </section>
      <div className="result-meta">
        <span>{campaign.experiments.length} experiments</span>
        <span>{duration(campaign.elapsed_seconds || 0)} elapsed</span>
        <span className="mono">{f.checkpoint_id}</span>
        <span>95% Wilson intervals in report</span>
      </div>
      <section className="panel">
        <PanelHeader
          title="Read the model's work"
          subtitle="The same questions. The original output. The independently tested candidate."
          right={<span className="tag">Sample-level evidence</span>}
        />
        <div className="table-tools">
          <label className="filter-label">
            Compare
            <select
              value={filter}
              onChange={(e) => {
                setFilter(e.target.value as ResultFilter);
                setLimit(6);
              }}
            >
              <option value="all">All predictions</option>
              <option value="improved">Improved answers</option>
              <option value="regressed">Regressed answers</option>
            </select>
          </label>
          <span className="mono muted small">
            {selected.length} / {cases.length} QUESTIONS
          </span>
        </div>
        <div className="paired-predictions">
          {selected.slice(0, limit).map(({ before, after }) => (
            <article key={before.id}>
              <header>
                <span className="mono muted">{before.id}</span>
                <span className="mono">Expected answer: {before.expected}</span>
              </header>
              <h3>{before.question}</h3>
              <div className="paired-outputs">
                <div>
                  <header>
                    <strong>Original model</strong>
                    <Status value={before.correct ? "correct" : "incorrect"} />
                  </header>
                  <p>{before.output}</p>
                </div>
                <div>
                  <header>
                    <strong>Frozen candidate</strong>
                    <Status value={after.correct ? "correct" : "incorrect"} />
                  </header>
                  <p>{after.output}</p>
                </div>
              </div>
            </article>
          ))}
          {!selected.length && (
            <Empty
              icon="chart"
              title="No predictions in this group"
              description="Choose another comparison to inspect the complete set."
            />
          )}
        </div>
        {limit < selected.length && (
          <div className="result-more">
            <button className="button" onClick={() => setLimit((n) => n + 6)}>
              Show 6 more questions
              <Icon name="down" size={15} />
            </button>
          </div>
        )}
      </section>
    </div>
  );
}
