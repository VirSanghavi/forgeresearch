import type { Document } from "../../../packages/terse-actors/src/contracts.js";
import { digest } from "../../../packages/terse-actors/src/core.js";

/**
 * Deterministic research policy. Every choice is a function of measured development
 * evidence and declared limits; it never authorizes compute. Recipes use the campaign's
 * training method (`sft_lora` or reward-driven `grpo`) and differ in training data and
 * optimization schedule.
 */
export const POLICY_VERSION = "director-rules-v1";

export interface Measurement {
  label: string;
  experimentId: string | null;
  reportSha256: string;
  numExamples: number;
  correct: number;
  formatFailures: number;
  incorrect: number;
  limitHits: number;
  outcomes: Record<string, boolean>;
  predictionsSha256: string;
}
export type DataSelection = "full" | "concise";
export type Method = "sft_lora" | "grpo";
export interface Recipe {
  data: DataSelection;
  learningRate: number;
  maxSteps: number;
}
export type Operator = "explore" | "extend" | "raise_lr" | "lower_lr";
export interface Candidate {
  label: string;
  recipe: Recipe;
  operator: Operator;
  parent: string | null;
  hypothesis: string;
}
export interface PolicyLimits {
  minLearningRate: number;
  maxLearningRate: number;
  maxSteps: number;
  maxRevisions: number;
  maxFailures: number;
}
export interface Diagnosis {
  primary: "none" | "length" | "format" | "reasoning";
  numExamples: number;
  correct: number;
  formatFailures: number;
  incorrect: number;
  limitHits: number;
  summary: string;
}
export interface Rejection {
  option: string;
  reason: string;
}
export interface History {
  baseline: Measurement;
  measured: Measurement[];
  failed: string[];
  /** Candidates of the most recent round, in launch order. */
  round: Candidate[];
  revisions: number;
  remainingMicroUsd: number;
  experimentsRemaining: number;
}
export type Decision =
  | {
      kind: "launch";
      selected: Measurement | null;
      launch: Candidate[];
      rejected: Rejection[];
      summary: string;
    }
  | {
      kind: "stop";
      rule: string;
      selected: Measurement | null;
      rejected: Rejection[];
      summary: string;
    };

const score = (m: Measurement) => `${m.correct}/${m.numExamples}`;
const round3 = (value: number) => Number(value.toPrecision(3));

/** Reads the training worker's development report (`forge_numeric_v1` diagnoses). */
export function measure(
  label: string,
  experimentId: string | null,
  reportSha256: string,
  report: Document,
): Measurement {
  const samples = report.samples as Document[];
  const count = (diagnosis: string) => samples.filter((s) => s.diagnosis === diagnosis).length;
  return {
    label,
    experimentId,
    reportSha256,
    numExamples: Number(report.num_examples),
    correct: Number(report.num_correct),
    formatFailures: count("invalid_answer_format"),
    incorrect: count("incorrect_answer"),
    limitHits: samples.filter((s) => s.hit_generation_limit === true).length,
    outcomes: Object.fromEntries(samples.map((s) => [String(s.id), s.correct === true])),
    predictionsSha256: digest(samples.map((s) => [s.id, s.prediction])),
  };
}

export function diagnose(baseline: Measurement): Diagnosis {
  const failures = baseline.numExamples - baseline.correct;
  const primary =
    failures === 0
      ? "none"
      : baseline.limitHits * 2 >= failures
        ? "length"
        : baseline.formatFailures * 2 >= failures
          ? "format"
          : "reasoning";
  return {
    primary,
    numExamples: baseline.numExamples,
    correct: baseline.correct,
    formatFailures: baseline.formatFailures,
    incorrect: baseline.incorrect,
    limitHits: baseline.limitHits,
    summary:
      `Baseline ${score(baseline)}: ${baseline.formatFailures} invalid final-answer format, ` +
      `${baseline.incorrect} wrong answers, ${baseline.limitHits} hit the generation limit; ` +
      `primary failure mode: ${primary}.`,
  };
}

/** Higher is better: correct, then fewer format failures, then fewer truncations. */
export function compare(a: Measurement, b: Measurement): number {
  return (
    b.correct - a.correct ||
    a.formatFailures - b.formatFailures ||
    a.limitHits - b.limitHits ||
    a.label.localeCompare(b.label)
  );
}
const strictlyBetter = (a: Measurement, b: Measurement) =>
  a.correct > b.correct ||
  (a.correct === b.correct &&
    (a.formatFailures < b.formatFailures ||
      (a.formatFailures === b.formatFailures && a.limitHits < b.limitHits)));
function why(loser: Measurement, winner: Measurement): string {
  if (loser.correct !== winner.correct)
    return `Fewer correct: ${score(loser)} vs ${score(winner)} for ${winner.label}`;
  if (loser.formatFailures !== winner.formatFailures)
    return `Same accuracy; more format failures (${loser.formatFailures} vs ${winner.formatFailures})`;
  if (loser.limitHits !== winner.limitHits)
    return `Same accuracy and format; more truncations (${loser.limitHits} vs ${winner.limitHits})`;
  return `Tied on all measured criteria; ${winner.label} wins the deterministic label tie-break`;
}

const sftHypotheses: Record<DataSelection, Record<Diagnosis["primary"], string>> = {
  concise: {
    none: "Concise solutions preserve accuracy with shorter outputs.",
    length:
      "Training on the shortest solutions teaches the final '#### N' line within the generation limit.",
    format: "Short solutions make the required final-answer format the dominant pattern.",
    reasoning: "Concise solutions may underfit multi-step reasoning; included as a contrast.",
  },
  full: {
    none: "Full solutions preserve accuracy.",
    length: "Full solutions teach format but may keep outputs long; included as a contrast.",
    format: "Full worked solutions teach the format together with the reasoning steps.",
    reasoning: "Full worked solutions provide the most multi-step reasoning signal.",
  },
};

/** GRPO: rewards are 1 only for a world-verified final answer, 0 otherwise. */
const grpoHypotheses: Record<DataSelection, Record<Diagnosis["primary"], string>> = {
  concise: {
    none: "Rewarding verified answers on short problems preserves accuracy.",
    length:
      "On problems with the shortest solutions, sampled completions finish with a verified '#### N' more often, so groups carry reward signal.",
    format:
      "Short problems make a rewarded final-answer line likely enough to reinforce the required format.",
    reasoning:
      "Short problems give the densest correctness reward but may not exercise multi-step reasoning; included as a contrast.",
  },
  full: {
    none: "Rewarding verified answers on all problems preserves accuracy.",
    length:
      "All problems, rewarded only for verified final answers, push completions to finish within the limit; included as a contrast.",
    format: "Rewarding only verified final answers reinforces completions that end with '#### N'.",
    reasoning:
      "World-verified correctness rewards over all problems target wrong answers directly.",
  },
};
const hypotheses: Record<Method, typeof sftHypotheses> = {
  sft_lora: sftHypotheses,
  grpo: grpoHypotheses,
};

function label(round: number, operator: Operator, data: DataSelection): string {
  return `r${round}-${operator}-${data}`;
}

/** Two recipes, ordered by the diagnosed failure mode. Budget permitting, both run. */
export function initialDecision(
  baseline: Measurement,
  base: Recipe,
  costMicroUsd: number,
  history: Pick<History, "remainingMicroUsd" | "experimentsRemaining">,
  method: Method = "sft_lora",
): Decision {
  const diagnosis = diagnose(baseline);
  if (diagnosis.primary === "none")
    return {
      kind: "stop",
      rule: "BASELINE_SATISFIES_OBJECTIVE",
      selected: null,
      rejected: [],
      summary: "The baseline answers every development example; no training is justified.",
    };
  const order: DataSelection[] =
    diagnosis.primary === "reasoning" ? ["full", "concise"] : ["concise", "full"];
  const affordable = Math.min(
    history.experimentsRemaining,
    Math.floor(history.remainingMicroUsd / costMicroUsd),
  );
  if (affordable < 1)
    return {
      kind: "stop",
      rule: "BUDGET_EXHAUSTED",
      selected: null,
      rejected: [],
      summary: "No experiment fits the remaining budget or experiment limit.",
    };
  const candidates = order.map<Candidate>((data) => ({
    label: label(1, "explore", data),
    recipe: { ...base, data },
    operator: "explore",
    parent: null,
    hypothesis: hypotheses[method][data][diagnosis.primary],
  }));
  return {
    kind: "launch",
    selected: null,
    launch: candidates.slice(0, affordable),
    rejected: candidates.slice(affordable).map((c) => ({
      option: c.label,
      reason: "Deferred: exceeds the remaining budget or experiment limit",
    })),
    summary: `${diagnosis.summary} Explore ${candidates
      .slice(0, affordable)
      .map((c) => c.recipe.data)
      .join(" and ")} ${method === "grpo" ? "world-rewarded task" : "solution"} recipes.`,
  };
}

/**
 * Chooses how to revise the development-selected recipe from what was measured.
 * The rule order is fixed; the measurements decide which rule applies.
 */
export function revise(
  best: Measurement,
  bestRecipe: Recipe,
  baseline: Measurement,
  limits: PolicyLimits,
  round: number,
): { candidate: Candidate | null; rejected: Rejection[] } {
  const options: Record<Exclude<Operator, "explore">, Recipe> = {
    extend: { ...bestRecipe, maxSteps: bestRecipe.maxSteps * 2 },
    raise_lr: { ...bestRecipe, learningRate: round3(bestRecipe.learningRate * 3) },
    lower_lr: { ...bestRecipe, learningRate: round3(bestRecipe.learningRate / 3) },
  };
  let operator: Exclude<Operator, "explore">, reason: string;
  if (best.correct < baseline.correct) {
    operator = "lower_lr";
    reason = `Best candidate regressed (${score(best)} vs baseline ${score(baseline)}); reduce the step size.`;
  } else if (best.predictionsSha256 === baseline.predictionsSha256) {
    operator = "raise_lr";
    reason =
      "Predictions are identical to the baseline; the update was too small to change behavior.";
  } else {
    operator = "extend";
    reason =
      best.correct > baseline.correct
        ? `Accuracy improved (${score(baseline)} to ${score(best)}); continue optimizing the same recipe.`
        : `Outputs changed without an accuracy gain (format failures ${baseline.formatFailures} to ${best.formatFailures}); train longer.`;
  }
  const rejected: Rejection[] = Object.keys(options)
    .filter((name) => name !== operator)
    .map((name) => ({ option: `${name}:${best.label}`, reason: `Not indicated: ${reason}` }));
  const recipe = options[operator];
  const feasible =
    recipe.maxSteps <= limits.maxSteps &&
    recipe.learningRate >= limits.minLearningRate &&
    recipe.learningRate <= limits.maxLearningRate;
  if (!feasible) {
    rejected.push({ option: `${operator}:${best.label}`, reason: "Outside policy limits" });
    return { candidate: null, rejected };
  }
  return {
    candidate: {
      label: label(round, operator, recipe.data),
      recipe,
      operator,
      parent: best.label,
      hypothesis: reason,
    },
    rejected,
  };
}

/** After every launched candidate of a round is measured or failed. */
export function nextDecision(
  history: History,
  recipes: Record<string, Recipe>,
  limits: PolicyLimits,
  costMicroUsd: number,
  round: number,
): Decision {
  const ranked = [...history.measured].sort(compare);
  const best = ranked[0] ?? null;
  const rejected: Rejection[] = ranked
    .slice(1)
    .map((m) => ({ option: m.label, reason: why(m, best!) }));
  for (const label of history.failed)
    rejected.push({ option: label, reason: "Training or evaluation failed" });
  const stop = (rule: string, summary: string): Decision => ({
    kind: "stop",
    rule,
    selected: best,
    rejected,
    summary,
  });
  if (history.failed.length >= limits.maxFailures)
    return stop("FAILURE_LIMIT", `${history.failed.length} experiments failed.`);
  if (!best) return stop("NO_MEASUREMENTS", "No experiment produced a development measurement.");
  const latest = history.round.filter((c) => c.operator !== "explore");
  if (latest.length) {
    const revision = history.measured.find((m) => m.label === latest[0].label);
    const previous = ranked.filter((m) => m.label !== latest[0].label)[0];
    if (!revision || !previous || !strictlyBetter(revision, previous))
      return stop(
        "NO_FURTHER_IMPROVEMENT",
        `Revision ${latest[0].label} did not beat ${previous?.label ?? "earlier candidates"}; ` +
          `development-selected candidate is ${best.label} (${score(best)}).`,
      );
  }
  if (history.revisions >= limits.maxRevisions)
    return stop("REVISION_LIMIT", `Reached ${limits.maxRevisions} revisions.`);
  if (history.experimentsRemaining < 1)
    return stop("EXPERIMENT_LIMIT", "The campaign experiment limit is reached.");
  if (history.remainingMicroUsd < costMicroUsd)
    return stop("BUDGET_EXHAUSTED", "The remaining budget cannot fund another experiment.");
  const revision = revise(best, recipes[best.label], history.baseline, limits, round);
  rejected.push(...revision.rejected);
  if (!revision.candidate)
    return stop("SEARCH_SPACE_EXHAUSTED", "The indicated revision is outside policy limits.");
  return {
    kind: "launch",
    selected: best,
    launch: [revision.candidate],
    rejected,
    summary: `Selected ${best.label} (${score(best)}). ${revision.candidate.hypothesis}`,
  };
}
