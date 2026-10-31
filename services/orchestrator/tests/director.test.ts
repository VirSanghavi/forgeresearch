import { expect, test } from "bun:test";
import {
  compare,
  diagnose,
  initialDecision,
  nextDecision,
  revise,
  type Candidate,
  type History,
  type Measurement,
  type Recipe,
} from "../src/director.js";

// Pure policy checks: identical histories that differ only in measurements must be able
// to produce different decisions. No services or models are involved.
const limits = {
  minLearningRate: 0.00002,
  maxLearningRate: 0.002,
  maxSteps: 128,
  maxRevisions: 2,
  maxFailures: 2,
};
const base: Recipe = { data: "full", learningRate: 0.0002, maxSteps: 32 };
const cost = 100;
function m(label: string, fields: Partial<Measurement> = {}): Measurement {
  return {
    label,
    experimentId: label === "baseline" ? null : `exp-${label}`,
    reportSha256: "0".repeat(64),
    numExamples: 16,
    correct: 2,
    formatFailures: 10,
    incorrect: 4,
    limitHits: 9,
    outcomes: {},
    predictionsSha256: `predictions-${label}`,
    ...fields,
  };
}
const baseline = m("baseline");
const explore = (data: "full" | "concise"): Candidate => ({
  label: `r1-explore-${data}`,
  recipe: { ...base, data },
  operator: "explore",
  parent: null,
  hypothesis: "fixture",
});
const recipes: Record<string, Recipe> = {
  "r1-explore-full": { ...base, data: "full" },
  "r1-explore-concise": { ...base, data: "concise" },
};
function history(measured: Measurement[], fields: Partial<History> = {}): History {
  return {
    baseline,
    measured,
    failed: [],
    round: [explore("concise"), explore("full")],
    revisions: 0,
    remainingMicroUsd: 1000,
    experimentsRemaining: 2,
    ...fields,
  };
}

test("diagnosis names the dominant measured failure mode", () => {
  expect(diagnose(baseline).primary).toBe("length");
  expect(diagnose(m("b", { limitHits: 1, formatFailures: 10 })).primary).toBe("format");
  expect(diagnose(m("b", { limitHits: 1, formatFailures: 2, incorrect: 12 })).primary).toBe(
    "reasoning",
  );
  expect(diagnose(m("b", { correct: 16 })).primary).toBe("none");
});

test("the first decision explores two recipes ordered by diagnosis, within budget", () => {
  const capacity = { remainingMicroUsd: 1000, experimentsRemaining: 4 };
  const lengthBound = initialDecision(baseline, base, cost, capacity);
  expect(lengthBound.kind).toBe("launch");
  if (lengthBound.kind !== "launch") return;
  expect(lengthBound.launch.map((c) => c.recipe.data)).toEqual(["concise", "full"]);
  const reasoning = initialDecision(
    m("baseline", { limitHits: 0, formatFailures: 1, incorrect: 13 }),
    base,
    cost,
    capacity,
  );
  expect(reasoning.kind === "launch" && reasoning.launch.map((c) => c.recipe.data)).toEqual([
    "full",
    "concise",
  ]);
  const tight = initialDecision(baseline, base, cost, { ...capacity, remainingMicroUsd: 150 });
  expect(tight.kind === "launch" && tight.launch).toHaveLength(1);
  expect(tight.rejected[0].reason).toContain("budget");
  expect(
    initialDecision(baseline, base, cost, { ...capacity, remainingMicroUsd: 50 }),
  ).toMatchObject({ kind: "stop", rule: "BUDGET_EXHAUSTED" });
  expect(initialDecision(m("baseline", { correct: 16 }), base, cost, capacity)).toMatchObject({
    kind: "stop",
    rule: "BASELINE_SATISFIES_OBJECTIVE",
  });
});

test("the revision operator is determined by what the selected candidate measured", () => {
  const recipe = recipes["r1-explore-full"];
  const operator = (fields: Partial<Measurement>) =>
    revise(m("r1-explore-full", fields), recipe, baseline, limits, 2).candidate?.operator;
  expect(operator({ correct: 5 })).toBe("extend");
  expect(operator({ correct: 2, formatFailures: 4 })).toBe("extend");
  expect(operator({ ...baseline, label: "r1-explore-full" })).toBe("raise_lr");
  expect(operator({ correct: 1 })).toBe("lower_lr");
  const extended = revise(m("r1-explore-full", { correct: 5 }), recipe, baseline, limits, 2);
  expect(extended.candidate?.recipe).toEqual({ ...recipe, maxSteps: 64 });
  expect(extended.rejected.map((r) => r.option)).toEqual([
    "raise_lr:r1-explore-full",
    "lower_lr:r1-explore-full",
  ]);
  const capped = revise(
    m("r1-explore-full", { correct: 5 }),
    { ...recipe, maxSteps: 128 },
    baseline,
    limits,
    2,
  );
  expect(capped.candidate).toBeNull();
  expect(capped.rejected.at(-1)?.reason).toBe("Outside policy limits");
});

test("swapping only the round-1 results changes which recipe is revised", () => {
  const conciseWins = nextDecision(
    history([m("r1-explore-concise", { correct: 6 }), m("r1-explore-full", { correct: 3 })]),
    recipes,
    limits,
    cost,
    2,
  );
  const fullWins = nextDecision(
    history([m("r1-explore-concise", { correct: 3 }), m("r1-explore-full", { correct: 6 })]),
    recipes,
    limits,
    cost,
    2,
  );
  if (conciseWins.kind !== "launch" || fullWins.kind !== "launch")
    throw new Error("expected launch");
  expect(conciseWins.selected?.label).toBe("r1-explore-concise");
  expect(conciseWins.launch[0]).toMatchObject({
    label: "r2-extend-concise",
    parent: "r1-explore-concise",
    recipe: { data: "concise", maxSteps: 64 },
  });
  expect(fullWins.launch[0]).toMatchObject({ label: "r2-extend-full", parent: "r1-explore-full" });
  expect(conciseWins.rejected[0]).toEqual({
    option: "r1-explore-full",
    reason: "Fewer correct: 3/16 vs 6/16 for r1-explore-concise",
  });
});

test("stop rules: no further improvement, failures, revisions, experiments and budget", () => {
  const winner = m("r1-explore-full", { correct: 6 });
  const revision: Candidate = {
    label: "r2-extend-full",
    recipe: { ...base, maxSteps: 64 },
    operator: "extend",
    parent: "r1-explore-full",
    hypothesis: "fixture",
  };
  const after = (correct: number, fields: Partial<History> = {}) =>
    nextDecision(
      history([winner, m("r1-explore-concise", { correct: 3 }), m("r2-extend-full", { correct })], {
        round: [revision],
        revisions: 1,
        ...fields,
      }),
      { ...recipes, "r2-extend-full": revision.recipe },
      limits,
      cost,
      3,
    );
  expect(after(6)).toMatchObject({ kind: "stop", rule: "NO_FURTHER_IMPROVEMENT" });
  expect(after(6).selected?.label).toBe("r1-explore-full");
  const improved = after(8);
  expect(improved).toMatchObject({ kind: "launch", selected: { label: "r2-extend-full" } });
  expect(improved.kind === "launch" && improved.launch[0].label).toBe("r3-extend-full");
  expect(after(8, { revisions: 2 })).toMatchObject({ rule: "REVISION_LIMIT" });
  expect(after(8, { experimentsRemaining: 0 })).toMatchObject({ rule: "EXPERIMENT_LIMIT" });
  expect(after(8, { remainingMicroUsd: 99 })).toMatchObject({ rule: "BUDGET_EXHAUSTED" });
  expect(
    nextDecision(
      history([], { failed: ["r1-explore-concise", "r1-explore-full"] }),
      recipes,
      limits,
      cost,
      2,
    ),
  ).toMatchObject({ kind: "stop", rule: "FAILURE_LIMIT" });
});

test("ranking prefers accuracy, then format compliance, then fewer truncations", () => {
  const ranked = [
    m("a", { correct: 4, formatFailures: 3 }),
    m("b", { correct: 5, formatFailures: 9 }),
    m("c", { correct: 4, formatFailures: 1, limitHits: 5 }),
    m("d", { correct: 4, formatFailures: 1, limitHits: 2 }),
  ].sort(compare);
  expect(ranked.map((x) => x.label)).toEqual(["b", "d", "c", "a"]);
});
