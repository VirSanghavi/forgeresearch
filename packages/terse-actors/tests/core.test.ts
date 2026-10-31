import { expect, test } from "bun:test";
import { canonical, digest, numericAnswer, scopedId } from "../src/core.js";
import { campaignSchema, model, proposalSchema } from "../src/contracts.js";

test("numeric scoring preserves exact decimals without float coercion", () => {
  for (const [input, answer] of [
    ["+0042.000", "42"],
    ["-0.00", "0"],
    [".500", "0.5"],
    ["-000.1250", "-0.125"],
    ["9007199254740993", "9007199254740993"],
  ]) {
    expect(numericAnswer(input)).toBe(answer);
  }
  for (const invalid of ["NaN", "Infinity", "1e2", "42 extra", "0x2a", "1/2", "1".repeat(25)])
    expect(numericAnswer(invalid)).toBeNull();
});
test("canonical hashes ignore key order, reject non-finite values and isolate actor IDs", () => {
  expect(digest({ a: 1, b: [true, null] })).toBe(digest({ b: [true, null], a: 1 }));
  expect(() => canonical({ value: Number.NaN })).toThrow("INVALID_JSON");
  expect(scopedId("exp", "one", "same")).not.toBe(scopedId("exp", "two", "same"));
  expect(() => scopedId("world", "one", "../bad")).toThrow();
});
test("strict schemas reject mutable model revisions and non-integer costs", () => {
  expect(model.safeParse({ modelId: "m", revision: "main" }).success).toBe(false);
  expect(model.safeParse({ modelId: "m", revision: "a".repeat(41) }).success).toBe(false);
  const proposal = {
    experimentId: "test",
    strategy: "sft_lora",
    model: { modelId: "m", revision: "a".repeat(40) },
    configSha256: digest("c"),
    trainingDatasetSha256: digest("t"),
    developmentDatasetSha256: digest("d"),
    hypothesis: "h",
    maxCostMicroUsd: 1,
  };
  expect(proposalSchema.safeParse(proposal).success).toBe(true);
  for (const value of [-1, 0.5, Number.MAX_SAFE_INTEGER, Infinity])
    expect(proposalSchema.safeParse({ ...proposal, maxCostMicroUsd: value }).success).toBe(false);
  expect(proposalSchema.safeParse({ ...proposal, arbitraryCode: "ignored?" }).success).toBe(false);
  expect(campaignSchema.safeParse({}).success).toBe(false);
});
