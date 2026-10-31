import { afterAll, beforeAll, expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import type { Document } from "../../../packages/terse-actors/src/contracts.js";
import { materialize, selectRecords, verifyRecipe } from "../src/recipes.js";

const limits = {
  minLearningRate: 0.00002,
  maxLearningRate: 0.002,
  maxSteps: 128,
  maxRevisions: 2,
  maxFailures: 2,
};
const row = (id: number, answer: string) =>
  JSON.stringify({ id: `train-${id}`, question: `Q${id}?`, answer });
const source = [
  row(1, "long reasoning then\n#### 1"),
  row(2, "#### 2"),
  row(3, "medium\n#### 3"),
  row(4, "#### 4"),
  row(5, "a much longer worked solution\n#### 5"),
].join("\n");
const template: Document = {
  campaign_id: "recipes",
  experiment_id: "baseline",
  model: { source: "hub", model_id: "org/model", revision: "a".repeat(40) },
  train: { records: "data/train.jsonl", manifest: "data/train.manifest.json" },
  development: { records: "data/dev.jsonl", manifest: "data/dev.manifest.json" },
  training: { learning_rate: 0.0002, max_steps: 32, lora_rank: 8 },
  evaluation: { max_new_tokens: 256 },
  seed: 42,
  device: "mps",
  output_dir: "unused",
};
let dir: string;
let sourcePath: string;
let manifestPath: string;
let config: Document;
const recipe = { data: "concise" as const, learningRate: 0.0006, maxSteps: 64 };

beforeAll(async () => {
  dir = await mkdtemp(join(tmpdir(), "forge-recipes-test-"));
  sourcePath = join(dir, "train.jsonl");
  manifestPath = join(dir, "train.manifest.json");
  await writeFile(sourcePath, `${source}\n`);
  await writeFile(
    manifestPath,
    JSON.stringify({
      dataset_id: "train",
      purpose: "train",
      num_examples: 5,
      content_sha256: createHash("sha256").update(`${source}\n`).digest("hex"),
    }),
  );
  await writeFile(join(dir, "template.json"), JSON.stringify(template));
  const path = await materialize({
    directory: join(dir, "exp"),
    template,
    templatePath: join(dir, "template.json"),
    source: { records: sourcePath, manifest: manifestPath },
    campaignId: "recipes",
    experimentId: "exp-1",
    recipe,
  });
  config = JSON.parse(await readFile(path, "utf8"));
});
afterAll(async () => {
  if (dir) await rm(dir, { recursive: true, force: true });
});

const verify = (overrides: Partial<Parameters<typeof verifyRecipe>[0]> = {}) =>
  verifyRecipe({
    recipe,
    limits,
    source: { records: sourcePath, manifest: manifestPath },
    template,
    validatedConfig: config,
    trainingRecordsPath: join(dir, "exp/data/train.jsonl"),
    ...overrides,
  });

test("concise selection keeps the shorter half of exact source lines, in source order", () => {
  expect(selectRecords(`${source}\n`, "full")).toBe(`${source}\n`);
  const ids = selectRecords(`${source}\n`, "concise")
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line).id);
  expect(ids).toEqual(["train-2", "train-3", "train-4"]);
});

test("materialized config changes only recipe fields and rebases template paths", async () => {
  expect(config.training).toEqual({ learning_rate: 0.0006, max_steps: 64, lora_rank: 8 });
  expect(config.development).toEqual({
    records: "../data/dev.jsonl",
    manifest: "../data/dev.manifest.json",
  });
  expect(config.experiment_id).toBe("exp-1");
  const manifest = JSON.parse(await readFile(join(dir, "exp/data/train.manifest.json"), "utf8"));
  expect(manifest).toMatchObject({
    dataset_id: "train-concise",
    purpose: "train",
    num_examples: 3,
  });
  expect(await verify()).toEqual([]);
});

test("the reviewer rejects any deviation from the ledger recipe or campaign template", async () => {
  expect(await verify({ recipe: { ...recipe, data: "full" } })).toContain(
    "training records are not the recipe's selection from the allowed partition",
  );
  expect(await verify({ recipe: { ...recipe, learningRate: 0.0002 } })).toContain(
    "learning rate differs",
  );
  expect(await verify({ recipe: { ...recipe, maxSteps: 512 } })).toEqual([
    "recipe outside policy limits",
    "step count differs",
  ]);
  const training = config.training as Document;
  expect(
    await verify({ validatedConfig: { ...config, training: { ...training, lora_rank: 64 } } }),
  ).toEqual(["training settings differ from the campaign template"]);
  expect(
    await verify({ validatedConfig: { ...config, evaluation: { max_new_tokens: 1024 } } }),
  ).toEqual(["evaluation differs from the frozen campaign protocol"]);
  const tampered = join(dir, "tampered.jsonl");
  await writeFile(tampered, `${row(2, "#### 2")}\n${row(4, "#### 4")}\n${row(9, "#### 99")}\n`);
  expect(await verify({ trainingRecordsPath: tampered })).toEqual([
    "training records are not the recipe's selection from the allowed partition",
  ]);
});

test("absolute model and development paths still point to the template inputs", async () => {
  const directory = join(dir, "absolute");
  const modelPath = join(dir, "base-model");
  const development = {
    records: join(dir, "data/dev.jsonl"),
    manifest: join(dir, "data/dev.manifest.json"),
  };
  const path = await materialize({
    directory,
    template: { ...template, model: { source: "local", model_id: modelPath }, development },
    templatePath: join(dir, "template.json"),
    source: { records: sourcePath, manifest: manifestPath },
    campaignId: "recipes",
    experimentId: "absolute",
    recipe,
  });
  const actual = JSON.parse(await readFile(path, "utf8"));
  expect(resolve(directory, actual.model.model_id)).toBe(modelPath);
  expect(resolve(directory, actual.development.records)).toBe(development.records);
  expect(resolve(directory, actual.development.manifest)).toBe(development.manifest);
});

test("source checksum mismatches cannot be laundered into a new recipe manifest", async () => {
  const records = join(dir, "changed-source.jsonl");
  await writeFile(records, `${source.replace("#### 1", "#### 999")}\n`);
  const changed = { records, manifest: manifestPath };
  await expect(
    materialize({
      directory: join(dir, "changed"),
      template,
      templatePath: join(dir, "template.json"),
      source: changed,
      campaignId: "recipes",
      experimentId: "changed",
      recipe,
    }),
  ).rejects.toThrow("Source partition checksum mismatch");
  await expect(verify({ source: changed })).rejects.toThrow("Source partition checksum mismatch");
});

test("redelivery reuses matching recipe files but never overwrites different evidence", async () => {
  const options = {
    directory: join(dir, "redelivery"),
    template,
    templatePath: join(dir, "template.json"),
    source: { records: sourcePath, manifest: manifestPath },
    campaignId: "recipes",
    experimentId: "redelivery",
    recipe,
  };
  const path = await materialize(options);
  const expected = await readFile(path, "utf8");
  await rm(path); // Interrupted before the final config was committed.
  expect(await materialize(options)).toBe(path);
  expect(await readFile(path, "utf8")).toBe(expected);
  await expect(materialize({ ...options, recipe: { ...recipe, data: "full" } })).rejects.toThrow(
    "RECIPE_FILE_CONFLICT",
  );
  expect(await readFile(path, "utf8")).toBe(expected);
});
