import { afterAll, beforeAll, expect, test } from "bun:test";
import { mkdtemp, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Document } from "../../../packages/terse-actors/src/contracts.js";
import { chooseEvaluationBatch, startingEvaluation } from "../src/evaluations.js";
import { pythonJson, readJson } from "../src/worker.js";

// Real Python evaluations of the offline tiny fixture model; no services involved.
const python = process.env.FORGE_PYTHON;
let directory: string;
let templatePath: string;
const cacheDir = () => join(directory, "cache");

beforeAll(async () => {
  if (!python) throw new Error("Set FORGE_PYTHON to the training-core virtualenv Python");
  directory = await mkdtemp(join(tmpdir(), "forge-evaluations-test-"));
  await pythonJson(python, [
    "-c",
    "import sys,json; from pathlib import Path; from forge_training.fixtures import make_smoke_fixture; print(json.dumps({'config':str(make_smoke_fixture(Path(sys.argv[1])))}))",
    join(directory, "fixture"),
  ]);
  templatePath = join(directory, "fixture", "experiment.json");
  const template = await readJson(templatePath);
  await writeFile(
    templatePath,
    JSON.stringify({
      ...template,
      evaluation: { ...(template.evaluation as Document), batch_size: 2 },
    }),
  );
}, 120000);
afterAll(async () => {
  if (directory) await rm(directory, { recursive: true, force: true });
});

test("a batched campaign protocol is adopted only if it matches batch size one", async () => {
  const choice = await chooseEvaluationBatch({
    python: python!,
    templatePath,
    directory,
    cacheDir: cacheDir(),
    timeoutMs: 120000,
  });
  expect(choice.requested).toBe(2);
  expect(choice.numExamples).toBe(2);
  expect(choice.baseline.cached).toBe(false);
  expect(choice.reference?.cached).toBe(false);
  // Either outcome is valid; the choice must follow the measured agreement.
  expect(choice.used).toBe(choice.identicalPredictions === choice.numExamples ? 2 : 1);
  const adopted = await readJson(choice.baseline.path);
  expect((adopted.protocol as Document).evaluation).toMatchObject({ batch_size: choice.used });
  const used = await readJson(choice.templatePath);
  expect((used.evaluation as Document).batch_size).toBe(choice.used);
}, 300000);

test("identical untrained evaluations are served from the cache, byte for byte", async () => {
  const first = join(directory, "frozen-baseline-batch2.json");
  const again = await startingEvaluation({
    python: python!,
    configPath: templatePath,
    output: join(directory, "again.json"),
    cacheDir: cacheDir(),
    timeoutMs: 120000,
  });
  expect(again.cached).toBe(true);
  expect(await readFile(again.path)).toEqual(await readFile(first));
}, 120000);

test("a cache entry that does not match its key is refused", async () => {
  const [entry] = (await readdir(cacheDir())).filter((name) => name.endsWith(".json"));
  const path = join(cacheDir(), entry);
  const report = await readJson(path);
  const files = await readdir(cacheDir());
  for (const name of files.filter((n) => n.endsWith(".json"))) {
    const other = await readJson(join(cacheDir(), name));
    await writeFile(
      join(cacheDir(), name),
      JSON.stringify({ ...other, protocol: { ...(other.protocol as Document), seed: 7 } }),
    );
  }
  await expect(
    startingEvaluation({
      python: python!,
      configPath: templatePath,
      output: join(directory, "conflict.json"),
      cacheDir: cacheDir(),
      timeoutMs: 120000,
    }),
  ).rejects.toThrow("EVALUATION_CACHE_CONFLICT");
  expect(report.checkpoint_sha256).toBeNull();
}, 120000);
