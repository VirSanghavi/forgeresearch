import { afterAll, beforeAll, expect, test } from "bun:test";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { LocalActorRuntime } from "durable-actors/dev";
import {
  clients,
  launch,
  localAuthority,
  type LocalAuthority,
} from "../../../packages/terse-actors/scripts/local-runtime.js";
import type { Document } from "../../../packages/terse-actors/src/contracts.js";
import { scopedId } from "../../../packages/terse-actors/src/core.js";
import {
  WorldGateway,
  loadWorldTasks,
  referenceAnswer,
  verifyWorldRollouts,
  type WorldGatewayOptions,
} from "../src/worlds.js";
import { pythonJson, readJson } from "../src/worker.js";

// Real native Terse runtime and real tiny-model GRPO. CONTROLLED FAILURE INJECTION below:
// lost step replies and a SIGKILL of the Terse runtime while rollouts are being scored.
const python = process.env.FORGE_PYTHON!;
const campaign = "grpo-mechanics";
let output: string, fixture: string;
let authority: LocalAuthority;
let runtime: LocalActorRuntime;
let actors: ReturnType<typeof clients>;
let world: Awaited<ReturnType<typeof loadWorldTasks>>;

async function fresh(name: string): Promise<string> {
  const directory = join(output, name);
  await pythonJson(python, [
    "-c",
    "import sys,json; from pathlib import Path; from forge_training.fixtures import make_grpo_fixture; print(json.dumps({'config':str(make_grpo_fixture(Path(sys.argv[1])))}))",
    directory,
  ]);
  return directory;
}
function gatewayFor(jobId: string, extra: Partial<WorldGatewayOptions> = {}): WorldGateway {
  return new WorldGateway({
    campaignId: campaign,
    experimentId: "tiny-grpo",
    jobId,
    attempt: 1,
    tasks: world.tasks,
    taskVersion: world.taskVersion,
    seed: 42,
    rewardSecret: authority.rewardSecret,
    actors: () => ({ data: actors.data, worker: actors.worker }),
    rolloutsPerUpdate: 8,
    maxUpdates: 6,
    lanes: 4,
    ...extra,
  });
}
async function train(directory: string, gateway: WorldGateway): Promise<Document> {
  return pythonJson(
    python,
    ["-m", "forge_training", "run", "--config", join(directory, "experiment.json")],
    { timeoutMs: 300000, env: gateway.environment() },
  );
}
const post = (gateway: WorldGateway, body: Document, token = gateway.token) =>
  gateway.handle(
    new Request("http://127.0.0.1/v1/rollouts", {
      method: "POST",
      headers: { authorization: `Bearer ${token}` },
      body: JSON.stringify(body),
    }),
  );
const batch = (update: number, answers: string[], taskId = "train-1") => ({
  schemaVersion: 1,
  campaignId: campaign,
  experimentId: "tiny-grpo",
  update,
  policySha256: "0".repeat(64),
  rollouts: answers.map((answer, index) => ({
    rolloutId: `u${String(update).padStart(5, "0")}-r${String(index).padStart(3, "0")}`,
    taskId,
    action: { type: "submit_answer", answer },
    completionSha256: "1".repeat(64),
  })),
});

beforeAll(async () => {
  if (!python) throw new Error("Set FORGE_PYTHON to the training-core virtualenv Python");
  output = await mkdtemp(join(tmpdir(), "forge-worlds-test-"));
  fixture = await fresh("fixture");
  world = await loadWorldTasks({
    sourceRecords: join(fixture, "data/train.jsonl"),
    sourceManifest: join(fixture, "data/train.manifest.json"),
    tasksRecords: join(fixture, "data/tasks.jsonl"),
    tasksManifest: join(fixture, "data/tasks.manifest.json"),
  });
  authority = localAuthority([campaign]);
  runtime = await launch(authority, join(output, "terse"));
  actors = clients(authority, runtime);
}, 240000);
afterAll(async () => {
  await runtime?.stop();
  if (output) await rm(output, { recursive: true, force: true });
});

test("the task service holds answers only for allowed, answer-free tasks", async () => {
  expect(world.tasks.map((task) => [task.taskId, task.answer])).toEqual([
    ["train-1", "2"],
    ["train-3", "4"],
    ["train-5", "6"],
    ["train-7", "8"],
  ]);
  expect(referenceAnswer("Work.\n#### 1,234\n")).toBe("1234");
  const tasks = join(output, "tasks-with-answers.jsonl");
  await writeFile(tasks, '{"id":"train-1","question":"What is 1 plus 1?","answer":"#### 2"}\n');
  await expect(
    loadWorldTasks({
      sourceRecords: join(fixture, "data/train.jsonl"),
      sourceManifest: join(fixture, "data/train.manifest.json"),
      tasksRecords: tasks,
      tasksManifest: join(fixture, "data/tasks.manifest.json"),
    }),
  ).rejects.toThrow("TASK_SOURCE_MISMATCH");
});

test("concurrent and repeated scoring of one update applies each transition once", async () => {
  const gateway = gatewayFor("probe");
  const answers = ["2", "1", "2.0", "+02", "none", "3", "2", "1,0"];
  const [first, second] = await Promise.all([
    post(gateway, batch(1, answers)),
    post(gateway, batch(1, answers)),
  ]);
  expect(first.status).toBe(200);
  const results = ((await first.json()) as Document).results as Document[];
  expect(((await second.json()) as Document).results).toEqual(results);
  // Exact decimal rewards from the actor; "none" and a comma-grouped answer score 0.
  expect(results.map((r) => r.reward)).toEqual([1, 0, 1, 1, 0, 0, 1, 0]);
  expect(new Set(results.map((r) => r.worldId)).size).toBe(4);
  expect(gateway.stats.transitions).toBe(8);
  const repeat = await post(gateway, batch(1, answers));
  expect(((await repeat.json()) as Document).results).toEqual(results);
  expect(gateway.stats.transitions).toBe(8);
  // Refusals: changed rollouts, skipped update, wrong token, foreign task, wrong batch size.
  expect((await post(gateway, batch(1, ["9", ...answers.slice(1)]))).status).toBe(409);
  expect((await post(gateway, batch(3, answers))).status).toBe(409);
  expect((await post(gateway, batch(2, answers), "f".repeat(64))).status).toBe(401);
  expect((await post(gateway, batch(2, answers, "development-2"))).status).toBe(400);
  expect((await post(gateway, batch(2, answers.slice(1)))).status).toBe(400);
});

test("world-rewarded GRPO survives lost replies and a runtime SIGKILL exactly once", async () => {
  const directory = await fresh("interrupted");
  let killed = false;
  const gateway = gatewayFor("tiny", {
    retryMs: 90000,
    faults: {
      // FAULT: two applied steps lose their replies and are resent.
      loseStepReplies: 2,
      // FAULT: SIGKILL the Terse runtime after the 20th transition, then restart it.
      afterTransition: async (count) => {
        if (count !== 20 || killed) return;
        killed = true;
        const port = Number(new URL(runtime.connection.controlPlaneUrl).port);
        const closed = runtime.closed.catch(() => undefined);
        process.kill(runtime.connection.pid, "SIGKILL");
        await closed;
        runtime = await launch(authority, join(output, "terse"), true, port);
      },
    },
  }).start();
  let result: Document;
  try {
    result = await train(directory, gateway);
  } finally {
    await gateway.stop();
  }
  expect(result.status).toBe("completed");
  expect(killed).toBe(true);
  expect(gateway.stats.lostReplies).toBe(2);
  expect(gateway.stats.retries).toBeGreaterThan(0);
  const runDir = join(directory, "run");
  const verified = await pythonJson(python, [
    "-m",
    "forge_training",
    "verify",
    "--run-dir",
    runDir,
  ]);
  expect((verified.rollouts as Document).rollouts).toBe(48);
  // The worlds' journals hold exactly one transition per rollout, with the same rewards,
  // and every recorded step replays from its receipt without changing any world.
  const evidence = await verifyWorldRollouts({
    reader: actors.evaluator,
    replayer: actors.worker,
    campaignId: campaign,
    jobId: "tiny",
    runDir,
  });
  expect(evidence).toMatchObject({ rollouts: 48, transitionsInJournals: 48, replayed: 48 });
  expect(evidence.worlds).toBe(4);
  const training = await readJson(join(runDir, "training.json"));
  expect(training.rewarded_rollouts).toBe(evidence.rewarded);
  expect(Number(evidence.rewarded)).toBeGreaterThan(0);
  // The trainer's rewards are the actor's: one world's journal, read directly.
  const records = (await readFile(join(runDir, "rollouts.jsonl"), "utf8"))
    .trimEnd()
    .split("\n")
    .map((line) => JSON.parse(line) as Document);
  const first = records[0].transition as Document;
  const page = (await actors.director.read(
    "RLWorldActor",
    String(first.worldId),
    "getEvents",
    [0, 100],
  ))!;
  const event = (page.events as Document[]).find(
    (e) =>
      e.type === "ENVIRONMENT_TRANSITION" && (e.payload as Document).episodeId === first.episodeId,
  )!;
  expect((event.payload as Document).reward).toBe(records[0].reward);
  expect(first.worldId).toBe(scopedId("world", campaign, "tiny-a1-l0-w0"));
}, 300000);

test("a reward rewritten after training is caught by the world journals", async () => {
  const runDir = join(output, "interrupted", "run");
  const path = join(runDir, "rollouts.jsonl");
  const original = await readFile(path, "utf8");
  const records = original
    .trimEnd()
    .split("\n")
    .map((line) => JSON.parse(line) as Document);
  const target = records.find((record) => record.reward === 0)!;
  target.reward = 1;
  (target.transition as Document).reward = 1;
  await writeFile(path, records.map((record) => JSON.stringify(record)).join("\n") + "\n");
  try {
    await expect(
      verifyWorldRollouts({ reader: actors.reviewer, campaignId: campaign, jobId: "tiny", runDir }),
    ).rejects.toThrow("REWARD_MISMATCH");
  } finally {
    await writeFile(path, original);
  }
});

test("a reward the worlds never returned is exposed by their journals", async () => {
  // FAULT: the gateway reports one flipped reward. The trainer cannot tell and trains on it.
  const directory = await fresh("misreported");
  const gateway = gatewayFor("forged", { faults: { misreportRewards: 1 } }).start();
  try {
    expect((await train(directory, gateway)).status).toBe("completed");
  } finally {
    await gateway.stop();
  }
  const runDir = join(directory, "run");
  // The run is internally consistent, so file verification alone passes...
  await pythonJson(python, ["-m", "forge_training", "verify", "--run-dir", runDir]);
  // ...but its rewards are not the worlds', so the evidence path refuses it.
  await expect(
    verifyWorldRollouts({ reader: actors.reviewer, campaignId: campaign, jobId: "forged", runDir }),
  ).rejects.toThrow("REWARD_MISMATCH");
}, 300000);
