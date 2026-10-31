import { createHash, randomBytes, timingSafeEqual } from "node:crypto";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { z } from "zod";
import type { ForgeActorClient } from "../../../packages/terse-actors/src/client.js";
import { hash, id, type Document } from "../../../packages/terse-actors/src/contracts.js";
import { digest, rewardCommitment, scopedId } from "../../../packages/terse-actors/src/core.js";

/** Episodes per world before rotating to a fresh one (RLWorldActor's EPISODE_LIMIT). */
const WORLD_EPISODES = 1000;
/** The training core's development scorer's final-line rule (`scoring.py` FINAL_ANSWER). */
const FINAL_ANSWER =
  /(?:^|\n)####\s*([+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+))[ \t\r\n]*$/;

/** A training task as the trusted task service holds it, with its reference answer. */
export interface WorldTask {
  taskId: string;
  question: string;
  answer: string;
}
export interface GatewayActors {
  /** Data role: resets worlds with answer commitments. */
  data: ForgeActorClient;
  /** Worker role: submits the policy's actions. */
  worker: ForgeActorClient;
}
export interface WorldGatewayOptions {
  campaignId: string;
  experimentId: string;
  /** Short job label; with the attempt it names worlds and episodes. */
  jobId: string;
  /** Lease generation of the training attempt; a retry never reuses episodes. */
  attempt: number;
  tasks: WorldTask[];
  taskVersion: string;
  seed: number;
  rewardSecret: string;
  /** Read at call time, so a caller may swap clients after a runtime restart. */
  actors: () => GatewayActors;
  rolloutsPerUpdate: number;
  maxUpdates: number;
  /** Worlds stepped concurrently; each lane's episodes are sequential. */
  lanes?: number;
  /** How long transport failures are retried with the same request IDs. */
  retryMs?: number;
  /** Labeled failure injection. */
  faults?: {
    /** Discard this many step replies after the actor applied them, then resend. */
    loseStepReplies?: number;
    /** Report this many rewards that the world did not return (a disconnected reward). */
    misreportRewards?: number;
    /** Runs after each applied transition (counted from 1), e.g. to kill the runtime. */
    afterTransition?: (count: number) => Promise<void> | void;
  };
}

export function referenceAnswer(answer: string): string {
  const match = FINAL_ANSWER.exec(answer);
  if (!match) throw new Error("INVALID_REFERENCE: training answers need a final '#### N' line");
  return match[1].replaceAll(",", "");
}
const sha256 = (bytes: string | Buffer) => createHash("sha256").update(bytes).digest("hex");
function lines(text: string): string[] {
  if (!text.endsWith("\n")) throw new Error("Dataset records must end with a newline");
  return text.slice(0, -1).split("\n");
}

/**
 * Task service: answers for exactly the job's answer-free tasks, from the allowed partition.
 * Both files are checked against their manifests; every task must match its source question.
 */
export async function loadWorldTasks(options: {
  sourceRecords: string;
  sourceManifest: string;
  tasksRecords: string;
  tasksManifest: string;
}): Promise<{ tasks: WorldTask[]; taskVersion: string }> {
  const read = async (records: string, manifest: string, kind: string) => {
    const text = await readFile(records, "utf8");
    const meta = JSON.parse(await readFile(manifest, "utf8")) as Document;
    if (sha256(text) !== meta.content_sha256 || (meta.record_kind ?? "example") !== kind)
      throw new Error(`TASK_SOURCE_MISMATCH: ${records} does not match its ${kind} manifest`);
    return { rows: lines(text).map((line) => JSON.parse(line) as Document), sha256: sha256(text) };
  };
  const source = await read(options.sourceRecords, options.sourceManifest, "example");
  const tasks = await read(options.tasksRecords, options.tasksManifest, "task");
  const byId = new Map(source.rows.map((row) => [String(row.id), row]));
  return {
    taskVersion: `tasks-${tasks.sha256.slice(0, 16)}`,
    tasks: tasks.rows.map((task) => {
      const row = byId.get(String(task.id));
      if (!row || row.question !== task.question || "answer" in task)
        throw new Error(`TASK_SOURCE_MISMATCH: ${String(task.id)} is not an allowed source task`);
      return {
        taskId: String(task.id),
        question: String(task.question),
        answer: referenceAnswer(String(row.answer)),
      };
    }),
  };
}

const rolloutRequest = z.strictObject({
  schemaVersion: z.literal(1),
  campaignId: id,
  experimentId: id,
  update: z.number().int().min(1).max(100_000),
  policySha256: hash,
  rollouts: z
    .array(
      z.strictObject({
        rolloutId: z.string().regex(/^u\d{5}-r\d{3}$/),
        taskId: id,
        action: z.strictObject({
          type: z.literal("submit_answer"),
          answer: z.string().min(1).max(64),
        }),
        completionSha256: hash,
      }),
    )
    .min(1)
    .max(1024),
});
type RolloutRequest = z.infer<typeof rolloutRequest>;

/** Durable-actors codes for calls that never ran or may have run; Forge receipts dedupe both. */
function transient(error: unknown): boolean {
  const code = (error as { code?: unknown } | null)?.code;
  return (
    code === "unavailable" ||
    code === "outcome_unknown" ||
    (error as Error | null)?.name === "ActorProtocolError"
  );
}
const pad = (value: number, width: number) => String(value).padStart(width, "0");
function reply(status: number, body: Document): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

/**
 * Loopback bridge from a GRPO trainer to Terse math worlds. It scores each completion's
 * action with one data-role reset and one worker-role step; the reward in its reply is the
 * actor's. Episodes, actions and request IDs derive from the job, attempt and rollout, so
 * retries after lost replies or a runtime restart replay receipts instead of re-scoring.
 * It holds the task answers and signing clients; the trainer holds only a per-job token.
 */
export class WorldGateway {
  readonly token = randomBytes(32).toString("hex");
  readonly stats = { requests: 0, transitions: 0, replayed: 0, retries: 0, lostReplies: 0 };
  private readonly tasks: Map<string, WorldTask>;
  private readonly lanes: number;
  private readonly updates = new Map<
    number,
    { fingerprint: string; result: Promise<Document[]> }
  >();
  private completed = 0;
  private lostReplies: number;
  private misreports: number;
  private server: ReturnType<typeof Bun.serve> | null = null;

  constructor(readonly options: WorldGatewayOptions) {
    this.tasks = new Map(options.tasks.map((task) => [task.taskId, task]));
    this.lanes = Math.max(1, Math.min(options.lanes ?? 4, options.rolloutsPerUpdate));
    this.lostReplies = options.faults?.loseStepReplies ?? 0;
    this.misreports = options.faults?.misreportRewards ?? 0;
    // Names must fit the actor's ID rules for the longest world this run can create.
    const worlds = Math.ceil(
      (options.maxUpdates * Math.ceil(options.rolloutsPerUpdate / this.lanes)) / WORLD_EPISODES,
    );
    this.worldId(this.lanes - 1, worlds - 1);
    id.parse(this.episodeId(`u${pad(options.maxUpdates, 5)}-r999`));
  }
  private worldId(lane: number, generation: number): string {
    const { campaignId, jobId, attempt } = this.options;
    return scopedId("world", campaignId, `${jobId}-a${attempt}-l${lane}-w${generation}`);
  }
  private episodeId(rolloutId: string): string {
    return `${this.options.jobId}-a${this.options.attempt}-${rolloutId}`;
  }

  get url(): string {
    if (!this.server) throw new Error("World gateway is not running");
    return `http://127.0.0.1:${this.server.port}`;
  }
  /** The only credentials the trainer receives. */
  environment(): Record<string, string> {
    return { FORGE_WORLD_GATEWAY_URL: this.url, FORGE_WORLD_GATEWAY_TOKEN: this.token };
  }
  start(): this {
    // Requests stay open while transitions retry across a runtime restart.
    this.server = Bun.serve({
      hostname: "127.0.0.1",
      port: 0,
      idleTimeout: 255,
      fetch: (request) => this.handle(request),
    });
    return this;
  }
  async stop(): Promise<void> {
    await this.server?.stop(true);
    this.server = null;
  }

  async handle(request: Request): Promise<Response> {
    if (request.method !== "POST" || new URL(request.url).pathname !== "/v1/rollouts")
      return reply(404, { error: "NOT_FOUND" });
    const offered = Buffer.from(request.headers.get("authorization") ?? "");
    const expected = Buffer.from(`Bearer ${this.token}`);
    if (offered.length !== expected.length || !timingSafeEqual(offered, expected))
      return reply(401, { error: "UNAUTHORIZED" });
    this.stats.requests++;
    const text = await request.text();
    let payload: RolloutRequest;
    try {
      if (text.length > 4_000_000) throw new Error("Request exceeds 4 MB");
      payload = rolloutRequest.parse(JSON.parse(text));
    } catch (error) {
      return reply(400, { error: "INVALID_REQUEST", message: String(error).slice(0, 500) });
    }
    const problem = this.refuse(payload);
    if (problem) return reply(problem.status, { error: problem.code, message: problem.message });
    const fingerprint = digest(payload);
    let entry = this.updates.get(payload.update);
    if (entry && entry.fingerprint !== fingerprint)
      return reply(409, {
        error: "ROLLOUT_CONFLICT",
        message: "This update was already scored with different rollouts",
      });
    if (!entry) {
      // A retry of an in-flight or finished update shares its single scoring pass.
      entry = { fingerprint, result: this.score(payload) };
      this.updates.set(payload.update, entry);
    }
    try {
      const results = await entry.result;
      this.completed = Math.max(this.completed, payload.update);
      return reply(200, { results });
    } catch (error) {
      if (this.updates.get(payload.update) === entry) this.updates.delete(payload.update);
      const retryable = transient(error);
      return reply(retryable ? 503 : 502, {
        error: retryable ? "WORLD_UNAVAILABLE" : "WORLD_REJECTED",
        message: String(error instanceof Error ? error.message : error).slice(0, 500),
      });
    }
  }
  private refuse(
    payload: RolloutRequest,
  ): { status: number; code: string; message: string } | null {
    const { campaignId, experimentId, rolloutsPerUpdate, maxUpdates } = this.options;
    if (payload.campaignId !== campaignId || payload.experimentId !== experimentId)
      return { status: 403, code: "SCOPE_MISMATCH", message: "Rollouts belong to another job" };
    // Bounded remote transitions: a fixed batch per update, at most maxUpdates updates.
    if (payload.update > maxUpdates || payload.rollouts.length !== rolloutsPerUpdate)
      return { status: 400, code: "ROLLOUT_BOUND", message: "Batch outside the job's bounds" };
    if (!this.updates.has(payload.update) && payload.update !== this.completed + 1)
      return { status: 409, code: "OUT_OF_ORDER", message: "Updates are scored in order" };
    for (const [index, rollout] of payload.rollouts.entries()) {
      if (rollout.rolloutId !== `u${pad(payload.update, 5)}-r${pad(index, 3)}`)
        return { status: 400, code: "ROLLOUT_ID", message: "Rollout IDs must follow the batch" };
      if (!this.tasks.has(rollout.taskId))
        return {
          status: 400,
          code: "UNKNOWN_TASK",
          message: `Not an allowed task: ${rollout.taskId}`,
        };
    }
    return null;
  }

  private async score(payload: RolloutRequest): Promise<Document[]> {
    const results: Document[] = new Array(payload.rollouts.length);
    // Every lane settles before a failure is reported, so a retry never overlaps this pass.
    const lanes = await Promise.allSettled(
      Array.from({ length: this.lanes }, async (_, lane) => {
        for (let index = lane; index < payload.rollouts.length; index += this.lanes)
          results[index] = await this.rollout(payload.update, index, lane, payload.rollouts[index]);
      }),
    );
    const failed = lanes.find((lane) => lane.status === "rejected");
    if (failed) throw (failed as PromiseRejectedResult).reason;
    return results;
  }
  private async rollout(
    update: number,
    index: number,
    lane: number,
    rollout: RolloutRequest["rollouts"][number],
  ): Promise<Document> {
    const { campaignId, taskVersion, seed, rewardSecret } = this.options;
    const slot =
      (update - 1) * Math.ceil(this.options.rolloutsPerUpdate / this.lanes) +
      Math.floor(index / this.lanes);
    const worldId = this.worldId(lane, Math.floor(slot / WORLD_EPISODES));
    const episodeId = this.episodeId(rollout.rolloutId);
    const task = this.tasks.get(rollout.taskId)!;
    await this.call("data", worldId, "reset", `reset-${episodeId}`, {
      episodeId,
      task: {
        taskId: task.taskId,
        taskVersion,
        question: task.question,
        seed,
        answerCommitment: rewardCommitment(
          campaignId,
          worldId,
          task.taskId,
          taskVersion,
          seed,
          task.question,
          task.answer,
          rewardSecret,
        ),
        rewardVersion: "exact-numeric-v1",
        maxSteps: 1,
      },
    });
    const step = {
      episodeId,
      actionId: `${episodeId}-a0`,
      expectedVersion: 0,
      action: rollout.action,
    };
    let result = await this.call("worker", worldId, "step", `step-${episodeId}`, step);
    if (this.lostReplies > 0) {
      // FAULT: the actor applied the step but its reply is discarded; resend the same request.
      this.lostReplies--;
      this.stats.lostReplies++;
      result = await this.call("worker", worldId, "step", `step-${episodeId}`, step);
      if (result.replayed !== true)
        throw new Error("WORLD_REPLAY: a resent step was applied twice");
    }
    const transition = result.transition as Document;
    if (
      transition.episodeId !== episodeId ||
      transition.actionId !== step.actionId ||
      digest(transition.action) !== digest(rollout.action) ||
      transition.terminated !== true
    )
      throw new Error("WORLD_MISMATCH: transition does not answer this rollout");
    this.stats.transitions++;
    if (result.replayed === true) this.stats.replayed++;
    await this.options.faults?.afterTransition?.(this.stats.transitions);
    let reward = transition.reward;
    if (this.misreports > 0) {
      // FAULT: the trainer is told a reward the world did not return.
      this.misreports--;
      reward = reward === 1 ? 0 : 1;
    }
    return {
      rolloutId: rollout.rolloutId,
      worldId,
      episodeId,
      actionId: step.actionId,
      stateVersion: transition.stateVersion,
      stateSha256: transition.stateSha256,
      reward,
      terminated: transition.terminated,
      truncated: transition.truncated,
      replayed: result.replayed === true,
    };
  }
  /** Same request ID on every retry: the actor returns its receipt rather than acting twice. */
  private async call(
    role: keyof GatewayActors,
    worldId: string,
    method: string,
    requestId: string,
    payload: Document,
  ): Promise<Document> {
    const deadline = Date.now() + (this.options.retryMs ?? 120_000);
    for (let delay = 100; ; delay = Math.min(delay * 2, 2000)) {
      try {
        return await this.options
          .actors()
          [role].command(
            "RLWorldActor",
            worldId,
            this.options.campaignId,
            method,
            payload,
            requestId,
          );
      } catch (error) {
        if (!transient(error) || Date.now() + delay > deadline) throw error;
        this.stats.retries++;
        await Bun.sleep(delay);
      }
    }
  }
}

export interface WorldEvidence {
  rollouts: number;
  rewarded: number;
  worlds: number;
  attempt: number;
  transitionsInJournals: number;
  replayed: number | null;
}
async function journal(reader: ForgeActorClient, worldId: string) {
  const events: Document[] = [];
  for (let after = 0; ;) {
    const page = (await reader.read("RLWorldActor", worldId, "getEvents", [after, 100]))!;
    const batch = page.events as Document[];
    events.push(...batch);
    if (!batch.length || Number(batch.at(-1)!.sequence) >= Number(page.latestSequence))
      return { events, latestSequence: Number(page.latestSequence) };
    after = Number(batch.at(-1)!.sequence);
  }
}

/**
 * Checks a GRPO run's rollouts against the world actors' own journals: each rollout is
 * exactly one recorded transition with the same reward, state and task, each world holds
 * no other transitions, and nothing else of this job exists. With a worker client, every
 * recorded step is resent with its original request ID and must replay without effect.
 */
export async function verifyWorldRollouts(options: {
  reader: ForgeActorClient;
  campaignId: string;
  jobId: string;
  runDir: string;
  replayer?: ForgeActorClient;
}): Promise<WorldEvidence> {
  const { reader, campaignId, jobId, runDir } = options;
  const records = lines(await readFile(join(runDir, "rollouts.jsonl"), "utf8")).map(
    (line) => JSON.parse(line) as Document,
  );
  const attempts = new Set<string>();
  const byWorld = new Map<string, Document[]>();
  for (const record of records) {
    const transition = record.transition as Document;
    const match = new RegExp(`^${jobId.replace(/[.]/g, "\\.")}-a(\\d+)-${record.rollout_id}$`).exec(
      String(transition.episodeId),
    );
    if (!match || transition.actionId !== `${transition.episodeId}-a0`)
      throw new Error(`REWARD_PROVENANCE: ${String(record.rollout_id)} is not this job's episode`);
    attempts.add(match[1]);
    const world = String(transition.worldId);
    const prefix = scopedId("world", campaignId, `${jobId}-a${match[1]}`);
    if (!world.startsWith(`${prefix}-l`) || !/^-l\d+-w\d+$/.test(world.slice(prefix.length)))
      throw new Error(`REWARD_PROVENANCE: ${world} is not a world of this job`);
    byWorld.set(world, [...(byWorld.get(world) ?? []), record]);
  }
  if (attempts.size !== 1) throw new Error("REWARD_PROVENANCE: rollouts span several attempts");
  let transitionsInJournals = 0;
  const sequences = new Map<string, number>();
  for (const [worldId, rollouts] of byWorld) {
    const { events, latestSequence } = await journal(reader, worldId);
    sequences.set(worldId, latestSequence);
    const transitions = events.filter((e) => e.type === "ENVIRONMENT_TRANSITION");
    const resets = new Map(
      events
        .filter((e) => e.type === "ENVIRONMENT_RESET")
        .map((e) => [String((e.payload as Document).episodeId), e.payload as Document]),
    );
    transitionsInJournals += transitions.length;
    // Exactly once: the journal holds one transition per rollout and nothing else.
    if (transitions.length !== rollouts.length)
      throw new Error(
        `REWARD_PROVENANCE: ${worldId} journal has ${transitions.length} transitions for ${rollouts.length} rollouts`,
      );
    const byEpisode = new Map(
      transitions.map((e) => [String((e.payload as Document).episodeId), e.payload as Document]),
    );
    for (const record of rollouts) {
      const recorded = record.transition as Document;
      const event = byEpisode.get(String(recorded.episodeId));
      const reset = resets.get(String(recorded.episodeId));
      if (
        !event ||
        !reset ||
        event.actionId !== recorded.actionId ||
        event.reward !== record.reward ||
        event.reward !== recorded.reward ||
        event.stateVersion !== recorded.stateVersion ||
        event.stateSha256 !== recorded.stateSha256 ||
        reset.taskId !== record.task_id
      )
        throw new Error(
          `REWARD_MISMATCH: ${String(record.rollout_id)} differs from the world journal`,
        );
    }
  }
  let replayed: number | null = null;
  if (options.replayer) {
    replayed = 0;
    for (const record of records) {
      const recorded = record.transition as Document;
      const result = await options.replayer.command(
        "RLWorldActor",
        String(recorded.worldId),
        campaignId,
        "step",
        {
          episodeId: recorded.episodeId,
          actionId: recorded.actionId,
          expectedVersion: 0,
          action: record.action,
        },
        `step-${String(recorded.episodeId)}`,
      );
      const transition = result.transition as Document;
      if (
        result.replayed !== true ||
        transition.reward !== record.reward ||
        transition.stateSha256 !== recorded.stateSha256
      )
        throw new Error(
          `REPLAY_MISMATCH: ${String(record.rollout_id)} did not replay its transition`,
        );
      replayed++;
    }
    for (const [worldId, sequence] of sequences)
      if ((await journal(reader, worldId)).latestSequence !== sequence)
        throw new Error(`REPLAY_MISMATCH: replay changed ${worldId}`);
  }
  return {
    rollouts: records.length,
    rewarded: records.filter((record) => record.reward === 1).length,
    worlds: byWorld.size,
    attempt: Number([...attempts][0]),
    transitionsInJournals,
    replayed,
  };
}
