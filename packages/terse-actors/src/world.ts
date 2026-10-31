import { Actor, Atomic, Persisted } from "durable-actors";
import { timingSafeEqual } from "node:crypto";
import { z } from "zod";
import {
  actionSchema,
  empty,
  id,
  integer,
  taskSchema,
  type Document,
  type SignedRequest,
  type Transition,
  type WorldEpisode,
  type WorldState,
} from "./contracts.js";
import {
  authorize,
  commit,
  digest,
  document,
  duplicate,
  events,
  fail,
  journal,
  numericAnswer,
  requireScopedId,
  rewardCommitment,
} from "./core.js";

const actorName = "RLWorldActor";
function observation(episode: WorldEpisode): Document {
  return {
    episodeId: episode.episodeId,
    parentEpisodeId: episode.parentEpisodeId,
    taskId: episode.task.taskId,
    taskVersion: episode.task.taskVersion,
    question: episode.task.question,
    seed: episode.task.seed,
    rewardVersion: episode.task.rewardVersion,
    stateVersion: episode.stateVersion,
    stepCount: episode.stepCount,
    maxSteps: episode.task.maxSteps,
    terminated: episode.terminated,
    truncated: episode.truncated,
    lastCalculation: episode.lastCalculation,
  };
}
function current(state: WorldState, campaignId: string): WorldEpisode {
  if (!state.episode || state.episode.campaignId !== campaignId)
    fail("WORLD_SCOPE", "World is not initialized for this campaign");
  return state.episode;
}
export class RLWorldActor extends Actor {
  @Persisted private state: WorldState = {
    ...journal(),
    episode: null,
    usedEpisodes: [],
    snapshots: {},
  };

  @Atomic
  async reset(request: SignedRequest): Promise<Document> {
    const auth = authorize(
      request,
      actorName,
      this.id,
      "reset",
      ["data"],
      z.strictObject({ episodeId: id, task: taskSchema }),
    );
    requireScopedId("world", request.campaignId, this.id);
    const previous = duplicate(this.db, auth);
    if (previous) return previous;
    if (this.state.episode && this.state.episode.campaignId !== request.campaignId)
      fail("WORLD_SCOPE", "World campaign is immutable");
    if (this.state.usedEpisodes.includes(auth.payload.episodeId))
      fail("EPISODE_REUSED", "Episode IDs cannot be reused");
    if (this.state.usedEpisodes.length >= 1000)
      fail("EPISODE_LIMIT", "Create a new world after 1000 episodes");
    const episode: WorldEpisode = {
      campaignId: request.campaignId,
      episodeId: auth.payload.episodeId,
      parentEpisodeId: null,
      task: auth.payload.task,
      stateVersion: 0,
      stepCount: 0,
      terminated: false,
      truncated: false,
      lastCalculation: null,
      transitions: [],
      actionReceipts: {},
    };
    const result = observation(episode);
    this.state = commit(
      this.db,
      { ...this.state, episode, usedEpisodes: [...this.state.usedEpisodes, episode.episodeId] },
      auth,
      "ENVIRONMENT_RESET",
      result,
      { episodeId: episode.episodeId, taskId: episode.task.taskId, seed: episode.task.seed },
    );
    return result;
  }

  async observe(request: SignedRequest): Promise<Document> {
    authorize(request, actorName, this.id, "observe", ["worker", "data", "director"], empty);
    return observation(current(this.state, request.campaignId));
  }

  @Atomic
  async step(request: SignedRequest): Promise<Document> {
    const auth = authorize(
      request,
      actorName,
      this.id,
      "step",
      ["worker"],
      z.strictObject({
        episodeId: id,
        actionId: id,
        expectedVersion: integer,
        action: actionSchema,
      }),
    );
    const previous = duplicate(this.db, auth);
    if (previous) return { ...previous, replayed: true };
    const episode = current(this.state, request.campaignId),
      input = auth.payload;
    if (input.episodeId !== episode.episodeId)
      fail("EPISODE_CONFLICT", "Action belongs to another episode");
    const actionFingerprint = digest({ ...input, callerKeyId: auth.keyId });
    const applied = episode.actionReceipts[input.actionId];
    if (applied) {
      if (applied.fingerprint !== actionFingerprint)
        fail("ACTION_CONFLICT", "Action ID already identifies a different action");
      return document({ transition: applied.transition, replayed: true });
    }
    if (input.expectedVersion !== episode.stateVersion)
      fail("VERSION_CONFLICT", "Action was based on stale world state");
    if (episode.terminated || episode.truncated)
      fail("EPISODE_FINISHED", "Episode no longer accepts actions");
    const next = structuredClone(episode);
    let reward = 0;
    if (input.action.type === "submit_answer") {
      const answer = numericAnswer(input.action.answer);
      if (answer !== null) {
        const commitment = rewardCommitment(
          request.campaignId,
          this.id,
          episode.task.taskId,
          episode.task.taskVersion,
          episode.task.seed,
          episode.task.question,
          answer,
        );
        reward = Number(
          timingSafeEqual(
            Buffer.from(commitment, "hex"),
            Buffer.from(episode.task.answerCommitment, "hex"),
          ),
        );
      }
      next.terminated = true;
    } else {
      const left = BigInt(input.action.left),
        right = BigInt(input.action.right);
      next.lastCalculation = (
        input.action.operation === "add"
          ? left + right
          : input.action.operation === "subtract"
            ? left - right
            : left * right
      ).toString();
    }
    next.stepCount += 1;
    next.stateVersion += 1;
    next.truncated = !next.terminated && next.stepCount >= next.task.maxSteps;
    const observed = observation(next);
    const transition: Transition = {
      episodeId: next.episodeId,
      actionId: input.actionId,
      stateVersion: next.stateVersion,
      callerKeyId: auth.keyId,
      action: input.action,
      observation: observed,
      reward,
      terminated: next.terminated,
      truncated: next.truncated,
      stateSha256: digest(observed),
    };
    next.transitions = [...next.transitions, transition];
    next.actionReceipts = {
      ...next.actionReceipts,
      [input.actionId]: { fingerprint: actionFingerprint, transition },
    };
    const result = document({ transition, replayed: false });
    this.state = commit(
      this.db,
      { ...this.state, episode: next },
      auth,
      "ENVIRONMENT_TRANSITION",
      result,
      {
        episodeId: next.episodeId,
        actionId: input.actionId,
        stateVersion: next.stateVersion,
        reward,
        stateSha256: transition.stateSha256,
      },
    );
    return result;
  }

  @Atomic
  async saveSnapshot(request: SignedRequest): Promise<Document> {
    const auth = authorize(
      request,
      actorName,
      this.id,
      "saveSnapshot",
      ["data"],
      z.strictObject({ snapshotId: id }),
    );
    const previous = duplicate(this.db, auth);
    if (previous) return previous;
    const episode = current(this.state, request.campaignId);
    if (this.state.snapshots[auth.payload.snapshotId])
      fail("SNAPSHOT_IMMUTABLE", "Snapshot ID already exists");
    if (Object.keys(this.state.snapshots).length >= 32)
      fail("SNAPSHOT_LIMIT", "World snapshot capacity reached");
    const result = {
      snapshotId: auth.payload.snapshotId,
      stateSha256: digest(observation(episode)),
    };
    this.state = commit(
      this.db,
      {
        ...this.state,
        snapshots: { ...this.state.snapshots, [auth.payload.snapshotId]: structuredClone(episode) },
      },
      auth,
      "ENVIRONMENT_SNAPSHOT",
      result,
      result,
    );
    return result;
  }

  @Atomic
  async restore(request: SignedRequest): Promise<Document> {
    const auth = authorize(
      request,
      actorName,
      this.id,
      "restore",
      ["data"],
      z.strictObject({ snapshotId: id, newEpisodeId: id }),
    );
    const previous = duplicate(this.db, auth);
    if (previous) return previous;
    const old = current(this.state, request.campaignId);
    const snapshot = this.state.snapshots[auth.payload.snapshotId];
    if (!snapshot) fail("UNKNOWN_SNAPSHOT", "No saved snapshot with this ID");
    if (this.state.usedEpisodes.includes(auth.payload.newEpisodeId))
      fail("EPISODE_REUSED", "Restoration must fork into a fresh episode");
    if (this.state.usedEpisodes.length >= 1000)
      fail("EPISODE_LIMIT", "Create a new world after 1000 episodes");
    const episode = {
      ...structuredClone(snapshot),
      episodeId: auth.payload.newEpisodeId,
      parentEpisodeId: snapshot.episodeId,
      stateVersion: old.stateVersion + 1,
    };
    const result = observation(episode);
    this.state = commit(
      this.db,
      { ...this.state, episode, usedEpisodes: [...this.state.usedEpisodes, episode.episodeId] },
      auth,
      "ENVIRONMENT_RESTORED",
      result,
      { snapshotId: auth.payload.snapshotId, episodeId: episode.episodeId },
    );
    return result;
  }

  async getTrajectory(request: SignedRequest): Promise<Document> {
    authorize(request, actorName, this.id, "getTrajectory", ["worker", "data", "director"], empty);
    const episode = current(this.state, request.campaignId);
    return document({
      observation: observation(episode),
      transitions: episode.transitions,
      rewardTotal: episode.transitions.reduce((total, step) => total + step.reward, 0),
    });
  }
  async getEvents(after: number, limit: number): Promise<Document> {
    return events(this.db, this.state.version, after, limit);
  }
}
