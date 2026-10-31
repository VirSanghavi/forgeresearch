import { z } from "zod";

export type Json = null | boolean | number | string | Json[] | { [key: string]: Json };
export type Document = { [key: string]: Json };
export type ActorName =
  | "FinalEvaluationActor"
  | "ResearchLedgerActor"
  | "ResearchWorkflowActor"
  | "ResearchCampaignActor"
  | "ExperimentActor"
  | "GovernanceActor"
  | "ArtifactRegistryActor"
  | "RLWorldActor";
export type Role =
  | "owner"
  | "director"
  | "coordinator"
  | "worker"
  | "evaluator"
  | "reviewer"
  | "data"
  | "final_evaluator";

export interface SignedRequest {
  version: 1;
  projectId: string;
  actorName: ActorName;
  actorId: string;
  campaignId: string;
  method: string;
  keyId: string;
  requestId: string;
  issuedAtMs: number;
  expiresAtMs: number;
  payload: Document;
  signature: string;
}

export const id = z.string().regex(/^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$/);
export const hash = z.string().regex(/^[a-f0-9]{64}$/);
export const integer = z.number().int().min(0).max(Number.MAX_SAFE_INTEGER);
export const money = integer.max(1_000_000_000_000);
export const empty = z.strictObject({});
export const model = z.strictObject({
  modelId: z.string().min(1).max(200),
  revision: z.string().regex(/^(?:[a-f0-9]{40}|[a-f0-9]{64})$/),
});
export const baselineSchema = z
  .strictObject({
    reportSha256: hash,
    datasetSha256: hash,
    protocolSha256: hash,
    numExamples: integer.min(1).max(100_000),
    numCorrect: integer,
  })
  .refine((value) => value.numCorrect <= value.numExamples, "Invalid baseline counts");
export const campaignSchema = z.strictObject({
  requirePreflight: z.boolean().optional(),
  objective: z.string().min(1).max(2000),
  model,
  benchmark: z.string().min(1).max(100),
  policyVersion: id,
  allowedStrategies: z
    .array(z.enum(["sft_lora", "grpo"]))
    .min(1)
    .max(2),
  maxBudgetMicroUsd: money.min(1),
  maxConcurrent: integer.min(1).max(32),
  maxExperiments: integer.min(1).max(1000),
  maxAttempts: integer.min(1).max(5),
  maxLeaseMs: integer.min(50).max(300_000),
  maxRunMs: integer.min(100).max(86_400_000),
  minImprovementBps: integer.min(1).max(10_000),
  baseline: baselineSchema,
});
export type CampaignSpec = z.infer<typeof campaignSchema>;
export const proposalSchema = z.strictObject({
  experimentId: id,
  strategy: z.enum(["sft_lora", "grpo"]),
  model,
  configSha256: hash,
  trainingDatasetSha256: hash,
  developmentDatasetSha256: hash,
  hypothesis: z.string().min(1).max(2000),
  maxCostMicroUsd: money.min(1),
});
export type Proposal = z.infer<typeof proposalSchema>;
export const grantClaimsSchema = z.strictObject({
  requirePreflight: z.boolean().optional(),
  version: z.literal(1),
  projectId: id,
  campaignId: id,
  experimentId: id,
  objectiveSha256: hash,
  proposal: proposalSchema,
  maxAttempts: integer.min(1).max(5),
  maxLeaseMs: integer.min(50).max(300_000),
  maxRunMs: integer.min(100).max(86_400_000),
  issuedAtMs: integer,
  grantId: id,
});
export const grantSchema = z.strictObject({ claims: grantClaimsSchema, signature: hash });
export type ReservationGrant = z.infer<typeof grantSchema>;
export const artifactSchema = z.strictObject({
  artifactId: id,
  experimentId: id,
  kind: z.enum(["checkpoint", "evaluation", "dataset", "trajectory", "report"]),
  sha256: hash,
  uri: z.string().min(1).max(2048),
  sizeBytes: integer,
  provenanceSha256: hash,
});
export type Artifact = z.infer<typeof artifactSchema> & { registeredBy: string };
export const trainingResultSchema = z.strictObject({
  checkpointArtifactId: id,
  checkpointSha256: hash,
  manifestSha256: hash,
  configSha256: hash,
  weightsChanged: z.literal(true),
});
export type TrainingResult = z.infer<typeof trainingResultSchema>;
export const evaluationSchema = z
  .strictObject({
    reportArtifactId: id,
    reportSha256: hash,
    baselineReportSha256: hash,
    checkpointSha256: hash,
    datasetSha256: hash,
    protocolSha256: hash,
    evaluationRole: z.literal("development"),
    numExamples: integer.min(1).max(100_000),
    baselineCorrect: integer,
    candidateCorrect: integer,
  })
  .refine(
    (value) =>
      value.baselineCorrect <= value.numExamples && value.candidateCorrect <= value.numExamples,
    "Invalid evaluation counts",
  );
export type Evaluation = z.infer<typeof evaluationSchema>;
export const reviewSchema = z
  .strictObject({
    experimentId: id,
    checkpointSha256: hash,
    evaluationReportSha256: hash,
    objectiveSha256: hash,
    policyVersion: id,
    decision: z.enum(["approved", "rejected"]),
    reasonCodes: z.array(id).max(20),
    evidenceRefs: z.array(hash).min(1).max(20),
    checks: z.strictObject({
      checksumVerified: z.boolean(),
      objectiveAligned: z.boolean(),
      evaluationReproducible: z.boolean(),
      regressionsPassed: z.boolean(),
      dataAllowed: z.boolean(),
    }),
  })
  .superRefine((value, context) => {
    if (
      value.decision === "approved" &&
      (!Object.values(value.checks).every(Boolean) || value.reasonCodes.length)
    ) {
      context.addIssue({
        code: "custom",
        message: "Approval requires all checks and no rejection reasons",
      });
    }
    if (value.decision === "rejected" && !value.reasonCodes.length) {
      context.addIssue({ code: "custom", message: "A rejection requires a reason" });
    }
  });
export type Review = z.infer<typeof reviewSchema> & { reviewerKeyId: string };

export interface DomainEvent {
  eventId: string;
  campaignId: string;
  actorName: ActorName;
  actorId: string;
  sequence: number;
  timestampMs: number;
  type: string;
  payload: Document;
}
/** Persisted with domain state; events and receipts are rows in actor SQLite (core.ts). */
export interface Journal {
  version: number;
}
export interface CampaignState extends Journal {
  spec: CampaignSpec | null;
  objectiveSha256: string | null;
  status: "uninitialized" | "active" | "stopped";
  proposals: Record<string, Proposal>;
  reservations: Record<
    string,
    { grant: ReservationGrant; settled: boolean; spentMicroUsd: number | null }
  >;
  reservedMicroUsd: number;
  spentMicroUsd: number;
  best: {
    experimentId: string;
    checkpointSha256: string;
    candidateCorrect: number;
    numExamples: number;
  } | null;
}
export interface ExperimentState extends Journal {
  grant: ReservationGrant | null;
  status:
    "uninitialized" | "ready" | "running" | "retryable" | "completed" | "evaluated" | "failed";
  generation: number;
  attempts: number;
  firstStartedAtMs: number | null;
  lease: { workerKeyId: string; generation: number; expiresAtMs: number } | null;
  training: TrainingResult | null;
  evaluation: Evaluation | null;
  failure: string | null;
}
export interface GovernanceState extends Journal {
  campaignId: string | null;
  objectiveSha256: string | null;
  policyVersion: string | null;
  decisions: Record<string, Review>;
}
export interface RegistryState extends Journal {
  artifacts: Record<string, Artifact>;
}
export const taskSchema = z.strictObject({
  taskId: id,
  taskVersion: id,
  question: z.string().min(1).max(5000),
  seed: integer.max(2 ** 32 - 1),
  answerCommitment: hash,
  rewardVersion: z.literal("exact-numeric-v1"),
  maxSteps: integer.min(1).max(32),
});
export type MathTask = z.infer<typeof taskSchema>;
export const actionSchema = z.discriminatedUnion("type", [
  z.strictObject({ type: z.literal("submit_answer"), answer: z.string().min(1).max(64) }),
  z.strictObject({
    type: z.literal("calculate"),
    operation: z.enum(["add", "subtract", "multiply"]),
    left: z.string().regex(/^-?\d{1,18}$/),
    right: z.string().regex(/^-?\d{1,18}$/),
  }),
]);
export type MathAction = z.infer<typeof actionSchema>;
export interface Transition {
  episodeId: string;
  actionId: string;
  stateVersion: number;
  callerKeyId: string;
  action: MathAction;
  observation: Document;
  reward: number;
  terminated: boolean;
  truncated: boolean;
  stateSha256: string;
}
export interface WorldEpisode {
  campaignId: string;
  episodeId: string;
  task: MathTask;
  stateVersion: number;
  parentEpisodeId: string | null;
  stepCount: number;
  terminated: boolean;
  truncated: boolean;
  lastCalculation: string | null;
  transitions: Transition[];
  actionReceipts: Record<string, { fingerprint: string; transition: Transition }>;
}
export interface WorldState extends Journal {
  episode: WorldEpisode | null;
  usedEpisodes: string[];
  snapshots: Record<string, WorldEpisode>;
}
