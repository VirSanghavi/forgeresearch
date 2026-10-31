export type View =
  | "briefing"
  | "chat"
  | "production"
  | "terse"
  | "overview"
  | "experiments"
  | "environments"
  | "governance"
  | "results"
  | "artifacts"
  | "connections"
  | "overseer"
  | "evidence";
export type ResultFilter = "all" | "improved" | "regressed";
export type Agent =
  | "director"
  | "data"
  | "trainer"
  | "evaluator"
  | "governance"
  | "hephaestus";
export type Prediction = {
  id: string;
  question: string;
  output: string;
  answer: string;
  expected: string;
  correct: boolean;
  environment_id?: string;
};
export type Hardware = {
  backend: "local" | "modal";
  type: string;
  name: string | null;
  requested?: string;
  torch?: string;
  cuda?: string | null;
};
/** The device that ran the model work, or the GPU requested before it reports in. */
export const hardwareName = (h?: Hardware | null) =>
  h?.name ?? (h?.requested ? `NVIDIA ${h.requested} (starting)` : null);
export const hardwareLabel = (h?: Hardware | null) => {
  const name = hardwareName(h);
  return name && h?.backend === "modal" ? `${name} · Modal` : name;
};
export type Evaluation = {
  score: number;
  n: number;
  correct: number;
  ci95: number[];
  predictions: Prediction[];
  split: string;
  seconds: number;
  dataset_revision: string;
  scoring: string;
  decoding: { max_new_tokens: number; do_sample: boolean; batch_size?: number };
  device?: Hardware | null;
};
export type Metric = {
  step: number;
  loss: number;
  elapsed: number;
  learning_rate: number;
};
export type Review = {
  id: string;
  decision: "approved" | "rejected";
  reason_codes: string[];
  timestamp: string;
  evidence: Record<string, unknown>;
  policy_version: string;
};
export type Experiment = {
  id: string;
  strategy: string;
  hypothesis: string;
  status: string;
  created_at: string;
  max_steps: number;
  metrics: Metric[];
  proposal: {
    learning_rate: number;
    lora_rank: number;
    focus: string;
    director: string;
  };
  evaluation?: Evaluation;
  production?: Evaluation | null;
  review?: Review;
  training?: {
    weights_changed: boolean;
    sha256: string;
    before_hash: string;
    after_hash: string;
    seconds: number;
    max_sequence_length?: number;
    lora_targets?: string[];
    dtype?: string;
    device?: Hardware;
  };
};
export type Artifact = {
  id: string;
  name: string;
  kind: string;
  path: string;
  sha256: string;
  bytes: number;
  created_at: string;
};
export type Environment = {
  id: string;
  episode_id: string;
  task: string;
  seed: number;
  version: number;
  terminated: boolean;
  reward: number | null;
  actions: { action_id: string; answer: string; reward: number }[];
  reward_version: string;
  checksum: string;
};
const MODEL_LABELS: Record<string, string> = {
  "Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5 · 0.5B",
  "Qwen/Qwen3.5-4B": "Qwen3.5 · 4B",
};
export const modelLabel = (id?: string) =>
  (id && (MODEL_LABELS[id] ?? id.split("/").pop())) || "Qwen2.5 · 0.5B";
export type Config = {
  name: string;
  objective: string;
  model_id: string;
  benchmark: string;
  max_seconds: number;
  train_steps: number;
  max_experiments: number;
  dev_samples: number;
  final_samples: number;
  seed: number;
  strategies: string[];
  task_pack?: string;
  synthetic_dataset?: string | null;
};
// One Freestyle VM a research role used to check evidence with code
// (services/core/workbench.py, saved on the campaign as `workbenches`).
export type Workbench = {
  id: string;
  slug?: string | null;
  role: Agent;
  runs: number;
  seconds: number;
  deleted: boolean;
  error?: string;
};
export type Campaign = {
  workbenches?: Workbench[];
  briefing?: {
    as_of: string;
    phase: string;
    experiment_number: number;
    experiment_limit: number;
    experiment_id: string | null;
    strategy: string | null;
    hypothesis: string | null;
    progress?: { stage: string; done: number; total: number };
    director: {
      input_recorded: boolean;
      failure_examples: number | null;
      previous_results: number | null;
      development_score_seen: number | null;
      final_test_data_provided: boolean;
    };
    oversight: {
      decision: string;
      decision_scope: string;
      checks: { label: string; state: string }[];
      blocked_proposals: number;
      last_block: Review | null;
      regression_scope: string;
    };
    execution: {
      terse_worlds: number;
      rl_rollouts: number;
      relay_deliveries: number;
      settled_compute_seconds: number;
      time_limit_seconds: number;
      hardware?: Hardware | null;
    };
  };
  id: string;
  name: string;
  status: string;
  created_at: string;
  updated_at: string;
  config: Config;
  hardware?: Hardware;
  director?: string;
  experiments: Experiment[];
  reviews: Review[];
  artifacts: Artifact[];
  environments: Environment[];
  baseline?: Evaluation;
  best?: {
    id: string;
    evaluation: Evaluation;
    sha256: string;
    production?: Evaluation | null;
  };
  final?: {
    baseline: Evaluation;
    checkpoint: Evaluation;
    improvement_pp: number;
    checkpoint_id: string;
    valid_improvement: boolean;
    production?: ProductionVerdict;
  };
  production?: CampaignProduction;
  production_baseline?: Evaluation;
  actor?: {
    consumed_seconds: number;
    reservations: Record<string, { seconds: number; status: string }>;
    selection_frozen: boolean;
    best_checkpoint: string | null;
  };
  progress?: { stage: string; done: number; total: number };
  elapsed_seconds?: number;
  compute_seconds?: number;
  error?: string;
  cancel_requested?: boolean;
  recording?: { status: string };
};
export type RelayDelivery = {
  role: Agent;
  delivery_id: string | null;
  state: string;
  read_at: string | null;
  source: "inbox" | "readback";
};
export type RelayReaction = { emoji: string; count: number; agents: Agent[] };
export type Event = {
  id: string;
  timestamp: string;
  kind: string;
  text: string;
  agent: Agent;
  recipient?: Agent;
  category?: string;
  /** Forge message ID; message_pending carries it inside `message`. */
  message_id?: string;
  message?: {
    id: string;
    recipient?: Agent;
    route?: "dm" | "room";
    category?: string;
  };
  relay_id?: string;
  route?: "dm" | "room" | "thread";
  thread_parent_id?: string | null;
  emoji?: string;
  reactions?: RelayReaction[];
  receipt?: {
    delivered: boolean;
    recipient_readback_verified: boolean;
    route?: "dm" | "room" | "thread";
    conversation_id?: string | null;
    woke_by?: "push" | "inbox-sweep";
    deliveries?: RelayDelivery[];
    read_by?: { name: Agent; read_at: string | null }[];
    room?: {
      name: string;
      id: string;
      /** Earlier receipts copied every DM into the room. */
      message_id?: string;
      readback_verified?: boolean;
      readers?: Agent[];
    };
  };
  experiment_id?: string;
  metric?: Metric;
};
export type Health = {
  ok: boolean;
  worker: string;
  compute?: { backend: string; label: string | null; gpu: string | null };
  model?: { id: string; label: string; parameters: string };
  version: string;
  services: Record<
    string,
    {
      connected: boolean;
      mode: string;
      model?: string;
      transport?: string;
      endpoint?: string;
    }
  >;
};
export const TERMINAL = ["completed", "failed", "cancelled", "interrupted"];
export const agentNames: Record<Agent, string> = {
  director: "Research director",
  data: "Data researcher",
  trainer: "Training engineer",
  evaluator: "Experiment evaluator",
  governance: "Governance council",
  hephaestus: "Hephaestus",
};
const STRATEGY_NAMES: Record<string, string> = {
  broad_sft: "Broad SFT",
  targeted_sft: "Targeted SFT",
  distill_sft: "Teacher distillation",
  hard_sft: "Hard examples",
  grpo: "GRPO",
  dpo: "DPO alignment",
};
export const strategyName = (s: string) => STRATEGY_NAMES[s] ?? s;
export const percent = (n?: number) =>
  n === undefined ? "Not yet" : `${(n * 100).toFixed(1)}%`;
export const duration = (s: number) =>
  s < 60
    ? `${Math.round(s)}s`
    : s < 3600
      ? `${Math.floor(s / 60)}m ${Math.floor(s % 60)}s`
      : `${(s / 3600).toFixed(2)}h`;

export type DetectorKind =
  | "arithmetic_slip"
  | "missing_final_answer"
  | "truncated"
  | "user_reported";
export type DetectorAudit = {
  labeled: number;
  actually_wrong: number;
  true_positives: number;
  false_positives: number;
  missed: number;
  precision: number | null;
  recall: number | null;
};
export type DetectorCategory = {
  kind: DetectorKind;
  label: string;
  count: number;
};
export type ProductionTrace = {
  id: string;
  prompt: string;
  output: string;
  finish_reason: string | null;
  feedback?: { rating?: number; correction?: string };
  timestamp?: string;
};
export type ProductionSignal = {
  kind: DetectorKind;
  evidence: {
    equation?: string;
    computed?: string;
    tail?: string;
    finish_reason?: string;
    rating?: number;
    correction?: string;
  };
};
export type ProductionScanSummary = {
  id: string;
  name: string;
  created_at: string;
  source: "simulated" | "upload";
  source_sha256: string;
  traces: number;
  flagged: number;
  malformed: number;
  excluded_sealed_overlap: string[];
  categories: DetectorCategory[];
  audit: DetectorAudit | null;
};
export type ProductionScan = Omit<ProductionScanSummary, "malformed"> & {
  malformed: { line: number; reason: string }[];
  findings: { trace: ProductionTrace; signals: ProductionSignal[] }[];
};
export type ImprovementClaim = "validated" | "uncertain" | "none";
export type ProductionVerdict = {
  baseline: { correct: number; n: number };
  checkpoint: { correct: number; n: number };
  fixed_ids: string[];
  improvement_pp: number;
  paired_bootstrap_95_pp: number[];
  improved_examples: number;
  regressed_examples: number;
  improvement_claim: ImprovementClaim;
  valid_improvement: boolean;
};
export type CampaignProduction = {
  scan_id: string;
  name: string;
  traces: number;
  flagged: number;
  categories: DetectorCategory[];
  audit: DetectorAudit | null;
  train: { id: string; trace_id: string; kinds: DetectorKind[] }[];
  held_out: {
    id: string;
    trace_id: string;
    kinds: DetectorKind[];
    question: string;
  }[];
};

// GET /evidence: each campaign's recorded results, trust checks and observability.
export type Claim = "validated" | "uncertain" | "none";
export type CheckState =
  | "pass"
  | "fail"
  | "partial"
  | "not_recorded"
  | "not_applicable";
export type Count = { correct: number; n: number };
export type EvidenceCheck = {
  id: string;
  label: string;
  state: CheckState;
  detail: string;
};
export type AgentUsage = {
  role: string;
  model: string | null;
  calls: number;
  input_tokens: number;
  output_tokens: number;
  subagents: number;
  vm_runs: number;
};
export type CampaignEvidence = {
  id: string;
  name: string | null;
  status: string;
  created_at: string | null;
  model_id: string | null;
  director: string | null;
  hardware: { backend: string | null; name: string | null } | null;
  elapsed_seconds: number | null;
  compute_seconds: number | null;
  experiments: number;
  sealed: {
    baseline: Count | null;
    checkpoint: Count | null;
    improvement_pp: number | null;
    interval_pp: [number, number] | null;
    improved: number | null;
    regressed: number | null;
    claim: Claim | null;
    checkpoint_id: string | null;
    kept_original: boolean;
  } | null;
  production: {
    baseline: Count | null;
    checkpoint: Count | null;
    improvement_pp: number | null;
    interval_pp: [number, number] | null;
    claim: Claim | null;
  } | null;
  checks: EvidenceCheck[];
  agents: {
    calls: number;
    input_tokens: number;
    output_tokens: number;
    subagent_runs: number;
    vm_runs: number;
    subagent_kinds: string[];
    by_role: AgentUsage[];
  } | null;
  relay: { sent: number; delivered: number; readback_verified: number };
  vms: { count: number; runs: number; seconds: number; deleted: number } | null;
  terse: { state_version: number | null; decisions: number } | null;
  artifacts: {
    count: number;
    verified: number;
    mismatched: number;
    missing: number;
    bytes: number;
  };
};

/**
 * Seconds left in the current training run at its measured pace: the run's own
 * elapsed time per optimizer step, times the steps remaining. Null outside
 * training or before three steps, when there is no honest rate yet.
 */
export function trainingSecondsLeft(c: Campaign): number | null {
  if (c.status !== "training" || !c.progress) return null;
  const run =
    c.experiments.find((e) => e.id === c.progress?.stage) ??
    c.experiments.at(-1);
  const last = run?.metrics.at(-1);
  if (!last || last.step < 3 || !(last.elapsed > 0)) return null;
  return (
    (last.elapsed / last.step) * Math.max(0, c.progress.total - c.progress.done)
  );
}
