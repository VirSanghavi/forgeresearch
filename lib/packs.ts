// Client types and helpers for task packs, methods, teacher datasets and the overseer.

export type PackKind = "qa" | "extraction" | "preference";
export type Pack = {
  id: string;
  kind: PackKind;
  builtin: boolean;
  display: {
    name: string;
    objective: string;
    benchmark_label: string;
    scoring_label: string;
    unit: string;
  };
  capabilities: {
    terse_world: boolean;
    grpo: boolean;
    dpo: boolean;
    lm_eval_final: boolean;
    remote: boolean;
  };
  regression_title: string;
  regression_label: string;
  checker: string;
  sha256: string;
  counts: Partial<
    Record<"sealed" | "dev" | "train" | "sealed_remaining", number>
  >;
};
export type Strategy = {
  id: string;
  label: string;
  backends: string[];
  kinds: PackKind[];
};
export type ModelChoice = {
  id: string;
  label: string;
  parameters: string;
  remote_only: boolean;
  family: string;
  architecture: string | null;
  license: string | null;
  gated: boolean;
  custom: boolean;
  independent_final: boolean;
  problem: string | null;
};
export type TeacherJob = {
  id: string;
  pack: string;
  teacher: string;
  status: "running" | "completed" | "failed" | "budget_exhausted";
  created_at: string;
  local: boolean;
  progress?: { done: number; total: number; kept: number } | null;
  counts?: { prompts: number; kept: number; reasons: Record<string, number> };
  error?: string | null;
};
export type PackPreview = {
  rows: number;
  valid: number;
  invalid: number;
  invalid_examples: { line: number; reason: string }[];
  duplicates: number;
  split: Record<"sealed" | "dev" | "train", number>;
  samples: { id: string; question: string; answer: string }[];
  problems: string[];
  checker: string;
  pack_id: string;
};
export type OverseerSignal = {
  kind: string;
  label: string;
  count: number;
  rate: number;
};
export type OverseerSummary = {
  traces: number;
  sealed_overlap: number;
  flagged: number;
  signals: OverseerSignal[];
  latency_ms: { p50: number | null; p95: number | null; p99: number | null };
  negative_feedback_rate: number;
  judged: {
    total: number;
    random_sample: number;
    sample_failures: number;
    sample_failure_rate: number | null;
    sample_ci95: [number, number] | null;
    hallucinations: number;
  };
  models: Record<string, { traces: number; flagged: number }>;
  labels: number;
  spend_today: { usd: number; calls: number };
  budget: { daily_usd: number; daily_calls: number; sample_rate: number };
  judge: { ready: boolean; reason: string | null };
  alerts: {
    id: string;
    day: string;
    pack_id: string;
    kind: string;
    detail: string;
  }[];
};
export type Connection = {
  id: string;
  name: string;
  task_pack: string;
  kind: string;
  provider: string;
  host: string;
  status: string;
  detail: string;
  has_vm: boolean;
  restarts: number;
  mapping: Record<string, string>;
  fields_seen: string[];
  counts: Record<
    "fetched" | "accepted" | "duplicates" | "rejected" | "skipped",
    number
  >;
  last_success_at: string | null;
  last_seen_at: string | null;
  created_at: string;
};
export type ConnectionList = {
  connections: Connection[];
  default: string;
  providers: Record<string, { ready: boolean; reason: string | null }>;
};
export type QueueItem = {
  key: string;
  received_at: string;
  model_version: string | null;
  latency_ms: number | null;
  sampled: boolean;
  trace: {
    id: string;
    prompt: string;
    output: string;
    feedback?: { rating?: number; correction?: string };
    context?: string;
  };
  signals: { kind: string; evidence: Record<string, unknown> }[];
  judgment: {
    verdict: string;
    rationale?: string;
    hallucination?: boolean | null;
    failure_kinds?: string[];
  } | null;
};

export async function api<T>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(`/api/lab/${path}`, {
    method: body === undefined ? "GET" : "POST",
    headers:
      body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    const detail = (data as { detail?: unknown }).detail;
    throw new Error(
      typeof detail === "string" ? detail : `Request failed (${r.status})`,
    );
  }
  return data as T;
}

async function sha256(text: string) {
  const digest = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(text),
  );
  return Array.from(new Uint8Array(digest), (b) =>
    b.toString(16).padStart(2, "0"),
  ).join("");
}

// Inline below this size; larger files cross the 30-second local proxy as checksummed parts.
export const INLINE_LIMIT = 3_500_000;

export async function upload(
  name: string,
  text: string,
  onProgress?: (share: number) => void,
): Promise<string> {
  const bytes = new TextEncoder().encode(text);
  const start = await api<{ id: string; part_bytes: number }>("uploads", {
    name,
    size: bytes.length,
    sha256: await sha256(text),
  });
  const parts: string[] = [];
  let current = "";
  for (const line of text.split(/(?<=\n)/)) {
    if (current && current.length + line.length > 3_000_000) {
      parts.push(current);
      current = "";
    }
    current += line;
  }
  if (current) parts.push(current);
  for (const [index, part] of parts.entries()) {
    await api(`uploads/${start.id}/parts`, {
      index,
      text: part,
      sha256: await sha256(part),
    });
    onProgress?.((index + 1) / parts.length);
  }
  await api(`uploads/${start.id}/complete`, { parts: parts.length });
  return start.id;
}

export function columns(text: string, format: "csv" | "jsonl"): string[] {
  const first = text.split("\n").find((line) => line.trim()) ?? "";
  if (format === "csv")
    return first
      .split(",")
      .map((c) => c.trim().replace(/^"|"$/g, ""))
      .filter(Boolean);
  try {
    return Object.keys(JSON.parse(first));
  } catch {
    return [];
  }
}

export function firstValues(
  text: string,
  format: "csv" | "jsonl",
  column: string,
  n = 20,
): string[] {
  const lines = text.split("\n").filter((line) => line.trim());
  if (format === "jsonl")
    return lines.slice(0, n).flatMap((line) => {
      try {
        const v = JSON.parse(line)[column];
        return v === undefined
          ? []
          : [typeof v === "string" ? v : JSON.stringify(v)];
      } catch {
        return [];
      }
    });
  const header = columns(text, "csv");
  const index = header.indexOf(column);
  return lines.slice(1, n + 1).map((line) => line.split(",")[index] ?? "");
}
