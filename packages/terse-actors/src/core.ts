import { createHash, createHmac, createPublicKey, timingSafeEqual, verify } from "node:crypto";
import { z } from "zod";
import type { ActorDatabase } from "durable-actors";
import { createActorTransport } from "durable-actors/backend";
import {
  grantSchema,
  id,
  integer,
  type ActorName,
  type Document,
  type Journal,
  type ReservationGrant,
  type Role,
  type SignedRequest,
} from "./contracts.js";

export function fail(code: string, message: string): never {
  throw new Error(`${code}: ${message}`);
}
export function canonical(value: unknown): string {
  if (value === null || typeof value === "boolean" || typeof value === "string")
    return JSON.stringify(value);
  if (typeof value === "number" && Number.isFinite(value)) return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (typeof value === "object" && value !== null) {
    return `{${Object.keys(value)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonical((value as Record<string, unknown>)[key])}`)
      .join(",")}}`;
  }
  return fail("INVALID_JSON", "Only finite JSON values are accepted");
}
export function digest(value: unknown): string {
  return createHash("sha256").update(canonical(value)).digest("hex");
}
export function document(value: unknown): Document {
  return JSON.parse(canonical(value)) as Document;
}
export function journal(): Journal {
  return { version: 0 };
}
export function scopedId(kind: "exp" | "world", campaignId: string, label: string): string {
  id.parse(campaignId);
  z.string()
    .regex(/^[A-Za-z0-9][A-Za-z0-9_.-]{0,39}$/)
    .parse(label);
  return `${kind}-${digest(campaignId).slice(0, 32)}-${label}`;
}
export function requireScopedId(kind: "exp" | "world", campaignId: string, actorId: string): void {
  if (!actorId.startsWith(`${kind}-${digest(campaignId).slice(0, 32)}-`))
    fail("ACTOR_NAMESPACE", "Use a campaign-scoped actor ID");
}

const envelopeSchema = z.strictObject({
  version: z.literal(1),
  projectId: id,
  actorName: z.enum([
    "FinalEvaluationActor",
    "ResearchLedgerActor",
    "ResearchWorkflowActor",
    "ResearchCampaignActor",
    "ExperimentActor",
    "GovernanceActor",
    "ArtifactRegistryActor",
    "RLWorldActor",
  ]),
  actorId: id,
  campaignId: id,
  method: id,
  keyId: id,
  requestId: id,
  issuedAtMs: integer,
  expiresAtMs: integer,
  payload: z.record(z.string(), z.json()),
  signature: z.string().regex(/^[A-Za-z0-9_-]{86}$/),
});
const trustSchema = z.record(
  id,
  z.strictObject({
    role: z.enum([
      "owner",
      "director",
      "coordinator",
      "worker",
      "evaluator",
      "reviewer",
      "data",
      "final_evaluator",
    ]),
    publicKeyPem: z.string().min(1),
    campaignIds: z.array(id).min(1),
  }),
);
export interface Authorization {
  request: SignedRequest;
  keyId: string;
  role: Role;
  fingerprint: string;
}
export function authorize<T>(
  input: SignedRequest,
  actorName: ActorName,
  actorId: string,
  method: string,
  roles: readonly Role[],
  schema: z.ZodType<T>,
): Authorization & { payload: T } {
  if (canonical(input).length > 64 * 1024) fail("PAYLOAD_TOO_LARGE", "Request exceeds 64 KiB");
  const request = envelopeSchema.parse(input) as SignedRequest;
  if (
    request.projectId !== process.env.FORGE_ACTOR_PROJECT_ID ||
    request.actorName !== actorName ||
    request.actorId !== actorId ||
    request.method !== method
  )
    fail("SCOPE_MISMATCH", "Request is bound to another destination");
  const keys = trustSchema.parse(JSON.parse(process.env.FORGE_ACTOR_PUBLIC_KEYS ?? "{}"));
  const key = keys[request.keyId];
  if (!key || !roles.includes(key.role) || !key.campaignIds.includes(request.campaignId)) {
    fail("FORBIDDEN", "Signing identity does not have the required role and campaign scope");
  }
  const { signature, ...unsigned } = request;
  if (
    !verify(
      null,
      Buffer.from(canonical(unsigned)),
      createPublicKey(key.publicKeyPem),
      Buffer.from(signature, "base64url"),
    )
  ) {
    fail("INVALID_SIGNATURE", "Request signature could not be verified");
  }
  const now = Date.now();
  if (
    request.issuedAtMs > now + 5000 ||
    request.expiresAtMs <= now ||
    request.expiresAtMs <= request.issuedAtMs ||
    request.expiresAtMs - request.issuedAtMs > 300_000
  )
    fail("EXPIRED_REQUEST", "Request validity window is invalid");
  const payload = schema.parse(request.payload);
  const { issuedAtMs: _issued, expiresAtMs: _expires, ...operation } = unsigned;
  return { request, payload, keyId: request.keyId, role: key.role, fingerprint: digest(operation) };
}
/**
 * The journal (events) and idempotency receipts live in actor SQLite, in the same
 * transaction as the persisted domain state. Persisted fields are rewritten and re-read
 * whole on every invocation, so keeping these append-only logs out of them makes each
 * command independent of history length.
 */
const MAX_JOURNAL = 1_000_000;
function journalTables(db: ActorDatabase): void {
  // Not cached: a table created inside an invocation that later fails is rolled back.
  db.exec(
    "CREATE TABLE IF NOT EXISTS forge_receipts (request_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, result TEXT NOT NULL)",
  );
  db.exec(
    "CREATE TABLE IF NOT EXISTS forge_events (sequence INTEGER PRIMARY KEY, event TEXT NOT NULL)",
  );
}
export function duplicate(db: ActorDatabase, auth: Authorization): Document | undefined {
  journalTables(db);
  const [receipt] = db.exec<{ fingerprint: string; result: string }>(
    "SELECT fingerprint, result FROM forge_receipts WHERE request_id = ?",
    auth.request.requestId,
  );
  if (!receipt) return undefined;
  if (receipt.fingerprint !== auth.fingerprint)
    fail("IDEMPOTENCY_CONFLICT", "Request ID was already used for different work");
  return JSON.parse(receipt.result) as Document;
}
export function commit<S extends Journal>(
  db: ActorDatabase,
  state: S,
  auth: Authorization,
  type: string,
  result: Document,
  payload: Document,
): S {
  if (state.version >= MAX_JOURNAL)
    fail("STATE_LIMIT", "Actor journal reached its configured bound");
  const version = state.version + 1;
  const event = {
    eventId: `${auth.request.actorName}:${auth.request.actorId}:${version}`,
    campaignId: auth.request.campaignId,
    actorName: auth.request.actorName,
    actorId: auth.request.actorId,
    sequence: version,
    timestampMs: Date.now(),
    type,
    payload,
  };
  journalTables(db);
  db.exec(
    "INSERT INTO forge_receipts (request_id, fingerprint, result) VALUES (?, ?, ?)",
    auth.request.requestId,
    auth.fingerprint,
    canonical(result),
  );
  db.exec("INSERT INTO forge_events (sequence, event) VALUES (?, ?)", version, canonical(event));
  return { ...state, version };
}
export function events(db: ActorDatabase, version: number, after: number, limit: number): Document {
  integer.parse(after);
  z.number().int().min(1).max(100).parse(limit);
  // Reads never create tables: an actor without a journal simply has no events.
  const exists = db.exec(
    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'forge_events'",
  ).length;
  const rows = exists
    ? db.exec<{ event: string }>(
        "SELECT event FROM forge_events WHERE sequence > ? ORDER BY sequence LIMIT ?",
        after,
        limit,
      )
    : [];
  return document({
    events: rows.map((row) => JSON.parse(row.event) as Document),
    latestSequence: version,
  });
}
function secret(name: string): string {
  const value = process.env[name];
  if (!value || !/^[a-f0-9]{64}$/.test(value))
    fail("SERVER_CONFIGURATION", `${name} must be a 256-bit hex key`);
  return value;
}
export function signGrant(claims: ReservationGrant["claims"]): ReservationGrant {
  return {
    claims,
    signature: createHmac("sha256", secret("FORGE_COORDINATION_SECRET"))
      .update("forge/reservation/v1\n")
      .update(canonical(claims))
      .digest("hex"),
  };
}
export function verifyGrant(input: unknown, actorId: string, campaignId: string): ReservationGrant {
  const grant = grantSchema.parse(input);
  if (
    grant.claims.projectId !== process.env.FORGE_ACTOR_PROJECT_ID ||
    grant.claims.experimentId !== actorId ||
    grant.claims.proposal.experimentId !== actorId ||
    grant.claims.campaignId !== campaignId
  ) {
    fail("GRANT_SCOPE", "Reservation is for another experiment or campaign");
  }
  if (
    !timingSafeEqual(
      Buffer.from(grant.signature, "hex"),
      Buffer.from(signGrant(grant.claims).signature, "hex"),
    )
  ) {
    fail("INVALID_GRANT", "Reservation was not issued by the campaign authority");
  }
  return grant;
}
export async function actorRead(
  actorName: ActorName,
  actorId: string,
  method: string,
  args: unknown[] = [],
): Promise<unknown> {
  const transport = createActorTransport({
    controlPlaneUrl: process.env.DURABLE_ACTORS_CONTROL_PLANE_URL ?? "",
    projectId: process.env.FORGE_ACTOR_PROJECT_ID,
    apiKey: process.env.FORGE_ACTOR_TRANSPORT_KEY,
  });
  return transport.invoke(actorName, actorId, method, args);
}
export function numericAnswer(value: string): string | null {
  if (!/^[+-]?(?:\d{1,24}(?:\.\d{1,12})?|\.\d{1,12})$/.test(value.trim())) return null;
  const trimmed = value.trim();
  const negative = trimmed.startsWith("-");
  const [whole, fraction = ""] = trimmed.replace(/^[+-]/, "").split(".");
  const integerPart = (whole || "0").replace(/^0+(?=\d)/, "");
  const fractionPart = fraction.replace(/0+$/, "");
  const absolute = `${integerPart}${fractionPart ? `.${fractionPart}` : ""}`;
  return `${negative && absolute !== "0" ? "-" : ""}${absolute}`;
}
export function rewardCommitment(
  campaignId: string,
  worldId: string,
  taskId: string,
  taskVersion: string,
  seed: number,
  question: string,
  answer: string,
  /** The trusted task service passes its copy of the server reward secret explicitly. */
  rewardSecret: string = secret("FORGE_REWARD_SECRET"),
): string {
  const normalized = numericAnswer(answer);
  if (normalized === null) fail("INVALID_ANSWER", "Answer must be a bounded decimal");
  if (!/^[a-f0-9]{64}$/.test(rewardSecret))
    fail("SERVER_CONFIGURATION", "Reward secret must be a 256-bit hex key");
  return createHmac("sha256", rewardSecret)
    .update("forge/math/v1\n")
    .update(
      canonical({ campaignId, worldId, taskId, taskVersion, seed, question, answer: normalized }),
    )
    .digest("hex");
}
