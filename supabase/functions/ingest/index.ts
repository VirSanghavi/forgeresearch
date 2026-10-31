// Production trace intake for the Forge overseer.
// POST {"traces": [...]} (or one trace object) with "Authorization: Bearer <ingest token>".
// The token is checked by its SHA-256, each trace is validated, the token's daily cap is
// enforced, and rows are inserted idempotently per (source, trace id). Nothing is judged here
// and no model or judge credentials live in this function.

const MAX_BODY = 4 * 1024 * 1024;
const MAX_BATCH = 500;
const MAX_TEXT = 32000;
const URL_BASE = `${Deno.env.get("SUPABASE_URL")}/rest/v1`;
const KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "";
const HEADERS = {
  apikey: KEY,
  Authorization: `Bearer ${KEY}`,
  "Content-Type": "application/json",
};

type Trace = Record<string, unknown>;

function reply(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

async function sha256(text: string): Promise<string> {
  const digest = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(text),
  );
  return Array.from(new Uint8Array(digest), (b) =>
    b.toString(16).padStart(2, "0"),
  ).join("");
}

function check(trace: Trace): string | null {
  if (typeof trace !== "object" || trace === null || Array.isArray(trace))
    return "not an object";
  for (const key of ["id", "prompt", "output"]) {
    const value = trace[key];
    if (typeof value !== "string" || !value.trim()) return `missing ${key}`;
    if (value.length > MAX_TEXT) return `${key} too long`;
  }
  if ((trace.id as string).length > 200) return "id too long";
  if (typeof trace.context === "string" && trace.context.length > MAX_TEXT)
    return "context too long";
  if (trace.metadata && Object.keys(trace.metadata as object).length > 16)
    return "at most 16 metadata keys";
  return null;
}

Deno.serve(async (request) => {
  if (request.method !== "POST") return reply(405, { error: "POST traces" });
  const token = (request.headers.get("authorization") ?? "").replace(
    /^Bearer\s+/i,
    "",
  );
  if (!token) return reply(401, { error: "Missing ingest token" });
  const found = await fetch(
    `${URL_BASE}/forge_ingest_tokens?select=source,pack_id,daily_cap&revoked=eq.false&token_sha256=eq.${await sha256(token)}`,
    { headers: HEADERS },
  );
  const [owner] = found.ok ? await found.json() : [];
  if (!owner) return reply(401, { error: "Unknown or revoked ingest token" });

  const text = await request.text();
  if (text.length > MAX_BODY) return reply(413, { error: "Body over 4 MB" });
  let body: unknown;
  try {
    body = JSON.parse(text);
  } catch {
    return reply(400, { error: "Body is not JSON" });
  }
  const batch: Trace[] = Array.isArray((body as { traces?: unknown }).traces)
    ? (body as { traces: Trace[] }).traces
    : [body as Trace];
  if (batch.length > MAX_BATCH)
    return reply(413, { error: `At most ${MAX_BATCH} traces per request` });

  const since = new Date(Date.now() - 24 * 3600 * 1000).toISOString();
  const counted = await fetch(
    `${URL_BASE}/forge_traces?select=id&source=eq.${encodeURIComponent(owner.source)}&received_at=gte.${since}`,
    { method: "HEAD", headers: { ...HEADERS, Prefer: "count=exact" } },
  );
  const used = Number(
    (counted.headers.get("content-range") ?? "*/0").split("/")[1] ?? 0,
  );

  const rejected: { index: number; reason: string }[] = [];
  const rows = [];
  for (const [index, trace] of batch.entries()) {
    const problem = check(trace);
    if (problem) rejected.push({ index, reason: problem });
    else
      rows.push({
        source: owner.source,
        trace_id: trace.id,
        pack_id: owner.pack_id,
        payload: trace,
      });
  }
  if (used + rows.length > owner.daily_cap)
    return reply(429, { error: "Daily ingest cap reached", used });
  if (!rows.length) return reply(422, { accepted: 0, duplicates: 0, rejected });

  const inserted = await fetch(
    `${URL_BASE}/forge_traces?on_conflict=source,trace_id`,
    {
      method: "POST",
      headers: {
        ...HEADERS,
        Prefer: "resolution=ignore-duplicates,return=representation",
      },
      body: JSON.stringify(rows),
    },
  );
  if (!inserted.ok) return reply(502, { error: "Could not store traces" });
  const accepted = (await inserted.json()).length;
  return reply(202, { accepted, duplicates: rows.length - accepted, rejected });
});
