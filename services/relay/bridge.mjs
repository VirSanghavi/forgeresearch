import { AgentRelay } from "@agent-relay/sdk";
import { RelayCast } from "@relaycast/sdk";
import { randomUUID } from "node:crypto";
import { createServer } from "node:http";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { dispatcher, pushWaiter, roles } from "./delivery.mjs";
import { relayConfig } from "./config.mjs";
import { refuse } from "./boundary.mjs";

const config = relayConfig(process.env);
// Per-launch secret shared with the bridge's only callers (scripts/lab.py hands it to the
// API, which hands it to campaign workers). Without it the bridge does not start.
const token = process.env.FORGE_RELAY_BRIDGE_TOKEN ?? "";
if (token.length < 32) {
  console.error(
    "FORGE_RELAY_BRIDGE_TOKEN (32+ characters) is required; start Forge with npm run lab",
  );
  process.exit(2);
}
await mkdir("data", { recursive: true });
// Saved credentials are used only for the same workspace on the same engine. Files from
// before this model carry only `baseUrl` (or nothing, meaning the local engine).
let saved;
try {
  const row = JSON.parse(await readFile(config.credentialsPath, "utf8"));
  const owner =
    row.workspace ?? `minted@${row.baseUrl ?? "http://127.0.0.1:7528"}`;
  if (owner === config.workspaceId) saved = row;
} catch {
  /* First boot. */
}
// A workspace you own is reached with its key from the environment, never from disk.
// Otherwise the bridge mints one; hosted names are global and idempotent (the key is
// returned only on first creation), so the name carries a random suffix.
const workspaceKey = config.workspaceKey ?? saved?.key;
// Hosted Agent Relay answers an overload with 503 and Retry-After; ride it out for about two
// minutes rather than the SDK default of about 7 s.
const retryPolicy = {
  maxRetries: 7,
  backoffMs: 1000,
  backoffMultiplier: 2,
  jitter: true,
};
const relay = workspaceKey
  ? new AgentRelay({ workspaceKey, baseUrl: config.baseUrl, retryPolicy })
  : await AgentRelay.createWorkspace({
      name: `Forge Research Lab ${randomUUID().slice(0, 8)}`,
      baseUrl: config.baseUrl,
      retryPolicy,
    });
const agents = {};
const tokens = saved?.tokens || {};
const admin = new RelayCast({
  apiKey: relay.workspaceKey,
  baseUrl: config.baseUrl,
});
// A workspace that already has Forge's agents but no saved tokens here (another machine,
// a lost credentials file) is taken over with the workspace key, keeping each identity.
async function enroll(name) {
  try {
    return await relay.workspace.register({ name, type: "agent" });
  } catch (error) {
    if (!/already exists/i.test(String(error?.message))) throw error;
    const existing = await admin.agents.get(name);
    const taken = await admin.agents.takeOver({
      name,
      expectedAgentId: existing.id,
      actor: "forge-bridge",
      reason: "Forge bridge started without this agent's saved token",
      sessionRef: "forge-bridge",
      nodeId: "forge-bridge",
    });
    return relay.workspace.reconnect({ apiToken: taken.token });
  }
}
for (const name of roles) {
  agents[name] = tokens[name]
    ? await relay.workspace
        .reconnect({ apiToken: tokens[name] })
        .catch(() => enroll(name))
    : await enroll(name);
  tokens[name] = agents[name].token ?? tokens[name];
}
// Recipients are woken by Relay's push stream. A read-only observer token carries it, so
// no participant opens its own node socket (which would ack deliveries on receipt).
let observerToken = saved?.observerToken;
if (!observerToken) {
  // The old token's secret is not on this machine: retire it, then mint a fresh one.
  for (const row of await admin.observerTokens.list())
    if (String(row?.name ?? "").startsWith("forge-bridge-push"))
      await admin.observerTokens.revoke(row.id);
  const minted = await admin.observerTokens.create({
    name: `forge-bridge-push-${randomUUID().slice(0, 8)}`,
    description: "Wakes Forge recipients when a handoff lands",
    scopes: [
      "stream:read",
      "messages:read",
      "dms:read",
      "channels:read",
      "threads:read",
    ],
    filters: { include_dms: true },
  });
  observerToken = minted.token;
}
await writeFile(
  config.credentialsPath,
  JSON.stringify({
    workspace: config.workspaceId,
    baseUrl: config.baseUrl,
    // A provided key stays in the environment only; a minted one must survive restarts.
    key: config.workspaceKey ? undefined : relay.workspaceKey,
    tokens,
    observerToken,
  }),
  { mode: 0o600 },
);

const push = pushWaiter();
const observer = new AgentRelay({ observerToken, baseUrl: config.baseUrl });
let pushes = 0;
observer.addListener("*", (event) => {
  pushes++;
  push.observe(event);
});
const dispatch = dispatcher(agents, push);

async function body(req) {
  let text = "";
  for await (const chunk of req) {
    text += chunk;
    if (text.length > 20000) throw new Error("Message too large");
  }
  return JSON.parse(text);
}

const server = createServer(async (req, res) => {
  res.setHeader("Content-Type", "application/json");
  const refused = refuse(req, { port: config.bridgePort, token });
  if (refused) {
    res.writeHead(refused.status);
    return res.end(JSON.stringify({ error: refused.error }));
  }
  try {
    const url = new URL(req.url, "http://bridge");
    if (url.pathname === "/health")
      return res.end(
        JSON.stringify({
          ok: true,
          agents: roles,
          transport: config.local
            ? "SDK 14.0.0 · Engine 12.0.0"
            : "SDK 14.0.0 · Relaycast API v1",
          local: config.local,
          endpoint: new URL(config.baseUrl).host,
          workspace: config.workspaceKey ? "provided" : "minted",
          push: { stream: "observer", events: pushes },
        }),
      );
    if (req.method === "GET" && url.pathname === "/search")
      return res.end(
        JSON.stringify(
          await dispatch.search({
            campaign_id: url.searchParams.get("campaign_id"),
            q: url.searchParams.get("q"),
          }),
        ),
      );
    if (req.method === "POST" && url.pathname === "/message")
      return res.end(JSON.stringify(await dispatch.deliver(await body(req))));
    if (req.method === "POST" && url.pathname === "/react")
      return res.end(JSON.stringify(await dispatch.react(await body(req))));
    res.writeHead(404);
    return res.end("{}");
  } catch (error) {
    res.writeHead(503);
    res.end(JSON.stringify({ error: error.message }));
  }
});
server.listen(config.bridgePort, "127.0.0.1", () =>
  console.log(
    `Forge Agent Relay bridge ready on ${config.bridgePort} (${config.local ? "local" : "hosted"} engine, ${config.workspaceKey ? "provided" : "minted"} workspace)`,
  ),
);
for (const signal of ["SIGTERM", "SIGINT"])
  process.once(signal, () => {
    server.close();
    process.exit(0);
  });
