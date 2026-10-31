import { createHash } from "node:crypto";

export const HOSTED_RELAY_URL = "https://cast.agentrelay.com";

/**
 * Where Forge's agents talk, as one model:
 * - FORGE_RELAY_URL picks the engine. Unset or loopback: the local Relaycast engine that
 *   `npm run lab` starts. https://cast.agentrelay.com: hosted Agent Relay.
 * - FORGE_RELAY_WORKSPACE_KEY (rk_live_...) picks a workspace you own on that engine; with
 *   no URL it implies the hosted engine. It is never written to disk.
 * - Without a key, the bridge mints a workspace on the engine and saves that key, keyed by
 *   engine, so a key minted for one engine is never replayed against another.
 */
export function relayConfig(env) {
  const workspaceKey = env.FORGE_RELAY_WORKSPACE_KEY?.trim() || undefined;
  if (workspaceKey && !workspaceKey.startsWith("rk_"))
    throw new Error(
      "FORGE_RELAY_WORKSPACE_KEY must be an Agent Relay workspace key (rk_live_...)",
    );
  const baseUrl = (
    env.FORGE_RELAY_URL?.trim() ||
    (workspaceKey ? HOSTED_RELAY_URL : "http://127.0.0.1:7528")
  ).replace(/\/+$/, "");
  const local = ["127.0.0.1", "localhost"].includes(new URL(baseUrl).hostname);
  // The port comes from FORGE_RELAY_BRIDGE_PORT, else from FORGE_RELAY_BRIDGE (the address the
  // API posts to), so a second lab can run on other ports.
  const bridgePort = Number(
    env.FORGE_RELAY_BRIDGE_PORT ||
      (env.FORGE_RELAY_BRIDGE ? new URL(env.FORGE_RELAY_BRIDGE).port : "") ||
      7529,
  );
  if (!Number.isInteger(bridgePort) || bridgePort <= 0)
    throw new Error("Invalid FORGE_RELAY_BRIDGE_PORT");
  // Saved agent tokens belong to one workspace on one engine; a fingerprint (never the
  // key) names a provided workspace.
  const workspaceId = workspaceKey
    ? `key:${createHash("sha256").update(workspaceKey).digest("hex").slice(0, 16)}@${baseUrl}`
    : `minted@${baseUrl}`;
  return {
    local,
    hosted: !local,
    workspaceKey,
    baseUrl,
    bridgePort,
    workspaceId,
    credentialsPath:
      env.FORGE_RELAY_CREDENTIALS || "data/relay-credentials.json",
  };
}
