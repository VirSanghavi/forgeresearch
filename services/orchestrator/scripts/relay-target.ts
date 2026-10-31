import { AgentRelay } from "@agent-relay/sdk";
import { randomUUID } from "node:crypto";
import { launchRelay } from "./local-relay.js";

export interface RelayTarget {
  /** `local`: a loopback engine this process owns. `remote`: an engine at `FORGE_RELAY_URL`. */
  kind: "local" | "remote";
  url: string;
  /**
   * Workspace names are idempotent on Relaycast and the key is returned only on first
   * creation, so a remote workspace name carries a random suffix.
   */
  createWorkspace(label: string): Promise<AgentRelay>;
  /** Client for an existing workspace, after `restart`. */
  connect(workspaceKey: string): AgentRelay;
  /**
   * Local: SIGKILL the engine and relaunch it on the same port and database. Remote: the
   * engine is not ours to kill, so callers drop their clients and reconnect with persisted
   * tokens, the same recovery path from this process's side.
   */
  restart(): Promise<"relay-sigkill" | "relay-client-reconnect">;
  /** Stops a local engine; deletes remote workspaces unless `FORGE_RELAY_KEEP_WORKSPACE=1`. */
  stop(): Promise<void>;
  /** Remote workspace names created by this target, for reports. Keys are never reported. */
  workspaces: string[];
}

/** `FORGE_RELAY_URL` unset: a loopback engine under `directory`. Set: that engine. */
export async function openRelay(directory: string): Promise<RelayTarget> {
  const remote = process.env.FORGE_RELAY_URL;
  if (remote) return remoteTarget(remote.replace(/\/+$/, ""));
  let server = await launchRelay(directory);
  return {
    kind: "local",
    get url() {
      return server.url;
    },
    workspaces: [],
    createWorkspace: (label) => AgentRelay.createWorkspace({ name: label, baseUrl: server.url }),
    connect: (workspaceKey) => new AgentRelay({ workspaceKey, baseUrl: server.url }),
    async restart() {
      const port = Number(new URL(server.url).port);
      const exited = new Promise((resolve) => server.process.once("exit", resolve));
      server.process.kill("SIGKILL");
      await exited;
      server = await launchRelay(directory, port);
      return "relay-sigkill";
    },
    stop: () => server.stop(),
  };
}

// The hosted engine sheds load with 503 and a Retry-After header ("The database is temporarily
// overloaded"). The SDK's default of 3 retries gives up after about 7 s; this rides out an
// overload of about two minutes instead of failing the run.
const hostedRetryPolicy = { maxRetries: 7, backoffMs: 1000, backoffMultiplier: 2, jitter: true };

function remoteTarget(url: string): RelayTarget {
  const keys: string[] = [];
  const workspaces: string[] = [];
  return {
    kind: "remote",
    url,
    workspaces,
    async createWorkspace(label) {
      const name = `${label}-${randomUUID().slice(0, 12)}`;
      const relay = await AgentRelay.createWorkspace({
        name,
        baseUrl: url,
        retryPolicy: hostedRetryPolicy,
      });
      if (!relay.workspaceKey) throw new Error(`RELAY_WORKSPACE_EXISTS: ${name} returned no key`);
      keys.push(relay.workspaceKey);
      workspaces.push(name);
      return relay;
    },
    connect: (workspaceKey) =>
      new AgentRelay({ workspaceKey, baseUrl: url, retryPolicy: hostedRetryPolicy }),
    restart: async () => "relay-client-reconnect",
    async stop() {
      if (process.env.FORGE_RELAY_KEEP_WORKSPACE === "1") return;
      for (const key of keys.splice(0)) {
        const response = await fetch(`${url}/v1/workspace`, {
          method: "DELETE",
          headers: { Authorization: `Bearer ${key}` },
        });
        // Cleanup failure must not mask the run's own result; report it instead.
        if (!response.ok && response.status !== 404)
          console.error(`Relay workspace cleanup failed: HTTP ${response.status}`);
      }
    },
  };
}
