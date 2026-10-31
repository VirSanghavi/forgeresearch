import { generateKeyPairSync, randomBytes, type KeyObject } from "node:crypto";
import { mkdir } from "node:fs/promises";
import { resolve } from "node:path";
import { startLocalActors, type LocalActorRuntime } from "durable-actors/dev";
import { ForgeActorClient } from "../src/client.js";
import type { Role } from "../src/contracts.js";

export interface LocalAuthority {
  projectId: string;
  transportKey: string;
  coordinationSecret: string;
  rewardSecret: string;
  identities: Record<Role, { keyId: string; privateKey: KeyObject }>;
  publicKeys: Record<string, { role: Role; publicKeyPem: string; campaignIds: string[] }>;
}
export function localAuthority(campaignIds: string[]): LocalAuthority {
  const identities = {} as LocalAuthority["identities"];
  const publicKeys: LocalAuthority["publicKeys"] = {};
  const roles: Role[] = [
    "owner",
    "director",
    "coordinator",
    "worker",
    "evaluator",
    "reviewer",
    "data",
    "final_evaluator",
  ];
  for (const role of roles) {
    const keys = generateKeyPairSync("ed25519"),
      keyId = `${role}-local`;
    identities[role] = { keyId, privateKey: keys.privateKey };
    publicKeys[keyId] = {
      role,
      campaignIds,
      publicKeyPem: keys.publicKey.export({ type: "spki", format: "pem" }).toString(),
    };
  }
  return {
    projectId: "forge-local",
    identities,
    publicKeys,
    transportKey: randomBytes(32).toString("hex"),
    coordinationSecret: randomBytes(32).toString("hex"),
    rewardSecret: randomBytes(32).toString("hex"),
  };
}

/** Starts the real, pinned native runtime with only an explicit server environment.
 * Use serially: temporary process.env changes are local to this launcher process.
 */
export async function launch(
  authority: LocalAuthority,
  dataDir: string,
  quiet = true,
  /** Restart on a previous runtime's port so existing clients reconnect transparently. */
  port?: number,
): Promise<LocalActorRuntime> {
  await mkdir(dataDir, { recursive: true });
  const previous = { ...process.env };
  const allowed = ["PATH", "HOME", "USER", "TMPDIR", "TMP", "TEMP", "LANG", "LC_ALL"];
  try {
    for (const name of Object.keys(process.env))
      if (!allowed.includes(name)) delete process.env[name];
    Object.assign(process.env, {
      PATH: `${resolve(import.meta.dir, "../node_modules/.bin")}:${previous.PATH ?? ""}`,
      FORGE_ACTOR_PROJECT_ID: authority.projectId,
      FORGE_ACTOR_PUBLIC_KEYS: JSON.stringify(authority.publicKeys),
      FORGE_ACTOR_TRANSPORT_KEY: authority.transportKey,
      FORGE_COORDINATION_SECRET: authority.coordinationSecret,
      FORGE_REWARD_SECRET: authority.rewardSecret,
      DURABLE_ACTORS_TELEMETRY: "0",
    });
    return await startLocalActors({
      project: resolve(import.meta.dir, ".."),
      entrypoint: "src/actors.ts",
      dataDir: resolve(dataDir),
      projectId: authority.projectId,
      apiKey: authority.transportKey,
      startupTimeoutMs: 60_000,
      quiet,
      ...(port === undefined ? {} : { port }),
    });
  } finally {
    for (const name of Object.keys(process.env)) delete process.env[name];
    Object.assign(process.env, previous);
  }
}
export function clients(
  authority: LocalAuthority,
  runtime: LocalActorRuntime,
): Record<Role, ForgeActorClient> {
  return Object.fromEntries(
    Object.entries(authority.identities).map(([role, identity]) => [
      role,
      new ForgeActorClient(
        {
          controlPlaneUrl: runtime.connection.controlPlaneUrl,
          projectId: authority.projectId,
          apiKey: authority.transportKey,
        },
        identity,
      ),
    ]),
  ) as Record<Role, ForgeActorClient>;
}
