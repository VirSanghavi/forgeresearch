// A real SDK/server round trip against a disposable loopback database.
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { createInterface } from "node:readline";
import { once } from "node:events";
import { request } from "node:http";
import { createServer } from "node:net";
import { AgentRelay } from "@agent-relay/sdk";
import { RelayCast } from "@relaycast/sdk";
import { dispatcher, pushWaiter, roles } from "../services/relay/delivery.mjs";
import { relayConfig, HOSTED_RELAY_URL } from "../services/relay/config.mjs";

// Configuration: local by default, hosted only with a workspace key, which never reaches disk.
assert.equal(relayConfig({}).hosted, false);
assert.equal(relayConfig({}).baseUrl, "http://127.0.0.1:7528");
const hosted = relayConfig({ FORGE_RELAY_WORKSPACE_KEY: "rk_live_example" });
assert.equal(hosted.baseUrl, HOSTED_RELAY_URL);
assert.ok(!hosted.workspaceId.includes("rk_live_example"));
// FORGE_RELAY_URL alone picks the hosted engine with a minted workspace; keys are per engine.
const anonymous = relayConfig({ FORGE_RELAY_URL: `${HOSTED_RELAY_URL}/` });
assert.equal(anonymous.hosted, true);
assert.equal(anonymous.workspaceId, `minted@${HOSTED_RELAY_URL}`);
assert.notEqual(relayConfig({}).workspaceId, anonymous.workspaceId);
assert.throws(
  () => relayConfig({ FORGE_RELAY_WORKSPACE_KEY: "ot_live_wrong_kind" }),
  /workspace key/,
);

const directory = await mkdtemp(join(tmpdir(), "forge-relay-check-"));
const child = spawn(
  process.execPath,
  [resolve("services/relay/server.mjs"), join(directory, "relay.db"), "0"],
  { stdio: ["ignore", "pipe", "pipe"] },
);
let errors = "";
child.stderr.on("data", (chunk) => {
  errors = (errors + chunk.toString()).slice(-3000);
});
let observer;
let bridge;
try {
  const baseUrl = await new Promise((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error(`Relay startup timed out: ${errors}`)),
      15000,
    );
    const lines = createInterface({ input: child.stdout });
    lines.on("line", (line) => {
      try {
        const parsed = JSON.parse(line);
        if (parsed.url) {
          clearTimeout(timer);
          lines.close();
          resolve(parsed.url);
        }
      } catch {
        /* Engine diagnostic line. */
      }
    });
    child.once("exit", (code) => {
      clearTimeout(timer);
      reject(new Error(`Relay exited ${code}: ${errors}`));
    });
  });
  const relay = await AgentRelay.createWorkspace({
    name: "Forge transport mechanics",
    baseUrl,
  });
  const sender = await relay.workspace.register({
    name: "sender",
    type: "agent",
  });
  const recipient = await relay.workspace.register({
    name: "recipient",
    type: "agent",
  });
  const text = JSON.stringify({
    kind: "mechanics_check",
    action: "verify recipient readback",
  });
  const first = await sender.sendMessage({
    to: "@recipient",
    text,
    idempotencyKey: "same-assignment",
  });
  const duplicate = await sender.sendMessage({
    to: "@recipient",
    text,
    idempotencyKey: "same-assignment",
  });
  assert.equal(
    first.messageId,
    duplicate.messageId,
    "Duplicate assignment created another message",
  );
  const delivered = await recipient.messages.get(first.messageId);
  assert.equal(delivered.text, text);
  assert.equal(delivered.from.name, "sender");
  // The DM sits in the recipient's durable inbox until the recipient acks it.
  const queued = (await recipient.inbox.list()).items;
  assert.equal(queued.length, 1);
  assert.equal(queued[0].metadata?.reason, "dm");
  const acked = await recipient.inbox.ack({ inboxItemId: queued[0].id });
  assert.equal(acked.supported, true);
  assert.equal((await recipient.inbox.list()).items.length, 0);

  // Push: a read-only observer stream wakes recipients instead of polling.
  const admin = new RelayCast({ apiKey: relay.workspaceKey, baseUrl });
  const token = await admin.observerTokens.create({
    name: "forge-check-push",
    scopes: [
      "stream:read",
      "messages:read",
      "dms:read",
      "channels:read",
      "threads:read",
    ],
    filters: { include_dms: true },
  });
  const push = pushWaiter();
  observer = new AgentRelay({ observerToken: token.token, baseUrl });
  observer.addListener("*", (event) => push.observe(event));
  const agents = {};
  for (const name of roles)
    agents[name] = await relay.workspace.register({ name, type: "agent" });
  await new Promise((r) => setTimeout(r, 500));
  const dispatch = dispatcher(agents, push);
  const base = {
    campaign_id: "mechanics",
    category: "TRAINING_ASSIGNED",
    experiment_id: "exp-1",
  };

  // 1:1 handoff as a direct message, drained from the trainer's inbox, acked and read.
  const assignment = {
    ...base,
    id: "first",
    sender: "director",
    recipient: "trainer",
    route: "dm",
    text: "Train exp-1 after verified delivery.",
  };
  const receipt = await dispatch.deliver(assignment);
  assert.equal(receipt.route, "dm");
  assert.equal(
    receipt.woke_by,
    "push",
    "the recipient was woken by Relay's push stream",
  );
  assert.ok(receipt.conversation_id, "a DM carries its conversation");
  assert.deepEqual(
    receipt.read_by.map((r) => r.name),
    ["trainer"],
  );
  assert.equal(receipt.deliveries[0].source, "inbox");
  assert.ok(receipt.deliveries[0].delivery_id);
  assert.deepEqual(await dispatch.deliver(assignment), receipt);
  // A lost bridge response/restart reconciles the same Relay message, not a new copy.
  const replayed = await dispatcher(agents, push).deliver(assignment);
  assert.equal(replayed.message_id, receipt.message_id);
  assert.equal(replayed.deliveries[0].source, "readback");
  await assert.rejects(
    dispatch.deliver({ ...assignment, text: "Changed instruction" }),
    /conflict/,
  );
  await assert.rejects(
    dispatcher(agents, push).deliver({
      ...assignment,
      text: "Changed after restart",
    }),
    /different request payload/,
  );
  const [a, b] = await Promise.all([
    dispatch.deliver({ ...assignment, id: "concurrent" }),
    dispatch.deliver({ ...assignment, id: "concurrent" }),
  ]);
  assert.equal(a.message_id, b.message_id);
  await assert.rejects(
    dispatch.deliver({ ...assignment, id: "self", recipient: "director" }),
    /contract/,
  );

  // Decision in the research room: every other role drains, acks and reads it.
  const decision = {
    ...base,
    id: "checkpoint",
    category: "CHECKPOINT_SAVED",
    sender: "trainer",
    recipient: "governance",
    route: "room",
    text: "exp-1 checkpoint saved.",
  };
  const posted = await dispatch.deliver(decision);
  assert.equal(posted.route, "room");
  assert.deepEqual(
    posted.read_by.map((r) => r.name).sort(),
    roles.filter((r) => r !== "trainer").sort(),
  );
  assert.ok(posted.deliveries.every((d) => d.source === "inbox"));
  // Governance answers in the checkpoint's thread.
  const verdict = await dispatch.deliver({
    ...base,
    id: "verdict",
    category: "CHECKPOINT_APPROVED",
    sender: "governance",
    recipient: "director",
    route: "room",
    reply_to: posted.message_id,
    text: "exp-1 approved.",
  });
  assert.equal(verdict.route, "thread");
  assert.equal(verdict.thread_parent_id, posted.message_id);
  const thread = await agents.director.threads.get(posted.message_id);
  assert.deepEqual(
    thread.replies.map((r) => r.messageId),
    [verdict.message_id],
  );
  await assert.rejects(
    dispatch.deliver({
      ...base,
      id: "stray",
      category: "CHECKPOINT_APPROVED",
      sender: "governance",
      recipient: "director",
      route: "room",
      reply_to: receipt.message_id,
      text: "not a room thread",
    }),
    /not in this research room/,
  );
  const reaction = await dispatch.react({
    campaign_id: "mechanics",
    sender: "governance",
    message_id: posted.message_id,
    emoji: "white_check_mark",
  });
  assert.deepEqual(reaction.reactions, [
    { emoji: "white_check_mark", count: 1, agents: ["governance"] },
  ]);
  await assert.rejects(
    dispatch.react({
      campaign_id: "mechanics",
      sender: "governance",
      message_id: posted.message_id,
      emoji: "fire",
    }),
    /contract/,
  );
  const found = await dispatch.search({
    campaign_id: "mechanics",
    q: "checkpoint",
  });
  assert.ok(found.some((row) => row.message_id === posted.message_id));

  // A corrupted inbox copy is failed in Relay and blocks the handoff; the retry verifies
  // the genuine Relay record before anyone downstream proceeds.
  const realList = agents.governance.inbox.list.bind(agents.governance.inbox);
  let corrupt = true;
  agents.governance.inbox.list = async (input) => {
    const page = await realList(input);
    if (!corrupt) return page;
    return {
      ...page,
      items: page.items.map((item) => ({
        ...item,
        message: { ...item.message, text: "tampered" },
      })),
    };
  };
  await assert.rejects(
    dispatch.deliver({ ...decision, id: "room-retry" }),
    /did not match/,
  );
  corrupt = false;
  const failed = await fetch(`${baseUrl}/v1/deliveries?status=failed`, {
    headers: { authorization: `Bearer ${agents.governance.token}` },
  }).then((r) => r.json());
  assert.equal(
    failed.data.length,
    1,
    "the bad copy is recorded as a failed Relay delivery",
  );
  const recovered = await dispatch.deliver({ ...decision, id: "room-retry" });
  assert.ok(recovered.read_by.some((r) => r.name === "governance"));
  const page = await agents.trainer.messages.list(recovered.room.name, {
    limit: 100,
  });
  assert.equal(
    page.filter((row) => row.messageId === recovered.message_id).length,
    1,
  );
  for (const role of roles)
    assert.equal(
      (await agents[role].inbox.list({ limit: 200 })).items.length,
      0,
      `${role} inbox drained`,
    );

  // The bridge's local boundary: only Forge's own server processes may drive the agents.
  const port = await new Promise((done) => {
    const probe = createServer().listen(0, "127.0.0.1", () => {
      const { port } = probe.address();
      probe.close(() => done(port));
    });
  });
  const secret = "check-" + "x".repeat(40);
  const bridgeEnv = {
    PATH: process.env.PATH,
    FORGE_RELAY_URL: baseUrl,
    FORGE_RELAY_BRIDGE_PORT: String(port),
    FORGE_RELAY_CREDENTIALS: join(directory, "bridge-credentials.json"),
  };
  const missing = spawn(
    process.execPath,
    [resolve("services/relay/bridge.mjs")],
    {
      cwd: directory,
      env: bridgeEnv,
      stdio: "ignore",
    },
  );
  assert.equal(
    (await once(missing, "exit"))[0],
    2,
    "the bridge refuses to start without a token",
  );
  bridge = spawn(process.execPath, [resolve("services/relay/bridge.mjs")], {
    cwd: directory,
    env: { ...bridgeEnv, FORGE_RELAY_BRIDGE_TOKEN: secret },
    stdio: ["ignore", "pipe", "ignore"],
  });
  await new Promise((done, fail) => {
    const timer = setTimeout(
      () => fail(new Error("Bridge startup timed out")),
      30000,
    );
    bridge.stdout.on("data", (chunk) => {
      if (String(chunk).includes("bridge ready")) {
        clearTimeout(timer);
        done();
      }
    });
  });
  const call = (method, path, headers = {}, body) =>
    new Promise((done, fail) => {
      const req = request(
        {
          host: "127.0.0.1",
          port,
          method,
          path,
          headers: { host: `127.0.0.1:${port}`, ...headers },
        },
        (res) => {
          res.resume();
          res.on("end", () => done(res.statusCode));
        },
      );
      req.on("error", fail);
      req.end(body);
    });
  const auth = { authorization: `Bearer ${secret}` };
  const search = "/search?campaign_id=mechanics&q=checkpoint";
  const reactBody = JSON.stringify({
    campaign_id: "mechanics",
    sender: "evaluator",
    message_id: posted.message_id,
    emoji: "eyes",
  });
  assert.equal(await call("GET", "/health"), 200, "health needs no token");
  assert.equal(
    await call("GET", search, { ...auth, host: `evil.example:${port}` }),
    403,
    "rebound Host",
  );
  assert.equal(
    await call("GET", search, { ...auth, origin: "https://evil.example" }),
    403,
    "foreign Origin",
  );
  assert.equal(
    await call("GET", search, { ...auth, origin: "http://127.0.0.1:3000" }),
    403,
    "any browser Origin",
  );
  assert.equal(await call("GET", search), 401, "missing token");
  assert.equal(
    await call("GET", search, { authorization: "Bearer wrong" }),
    401,
    "wrong token",
  );
  assert.equal(
    await call(
      "POST",
      "/react",
      { ...auth, "content-type": "text/plain" },
      reactBody,
    ),
    415,
    "a form-style POST",
  );
  assert.equal(await call("GET", search, auth), 200, "the legitimate caller");
  // The bridge minted its own workspace, so the legitimate write is a fresh handoff.
  const handoff = JSON.stringify({
    ...assignment,
    id: "via-bridge",
    campaign_id: "bridge-check",
  });
  assert.equal(
    await call(
      "POST",
      "/message",
      { ...auth, "content-type": "application/json" },
      handoff,
    ),
    200,
    "the legitimate caller can hand off work",
  );
  console.log(
    "Relay SDK: DM inbox ack, push wake, room read receipts, thread reply, reaction, search, duplicates, concurrent send, restart, conflicting payload, forged copy, partial-delivery recovery and the bridge's local boundary passed.",
  );
} finally {
  bridge?.kill("SIGTERM");
  await observer?.messaging?.events?.disconnect?.().catch(() => undefined);
  child.kill("SIGTERM");
  const kill = setTimeout(() => child.kill("SIGKILL"), 5000);
  if (child.exitCode === null) await once(child, "exit");
  clearTimeout(kill);
  await rm(directory, { recursive: true, force: true });
}
process.exit(0);
