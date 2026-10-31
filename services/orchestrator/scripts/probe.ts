import { AgentRelay } from "@agent-relay/sdk";
import { spawn } from "node:child_process";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
const dir = await mkdtemp(join(tmpdir(), "forge-relay-probe-"));
const child = spawn("node", [join(import.meta.dir, "relay-server.mjs"), join(dir, "relay.db")], {
  stdio: ["ignore", "pipe", "inherit"],
});
try {
  const baseUrl = await new Promise<string>((resolve, reject) => {
    child.stdout.once("data", (data) => {
      try {
        resolve(JSON.parse(String(data)).url);
      } catch (e) {
        reject(e);
      }
    });
    child.once("exit", (code) => reject(new Error(`Exited ${code}`)));
  });
  const relay = await AgentRelay.createWorkspace({ name: "forge-probe", baseUrl });
  const director = await relay.workspace.register({ name: "director", type: "agent" });
  const worker = await relay.workspace.register({ name: "worker", type: "agent" });
  await director.channels.create({ name: "training" });
  await worker.channels.join("training");
  const sent = await director.sendMessage({
    to: "#training",
    text: '{"kind":"probe"}',
    idempotencyKey: "probe",
  });
  const received = [await worker.messages.get(sent.messageId)];
  console.log(
    JSON.stringify({
      sent: sent.messageId,
      received: received.map((x) => ({
        id: x.messageId,
        from: x.from,
        text: x.text,
        channel: x.channel,
        target: x.target,
      })),
    }),
  );
} finally {
  child.kill("SIGTERM");
}
