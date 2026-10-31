import { spawn, type ChildProcess } from "node:child_process";
import { join } from "node:path";
import { once } from "node:events";

/** Runs the official engine in Node (its SQLite addon is not a Bun addon). */
export async function launchRelay(
  directory: string,
  port = 0,
): Promise<{ url: string; process: ChildProcess; stop(): Promise<void> }> {
  const child = spawn(
    "node",
    [join(import.meta.dir, "relay-server.mjs"), join(directory, "relay.db"), String(port)],
    {
      env: { PATH: process.env.PATH, HOME: process.env.HOME },
      stdio: ["ignore", "pipe", "pipe"],
    },
  );
  // Consume diagnostics without retaining messages, secrets or large buffers.
  child.stderr!.on("data", () => undefined);
  let ready = false;
  const url = await new Promise<string>((resolve, reject) => {
    let buffer = "";
    const timer = setTimeout(() => {
      child.kill();
      reject(new Error("Relay startup timed out"));
    }, 30000);
    child.once("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.once("exit", (code) => {
      clearTimeout(timer);
      if (!ready) reject(new Error(`Relay startup exited ${code}`));
    });
    child.stdout!.on("data", (data) => {
      if (ready) return;
      buffer += String(data);
      for (const line of buffer.split("\n")) {
        try {
          const value = JSON.parse(line);
          if (typeof value.url === "string") {
            clearTimeout(timer);
            ready = true;
            resolve(value.url);
            return;
          }
        } catch {
          /* engine diagnostics */
        }
      }
      if (buffer.length > 64000) buffer = buffer.slice(-64000);
    });
  });
  return {
    url,
    process: child,
    async stop() {
      if (child.exitCode !== null || child.signalCode !== null) return;
      const exited = once(child, "exit");
      child.kill("SIGTERM");
      const timer = setTimeout(() => child.kill("SIGKILL"), 5000);
      try {
        await exited;
      } finally {
        clearTimeout(timer);
      }
    },
  };
}
