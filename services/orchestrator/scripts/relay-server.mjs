// Real Relaycast engine, HTTP polling transport, loopback-only test server.
import { createEngine } from "@relaycast/engine";
import { createNodeRuntime } from "@relaycast/engine/adapters/node";
import { serve } from "@hono/node-server";
const runtime = createNodeRuntime({
  dbPath: process.argv[2],
  baseUrl: "http://127.0.0.1",
  fileDir: `${process.argv[2]}.files`,
});
const app = createEngine(runtime.deps);
const server = serve(
  { fetch: app.fetch, port: Number(process.argv[3] ?? 0), hostname: "127.0.0.1" },
  ({ port }) => {
    console.log(JSON.stringify({ url: `http://127.0.0.1:${port}` }));
  },
);
for (const signal of ["SIGINT", "SIGTERM"])
  process.once(signal, () => {
    server.close(() => {
      runtime.close();
      process.exit(0);
    });
  });
