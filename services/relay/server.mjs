// Local Relaycast engine. `startServer` (not a bare fetch handler) serves the WebSocket
// upgrades that carry Relay's push stream; without them recipients could only poll.
import { startServer } from "@relaycast/engine/node";

const dbPath = process.argv[2] ?? "data/relaycast.db";
const requested = Number(process.argv[3] ?? 7528);
const running = startServer({
  dbPath,
  port: requested,
  baseUrl: `http://127.0.0.1:${requested}`,
  fileDir: `${dbPath}.files`,
  config: { environment: "test" },
});
// startServer binds every interface; Forge's engine is loopback-only, so refuse anything else.
const loopback = new Set(["127.0.0.1", "::1", "::ffff:127.0.0.1"]);
running.server.on("connection", (socket) => {
  if (!loopback.has(socket.remoteAddress)) socket.destroy();
});
const announce = () => {
  const address = running.server.address();
  const port =
    typeof address === "object" && address ? address.port : requested;
  console.log(JSON.stringify({ url: `http://127.0.0.1:${port}` }));
};
if (running.server.listening) announce();
else running.server.once("listening", announce);
for (const signal of ["SIGTERM", "SIGINT"])
  process.once(signal, () => running.stop().then(() => process.exit(0)));
