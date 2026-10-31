"""TEST DOUBLE for the orchestrator's world gateway: scores answers from a fixed table.

Real GRPO runs receive rewards only from Terse RLWorldActor transitions, through the
orchestrator (services/orchestrator/src/worlds.ts). This stub exercises the trainer's
client protocol, retries and evidence checks without actors.
"""

from __future__ import annotations

import hashlib
import json
import threading
from decimal import Decimal, InvalidOperation
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from forge_training.scoring import extract_answer


def _value(text: str) -> Decimal | None:
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


class StubWorld:
    def __init__(self, references: dict[str, str], *, constant: int | None = None):
        self.answers = {task: extract_answer(answer) for task, answer in references.items()}
        self.constant = constant
        self.token = "stub-token"
        self.requests: list[bytes] = []
        self.failures = 0  # respond 503 to this many upcoming requests
        self.drops = 0  # close this many upcoming connections without replying
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep test output quiet
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                stub.requests.append(body)
                if self.headers.get("Authorization") != f"Bearer {stub.token}":
                    return self._reply(401, {"error": "unauthorized"})
                if stub.drops:
                    stub.drops -= 1
                    self.close_connection = True
                    return
                if stub.failures:
                    stub.failures -= 1
                    return self._reply(503, {"error": "injected unavailability"})
                payload = json.loads(body)
                results = []
                for rollout in payload["rollouts"]:
                    expected = stub.answers[rollout["taskId"]]
                    answer = _value(rollout["action"]["answer"])
                    reward = (
                        stub.constant
                        if stub.constant is not None
                        else int(answer is not None and answer == expected)
                    )
                    episode = f"stub-{rollout['rolloutId']}"
                    results.append(
                        {
                            "rolloutId": rollout["rolloutId"],
                            "worldId": "stub-world",
                            "episodeId": episode,
                            "actionId": f"{episode}-a0",
                            "stateVersion": 1,
                            "stateSha256": hashlib.sha256(episode.encode()).hexdigest(),
                            "reward": reward,
                            "terminated": True,
                            "truncated": False,
                            "replayed": False,
                        }
                    )
                self._reply(200, {"results": results})

            def _reply(self, status: int, document: dict):
                data = json.dumps(document).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> StubWorld:
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()

    @property
    def environment(self) -> dict[str, str]:
        host, port = self.server.server_address[:2]
        return {
            "FORGE_WORLD_GATEWAY_URL": f"http://{host}:{port}",
            "FORGE_WORLD_GATEWAY_TOKEN": self.token,
        }
