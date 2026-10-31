"""Send production traces to the Forge overseer. Standard library only; never raises into your app.

    from client import Overseer
    overseer = Overseer(url="https://<project>.supabase.co/functions/v1/ingest", token=os.environ["FORGE_INGEST_TOKEN"])
    reply = overseer.chat(openai_client, model="my-model", messages=[...], model_version="v3")

Traces are queued and posted in the background in batches. If the overseer is unreachable,
traces are dropped after a few retries rather than slowing or failing your request path.
"""

import json
import queue
import threading
import time
import urllib.request
import uuid


class Overseer:
    def __init__(self, url, token, batch_size=50, flush_seconds=5.0, max_queue=10_000):
        self.url, self.token = url, token
        self.batch_size, self.flush_seconds = batch_size, flush_seconds
        self.queue = queue.Queue(maxsize=max_queue)
        self.dropped = 0
        threading.Thread(target=self._run, daemon=True).start()

    def record(self, prompt, output, **fields):
        """Queue one trace. Optional fields: finish_reason, latency_ms, tokens, model_version,
        deployment, context, feedback {"rating": -1|1, "correction": str}, metadata (≤16 keys)."""
        trace = {
            "id": fields.pop("id", None) or uuid.uuid4().hex,
            "prompt": prompt,
            "output": output,
            **fields,
        }
        try:
            self.queue.put_nowait(trace)
        except queue.Full:
            self.dropped += 1
        return trace["id"]

    def chat(self, client, model, messages, model_version=None, **kwargs):
        """Call an OpenAI-compatible client and record the exchange with latency and tokens."""
        started = time.monotonic()
        reply = client.chat.completions.create(model=model, messages=messages, **kwargs)
        try:
            choice = reply.choices[0]
            usage = getattr(reply, "usage", None)
            self.record(
                prompt=messages[-1]["content"],
                output=choice.message.content or "",
                finish_reason=choice.finish_reason,
                latency_ms=(time.monotonic() - started) * 1000,
                tokens={
                    "prompt": getattr(usage, "prompt_tokens", 0),
                    "completion": getattr(usage, "completion_tokens", 0),
                },
                model_version=model_version or model,
            )
        except Exception:  # Recording must never break the caller.
            self.dropped += 1
        return reply

    def _post(self, batch):
        body = json.dumps({"traces": batch}).encode()
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=10):
                    return
            except Exception:
                time.sleep(2**attempt)
        self.dropped += len(batch)

    def _run(self):
        while True:
            batch, deadline = [], time.monotonic() + self.flush_seconds
            while len(batch) < self.batch_size and time.monotonic() < deadline:
                try:
                    batch.append(self.queue.get(timeout=max(0.05, deadline - time.monotonic())))
                except queue.Empty:
                    pass
            if batch:
                self._post(batch)
