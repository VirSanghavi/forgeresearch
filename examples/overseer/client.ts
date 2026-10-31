// Send production traces to the Forge overseer from Node, Deno or the browser's server side.
// Traces are batched in the background; failures are dropped, never thrown into your app.
//
//   const overseer = new Overseer(process.env.FORGE_INGEST_URL!, process.env.FORGE_INGEST_TOKEN!);
//   overseer.record({ prompt, output, finish_reason, latency_ms, model_version: "v3" });

export type Trace = {
  id?: string;
  prompt: string;
  output: string;
  finish_reason?: string;
  latency_ms?: number;
  tokens?: { prompt?: number; completion?: number };
  model_version?: string;
  deployment?: string;
  context?: string;
  feedback?: { rating?: number; correction?: string };
  metadata?: Record<string, string | number | boolean | null>;
};

export class Overseer {
  private pending: Trace[] = [];
  private timer: ReturnType<typeof setTimeout> | null = null;
  dropped = 0;

  constructor(
    private url: string,
    private token: string,
    private batchSize = 50,
    private flushMs = 5000,
  ) {}

  record(trace: Trace): string {
    const id = trace.id ?? crypto.randomUUID();
    if (this.pending.length >= 10_000) {
      this.dropped += 1;
      return id;
    }
    this.pending.push({ ...trace, id });
    if (this.pending.length >= this.batchSize) void this.flush();
    else this.timer ??= setTimeout(() => void this.flush(), this.flushMs);
    return id;
  }

  async flush(): Promise<void> {
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    const batch = this.pending.splice(0, 500);
    if (!batch.length) return;
    for (let attempt = 0; attempt < 3; attempt += 1) {
      try {
        const response = await fetch(this.url, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${this.token}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ traces: batch }),
        });
        if (response.ok || response.status === 422) return;
      } catch {
        // Retry below.
      }
      await new Promise((resolve) => setTimeout(resolve, 1000 * 2 ** attempt));
    }
    this.dropped += batch.length;
  }
}
