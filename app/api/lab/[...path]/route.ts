import { NextRequest } from "next/server";
export const dynamic = "force-dynamic";
const port = process.env.PORT || "3000";
const trustedHosts = new Set([`127.0.0.1:${port}`, `localhost:${port}`]);
const trustedOrigins = new Set([...trustedHosts].map((h) => `http://${h}`));
async function proxy(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> },
) {
  const { path } = await context.params;
  // Check Host on every method: a DNS-rebound page reaches this server under
  // its own hostname with a matching Origin, so Origin === Host proves nothing.
  if (!trustedHosts.has(request.headers.get("host") ?? "")) {
    // A hosted copy (Vercel) serves the lab's interface only: the research engine,
    // Terse actors and training worker run on the operator's machine.
    if (process.env.VERCEL) {
      return Response.json(
        {
          detail:
            "This hosted copy has no research engine attached. The lab runs on your own machine: clone Forge and run npm run lab.",
        },
        { status: 503 },
      );
    }
    return Response.json({ detail: "Untrusted host" }, { status: 403 });
  }
  const origin = request.headers.get("origin");
  if (origin && !trustedOrigins.has(origin)) {
    return Response.json(
      { detail: "Untrusted request origin" },
      { status: 403 },
    );
  }
  const endpoint = new URL(
    path.map(encodeURIComponent).join("/"),
    (process.env.FORGE_API_URL || "http://127.0.0.1:8000") + "/",
  );
  endpoint.search = request.nextUrl.search;
  try {
    const upstream = await fetch(endpoint, {
      method: request.method,
      headers: { "Content-Type": "application/json" },
      body: request.method === "POST" ? await request.text() : undefined,
      cache: "no-store",
      signal: AbortSignal.timeout(30000),
    });
    const headers = new Headers();
    for (const key of ["content-type", "content-disposition"]) {
      const value = upstream.headers.get(key);
      if (value) headers.set(key, value);
    }
    return new Response(upstream.body, { status: upstream.status, headers });
  } catch {
    return Response.json(
      {
        detail:
          "The research engine is unavailable. Start the lab services, then retry.",
      },
      { status: 503 },
    );
  }
}
export { proxy as GET, proxy as POST };
