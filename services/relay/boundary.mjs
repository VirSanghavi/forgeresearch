import { timingSafeEqual, createHash } from "node:crypto";

/**
 * The bridge speaks for every Forge agent, so only Forge may call it. A browser page can
 * reach 127.0.0.1 through DNS rebinding or a cross-site form POST; each check below closes
 * one of those doors. Returns null when the request may proceed, else { status, error }.
 */
export function refuse(req, { port, token }) {
  const host = req.headers.host;
  if (host !== `127.0.0.1:${port}` && host !== `localhost:${port}`)
    return { status: 403, error: "Untrusted host" };
  // Only server processes call the bridge; no browser origin is ever legitimate.
  if (req.headers.origin !== undefined)
    return { status: 403, error: "Untrusted request origin" };
  const path = new URL(req.url, "http://bridge").pathname;
  if (path === "/health" && req.method === "GET") return null;
  if (!token || !sameSecret(bearer(req.headers.authorization), token))
    return { status: 401, error: "Missing or invalid bridge token" };
  if (req.method === "POST") {
    const type = String(req.headers["content-type"] ?? "")
      .split(";")[0]
      .trim()
      .toLowerCase();
    if (type !== "application/json")
      return { status: 415, error: "Content-Type must be application/json" };
  }
  return null;
}

function bearer(header) {
  if (typeof header !== "string" || !header.startsWith("Bearer ")) return "";
  return header.slice(7).trim();
}

/** Constant-time comparison of fixed-length digests, so length leaks nothing either. */
function sameSecret(given, expected) {
  const a = createHash("sha256").update(given).digest();
  const b = createHash("sha256").update(expected).digest();
  return timingSafeEqual(a, b) && given.length > 0;
}
