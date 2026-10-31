import { resolve } from "node:path";
import { bundleCampaign, reproduceBundle, verifyBundle } from "../src/bundle.js";

/**
 * Builds the Phase 8 evidence bundle of a finalized campaign (into `final/bundle`), checks a
 * copy of one, or reproduces its rehearsal into a new directory. Usage:
 *   bun scripts/bundle.ts <campaign-dir>
 *   bun scripts/bundle.ts --verify <bundle-dir>
 *   bun scripts/bundle.ts --reproduce <bundle-dir> --output <new-dir>
 * Building and reproducing use training-core's harness Python (FORGE_HARNESS_PYTHON).
 */
const [mode, target] = process.argv[2]?.startsWith("--")
  ? [process.argv[2], process.argv[3]]
  : ["--build", process.argv[2]];
const python = process.env.FORGE_HARNESS_PYTHON ?? process.env.FORGE_PYTHON;
if (!target || (mode !== "--verify" && !python))
  throw new Error(
    "Usage: FORGE_HARNESS_PYTHON=<python> bun scripts/bundle.ts [--verify | --reproduce] <directory> [--output <dir>]",
  );
if (mode === "--verify") {
  const result = await verifyBundle(resolve(target));
  console.log(JSON.stringify(result, null, 2));
  const intact =
    (result.changed as string[]).length === 0 &&
    (result.unlisted as string[]).length === 0 &&
    (result.claims as { failed: string[] }).failed.length === 0;
  process.exit(intact ? 0 : 1);
} else if (mode === "--reproduce") {
  const output = process.argv[process.argv.indexOf("--output") + 1];
  if (!process.argv.includes("--output") || !output) throw new Error("--reproduce needs --output");
  const result = await reproduceBundle(resolve(target), python!, resolve(output));
  console.log(JSON.stringify(result, null, 2));
  process.exit(result.reproduced ? 0 : 1);
} else console.log(JSON.stringify(await bundleCampaign(resolve(target), python!), null, 2));
