import { resolve, join } from "node:path";
import { runScenario } from "./scenario.js";
const python = process.env.FORGE_PYTHON;
if (!python)
  throw new Error("Set FORGE_PYTHON to the absolute training-core virtualenv Python executable");
const output = resolve(
  process.argv[2] ?? join(import.meta.dir, "../artifacts", `demo-${Date.now()}`),
);
const report = await runScenario(output, python);
console.log(JSON.stringify({ output, ...report }, null, 2));
