#!/usr/bin/env node
// One-command setup for a fresh clone: `npm run setup`. `npm run lab` runs it first with --check,
// so a clone goes from `git clone` to a running lab with one command.
//
// Steps, each skipped when its inputs have not changed since the last successful run:
//   1. Node 24 and uv (uv installs Python 3.12 itself; with consent this installs uv).
//   2. npm dependencies (npm ci).
//   3. Python dependencies, including the training core (uv sync).
//   4. The durable actors client (durable-actors generate).
//   5. .env from .env.example, mode 600, with optional keys asked for once.
//   6. A smoke import of everything the lab loads.
//
// --check  non-interactive: run what is needed, never prompt, keep output short.
// --keys   ask for the optional keys again even if .env exists.
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  chmodSync,
  copyFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const args = new Set(process.argv.slice(2));
const CHECK = args.has("--check");
const STAMP = join(ROOT, "data", ".setup.json");
// --check stays quiet, except on the very first run, so `npm run lab` alone can set up a clone.
const FIRST_RUN = !existsSync(STAMP);
const INTERACTIVE =
  process.stdin.isTTY && process.stdout.isTTY && (!CHECK || FIRST_RUN);
const say = (s) => console.log(s);
const step = (s) => say(`\n> ${s}`);

function fail(message, fix) {
  console.error(`\nSetup stopped: ${message}${fix ? `\n  Fix: ${fix}` : ""}`);
  process.exit(1);
}

function run(cmd, argv, opts = {}) {
  const r = spawnSync(cmd, argv, {
    cwd: ROOT,
    stdio: opts.quiet ? "pipe" : "inherit",
    encoding: "utf8",
    env: opts.env ?? process.env,
  });
  if (r.error) return { ok: false, out: String(r.error.message) };
  return { ok: r.status === 0, out: `${r.stdout ?? ""}${r.stderr ?? ""}` };
}

function hash(...files) {
  const h = createHash("sha256");
  for (const f of files)
    h.update(
      existsSync(join(ROOT, f)) ? readFileSync(join(ROOT, f)) : `missing:${f}`,
    );
  return h.digest("hex").slice(0, 16);
}

function ask(question, { hidden = false } = {}) {
  return new Promise((resolve) => {
    const rl = createInterface({
      input: process.stdin,
      output: process.stdout,
      terminal: true,
    });
    if (hidden) {
      // Echo nothing while a key is typed or pasted.
      rl._writeToOutput = (s) => {
        if (s.includes(question)) process.stdout.write(s);
      };
    }
    rl.question(question, (answer) => {
      rl.close();
      if (hidden) process.stdout.write("\n");
      resolve(answer.trim());
    });
  });
}

const stamp = existsSync(STAMP) ? JSON.parse(readFileSync(STAMP, "utf8")) : {};
const done = {};
function needed(name, key) {
  done[name] = key;
  return stamp[name] !== key || args.has("--force");
}

// 1. Toolchain -----------------------------------------------------------------------------------
const major = Number(process.versions.node.split(".")[0]);
if (major < 24)
  fail(
    `Node ${process.versions.node} is too old; Forge needs Node 24.`,
    "install Node 24 (https://nodejs.org, `brew install node@24`, or `nvm install 24`), then run `npm run setup` again.",
  );

function findUv() {
  for (const uv of [
    "uv",
    join(homedir(), ".local/bin/uv"),
    join(homedir(), ".cargo/bin/uv"),
  ]) {
    if (run(uv, ["--version"], { quiet: true }).ok) return uv;
  }
  return null;
}
let uv = findUv();
if (!uv) {
  const install = "curl -LsSf https://astral.sh/uv/install.sh | sh";
  if (!INTERACTIVE)
    fail(
      "uv (the Python package manager Forge uses) is not installed.",
      `run \`${install}\`, then \`npm run setup\`.`,
    );
  step("uv is not installed");
  say(
    "Forge uses uv to install Python 3.12 and its Python packages. It installs into ~/.local/bin.",
  );
  const yes = (
    await ask(`Install uv now with \`${install}\`? [Y/n] `)
  ).toLowerCase();
  if (yes && yes !== "y" && yes !== "yes")
    fail(
      "uv is required.",
      `run \`${install}\` yourself, then \`npm run setup\`.`,
    );
  if (!run("sh", ["-c", install]).ok)
    fail(
      "the uv installer failed.",
      "see https://docs.astral.sh/uv/getting-started/installation/",
    );
  uv = findUv();
  if (!uv)
    fail(
      "uv was installed but cannot be found.",
      "open a new terminal and run `npm run setup` again.",
    );
}

// 2. Node packages ---------------------------------------------------------------------------------
if (
  needed("npm", hash("package-lock.json") + process.versions.node) ||
  !existsSync(join(ROOT, "node_modules"))
) {
  step("Installing Node packages (npm ci)");
  if (!run("npm", ["ci", "--no-audit", "--no-fund"]).ok)
    fail(
      "npm ci failed.",
      "read the npm error above; delete node_modules and run `npm run setup` again.",
    );
}

// 3. Python, packages and the training core --------------------------------------------------------
// Every optional extra, so Modal GPUs and SuperServe VMs work with no further install step, and so
// setup never uninstalls one (uv sync removes extras it is not given).
const EXTRAS = ["advanced", "modal", "superserve"];
if (
  needed(
    "python",
    hash("uv.lock", "pyproject.toml", "services/training-core/pyproject.toml") +
      EXTRAS.join(","),
  ) ||
  !existsSync(join(ROOT, ".venv/bin/python"))
) {
  step(
    "Installing Python 3.12 and Python packages (uv sync), a few minutes the first time",
  );
  if (!run(uv, ["sync", ...EXTRAS.flatMap((extra) => ["--extra", extra])]).ok)
    fail(
      "uv sync failed.",
      "read the error above, then run `npm run setup` again.",
    );
}
const py = join(ROOT, ".venv/bin/python");

// 4. Durable actors client -------------------------------------------------------------------------
if (
  needed("actors", hash("services/actors/src/actors.py", "uv.lock")) ||
  !existsSync(join(ROOT, "services/actors/generated/__init__.py"))
) {
  step("Generating the durable actors client");
  const r = run(
    join(ROOT, ".venv/bin/durable-actors"),
    [
      "generate",
      "services/actors/src/actors.py",
      "--out-dir",
      "services/actors/generated",
    ],
    { quiet: true },
  );
  if (!r.ok) fail(`durable-actors generate failed:\n${r.out.trim()}`);
}

// 5. .env ------------------------------------------------------------------------------------------
const envPath = join(ROOT, ".env");
const fresh = !existsSync(envPath);
if (fresh) {
  copyFileSync(join(ROOT, ".env.example"), envPath);
  chmodSync(envPath, 0o600);
  say("\n> Created .env from .env.example (readable only by you)");
}
function setEnv(name, value) {
  let text = readFileSync(envPath, "utf8");
  const line = `${name}=${value}`;
  text = new RegExp(`^${name}=.*$`, "m").test(text)
    ? text.replace(new RegExp(`^${name}=.*$`, "m"), line)
    : `${text.trimEnd()}\n${line}\n`;
  writeFileSync(envPath, text, { mode: 0o600 });
}
const env = Object.fromEntries(
  readFileSync(envPath, "utf8")
    .split("\n")
    .filter((l) => /^[A-Z0-9_]+=/.test(l))
    .map((l) => [l.slice(0, l.indexOf("=")), l.slice(l.indexOf("=") + 1)]),
);
if (INTERACTIVE && (fresh || args.has("--keys"))) {
  step(
    "Optional connections (press Enter to skip any of them; everything works without them)",
  );
  if (!env.OPENAI_API_KEY && !env.AZURE_OPENAI_API_KEY) {
    say(
      "With an OpenAI key, the five research agents reason with a model and can start subagents.",
    );
    say(
      "Without one, a deterministic local director runs. (Azure OpenAI: set the AZURE_OPENAI_* lines in .env.)",
    );
    const key = await ask("OpenAI API key: ", { hidden: true });
    if (key) setEnv("OPENAI_API_KEY", key);
  }
  if (env.FORGE_BACKEND !== "modal") {
    say(
      "\nModel work runs on this machine by default. Modal runs it on a cloud H100 (billed to your Modal account).",
    );
    const modal = (await ask("Use Modal for training? [y/N] ")).toLowerCase();
    if (modal === "y" || modal === "yes") {
      const signedIn = existsSync(join(homedir(), ".modal.toml"));
      if (!signedIn) {
        say("Opening the Modal sign-in page in your browser...");
        if (!run(join(ROOT, ".venv/bin/modal"), ["token", "new"]).ok)
          say(
            "Modal sign-in did not finish; run `.venv/bin/modal token new` later.",
          );
      }
      setEnv("FORGE_BACKEND", "modal");
    }
  }
}

// 6. Smoke check -----------------------------------------------------------------------------------
const smoke = run(
  py,
  [
    "-c",
    "import services.api, forge_training.distill_cli, services.actors.generated; print('ok')",
  ],
  { quiet: true },
);
if (!smoke.ok)
  fail(
    `the lab's Python modules do not import:\n${smoke.out.trim().split("\n").slice(-4).join("\n")}`,
    "run `npm run setup -- --force`.",
  );

mkdirSync(dirname(STAMP), { recursive: true });
writeFileSync(STAMP, JSON.stringify(done, null, 2));
if (CHECK && !FIRST_RUN) say("Setup: up to date.");
else if (CHECK)
  say(
    "\nForge is set up. Starting the lab... (later: `npm run add-data -- file.csv` adds your own data)",
  );
else {
  say("\nForge is set up.");
  say("  Start the lab:          npm run lab");
  say("  Add your own data:      npm run add-data -- path/to/your-data.csv");
  say("  Change keys or Modal:   npm run setup -- --keys");
}
