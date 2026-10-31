import { expect, test } from "bun:test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { exists, pythonJson } from "../src/worker.js";

const python = process.env.FORGE_PYTHON!;

test("Python JSON preserves UTF-8 characters split across pipe chunks", async () => {
  const report = await pythonJson(python, [
    "-c",
    String.raw`
import os, time
os.write(1, b'{"value":"\xc3')
time.sleep(0.1)
os.write(1, b'\xa9"}')
`,
  ]);
  expect(report).toEqual({ value: "é" });
});

test("Python JSON drains large reports before parsing the result", async () => {
  const report = await pythonJson(python, [
    "-c",
    "import json; print(json.dumps({'value':'x'*2000000}))",
  ]);
  expect(String(report.value)).toHaveLength(2000000);
});

test("an already cancelled assignment never starts a Python worker", async () => {
  const directory = await mkdtemp(join(tmpdir(), "forge-cancelled-"));
  const marker = join(directory, "started");
  try {
    await expect(
      pythonJson(
        python,
        ["-c", "import pathlib,sys; pathlib.Path(sys.argv[1]).touch(); print('{}')", marker],
        {
          signal: AbortSignal.abort(),
        },
      ),
    ).rejects.toThrow("cancelled");
    expect(await exists(marker)).toBe(false);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("timed-out workers stop before the call returns", async () => {
  const directory = await mkdtemp(join(tmpdir(), "forge-timeout-"));
  const marker = join(directory, "finished");
  try {
    await expect(
      pythonJson(
        python,
        [
          "-c",
          "import pathlib,sys,time; time.sleep(0.5); pathlib.Path(sys.argv[1]).touch(); print('{}')",
          marker,
        ],
        {
          timeoutMs: 100,
        },
      ),
    ).rejects.toThrow("cancelled");
    await Bun.sleep(600);
    expect(await exists(marker)).toBe(false);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
