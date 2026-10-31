#!/usr/bin/env node
// Forge glyphs: each concept is drawn by hand on a 12 x 12 grid and rendered as a
// matrix of square cells, like the status panel of an accelerator. The body is
// lit hottest at its core and cools toward the rim in three fixed steps, so the
// glyph reads as one object and stays crisp at 18px.
//
//   node scripts/gen-glyphs.mjs
//
// Writes components/glyphs.ts (paths the lab renders inline in its theme color)
// and public/glyphs/<name>.svg (standalone copies for the site and documents).
//
// In a drawing, "#" is the body, "+" a part drawn one step cooler, "-" two
// steps cooler, and "." is empty.
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const N = 12;
// Opacity of the core, middle and rim; "+" cells start one step down.
const HEAT = [1, 0.72, 0.46];
const CELL = 2 / 3; // cell side in grid units; leaves a one-third gap

const GLYPHS = {
  // Training briefing: a sheet with its corner folded and three lines of text.
  briefing: [
    "............",
    "..######....",
    "..#######...",
    "..########..",
    "..#......#..",
    "..########..",
    "..#......#..",
    "..########..",
    "..#....####.",
    "..########..",
    "..########..",
    "............",
  ],
  // Agent chat: the reply in front of the message it answers.
  chat: [
    "............",
    ".++++++.....",
    ".++++++.....",
    ".++++++.....",
    ".++.#######.",
    ".+..#######.",
    "....#######.",
    "....#######.",
    "....#######.",
    "......##....",
    "......#.....",
    "............",
  ],
  // Production failures: logged traces, one of them broken.
  production: [
    "............",
    ".##########.",
    ".##########.",
    "............",
    ".####..####.",
    ".####..####.",
    "............",
    ".++++++++++.",
    ".++++++++++.",
    "............",
    ".++++++++...",
    "............",
  ],
  // Production overseer: an eye that never closes.
  overseer: [
    "............",
    "............",
    "....####....",
    "..########..",
    ".###....###.",
    "###..##..###",
    "###..##..###",
    ".###....###.",
    "..########..",
    "....####....",
    "............",
    "............",
  ],
  // Durable actors: stacked state that survives a crash.
  terse: [
    "............",
    "..########..",
    ".##########.",
    ".##########.",
    "............",
    ".##########.",
    ".##########.",
    "............",
    ".++++++++++.",
    ".++++++++++.",
    "..++++++++..",
    "............",
  ],
  // Overview: one large tile and two small ones.
  overview: [
    "............",
    ".#####.++++.",
    ".#####.++++.",
    ".#####.++++.",
    ".#####......",
    ".#####.++++.",
    ".#####.++++.",
    ".#####.++++.",
    "............",
    ".##########.",
    ".##########.",
    "............",
  ],
  // Experiments: a flask, half full.
  experiments: [
    "............",
    "....####....",
    ".....++.....",
    ".....++.....",
    "....+..+....",
    "...+....+...",
    "...######...",
    "..########..",
    ".##########.",
    ".##########.",
    "..########..",
    "............",
  ],
  // Environments: a world as a cube, its three faces lit differently.
  environments: [
    "............",
    ".....##.....",
    "...######...",
    ".##########.",
    ".++++##----.",
    ".+++++-----.",
    ".+++++-----.",
    ".+++++-----.",
    ".+++++-----.",
    "..++++----..",
    "....++--....",
    "............",
  ],
  // Governance: a shield with a slot where the veto goes.
  governance: [
    "............",
    ".....##.....",
    "..########..",
    ".##########.",
    ".##########.",
    ".####..####.",
    ".####..####.",
    "..########..",
    "..########..",
    "...######...",
    ".....##.....",
    "............",
  ],
  // Results: three measurements, the last one higher.
  results: [
    "............",
    "........###.",
    "........###.",
    "........###.",
    "....###.###.",
    "....###.###.",
    "....###.###.",
    "+++.###.###.",
    "+++.###.###.",
    "+++.###.###.",
    "............",
    "............",
  ],
  // Evidence: a seal with a check struck into it.
  evidence: [
    "............",
    "....####....",
    "..########..",
    ".#######..#.",
    ".######..##.",
    ".#..##..###.",
    ".##....####.",
    ".###..#####.",
    "..########..",
    "...+....+...",
    "..++....++..",
    "............",
  ],
  // Artifacts: a sealed crate with its lid on.
  artifacts: [
    "............",
    ".##########.",
    ".##########.",
    "............",
    ".++++++++++.",
    ".++++..++++.",
    ".++++..++++.",
    ".++++++++++.",
    ".++++++++++.",
    ".++++++++++.",
    "............",
    "............",
  ],
  // Connections: two services on one line.
  connections: [
    "............",
    ".####.......",
    ".####.......",
    ".####.......",
    ".####.......",
    "..##........",
    "..########..",
    "........##..",
    ".......++++.",
    ".......++++.",
    ".......++++.",
    ".......++++.",
  ],
  // Compute: an accelerator die with its pins.
  chip: [
    "............",
    "...#.##.#...",
    "............",
    ".#.######.#.",
    "...######...",
    ".#.##++##.#.",
    ".#.##++##.#.",
    "...######...",
    ".#.######.#.",
    "............",
    "...#.##.#...",
    "............",
  ],
  // Relay message: delivered forward, its earlier hops cooling behind it.
  message: [
    "............",
    "............",
    ".......#....",
    ".......##...",
    "-.+.#######.",
    "-.+.########",
    "-.+.#######.",
    ".......##...",
    ".......#....",
    "............",
    "............",
    "............",
  ],
  // Screen capture: a recording dot inside its frame.
  record: [
    "............",
    "............",
    ".++++++++++.",
    ".+........+.",
    ".+...##...+.",
    ".+..####..+.",
    ".+..####..+.",
    ".+...##...+.",
    ".+........+.",
    ".++++++++++.",
    "............",
    "............",
  ],
  // Sealed test: the lock that holds the final questions.
  lock: [
    "............",
    "....++++....",
    "...+....+...",
    "...+....+...",
    "...+....+...",
    ".##########.",
    ".##########.",
    ".####..####.",
    ".####..####.",
    ".##########.",
    ".##########.",
    "............",
  ],
  // Code: a terminal with a prompt.
  code: [
    "............",
    ".##########.",
    ".##########.",
    ".#........#.",
    ".#.#......#.",
    ".#..#.....#.",
    ".#.#..+++.#.",
    ".#........#.",
    ".##########.",
    "............",
    "............",
    "............",
  ],
  // A measurement over time.
  chart: [
    "............",
    "..........##",
    ".........###",
    "........##..",
    "...#...##...",
    "..###.##....",
    ".##.###.....",
    "##...#......",
    "............",
    "++++++++++++",
    "............",
    "............",
  ],
  book: [
    "............",
    ".####..####.",
    ".#####.####.",
    ".##+##.#+##.",
    ".#####.####.",
    ".##+##.#+##.",
    ".#####.####.",
    ".#####.####.",
    ".#####.####.",
    "..####.###..",
    ".....##.....",
    "............",
  ],
  clock: [
    "............",
    "....####....",
    "..########..",
    ".####.#####.",
    ".####.#####.",
    "#####.######",
    "#####....###",
    ".##########.",
    ".##########.",
    "..########..",
    "....####....",
    "............",
  ],
  // The research director's objective.
  target: [
    "............",
    "....++++....",
    "..++++++++..",
    ".+++....+++.",
    ".++..##..++.",
    "++..####..++",
    "++..####..++",
    ".++..##..++.",
    ".+++....+++.",
    "..++++++++..",
    "....++++....",
    "............",
  ],
  // Hephaestus: the anvil the lab is named for.
  anvil: [
    "............",
    "............",
    "##########..",
    ".###########",
    "..#########.",
    "....#####...",
    ".....###....",
    ".....###....",
    "...#######..",
    "..+++++++++.",
    "..+++++++++.",
    "............",
  ],
  // Read receipt: a second check landing on the first.
  receipt: [
    "............",
    "............",
    "........+..#",
    ".......++.##",
    ".+....++.##.",
    ".++..++.##..",
    "..++++.##...",
    "...++.##.#..",
    ".....##.##..",
    "......###...",
    ".......#....",
    "............",
  ],
  // Halt: a process stopped on purpose, cut through the middle.
  halt: [
    "............",
    "...######...",
    "..########..",
    ".##########.",
    ".##########.",
    ".#........#.",
    ".#........#.",
    ".##########.",
    ".##########.",
    "..########..",
    "...######...",
    "............",
  ],
  // Resume: the loop closes and work picks up again.
  resume: [
    "............",
    "....####.#..",
    "..###..####.",
    ".##.....###.",
    ".#.....####.",
    "##..........",
    "##..........",
    ".#.......##.",
    ".##.....##..",
    "..###..###..",
    "....####....",
    "............",
  ],
};

function cells(rows) {
  const lit = [];
  rows.forEach((row, y) =>
    [...row].forEach((c, x) => {
      if (c !== ".") lit.push({ x, y, cooler: c === "+" ? 1 : c === "-" ? 2 : 0 });
    }),
  );
  return lit;
}

function render(name, rows) {
  if (rows.length !== N || rows.some((r) => r.length !== N))
    throw new Error(`${name} must be ${N} x ${N}`);
  const lit = cells(rows);
  const body = lit.filter((c) => !c.cooler);
  const core = body.length ? body : lit;
  const cx = core.reduce((a, c) => a + c.x, 0) / core.length;
  const cy = core.reduce((a, c) => a + c.y, 0) / core.length;
  const reach = Math.max(...lit.map((c) => Math.hypot(c.x - cx, c.y - cy)), 1);
  const steps = [[], [], []];
  for (const c of lit) {
    const d = Math.hypot(c.x - cx, c.y - cy) / reach;
    const step = Math.min(2, (d < 0.45 ? 0 : d < 0.8 ? 1 : 2) + c.cooler);
    steps[step].push(`M${c.x} ${c.y}h${fmt(CELL)}v${fmt(CELL)}h-${fmt(CELL)}z`);
  }
  return steps
    .map((paths, i) => ({ opacity: HEAT[i], d: paths.join("") }))
    .filter((s) => s.d);
}

const fmt = (n) => Number(n.toFixed(4)).toString();

const root = join(import.meta.dirname, "..");
const data = Object.fromEntries(
  Object.entries(GLYPHS).map(([name, rows]) => [name, render(name, rows)]),
);

writeFileSync(
  join(root, "components/glyphs.ts"),
  "// Generated by scripts/gen-glyphs.mjs. Edit the drawings there, then rerun it.\n" +
    `export const GLYPH_GRID = ${N};\n` +
    "export const GLYPHS: Record<string, { opacity: number; d: string }[]> = " +
    JSON.stringify(data, null, 2) +
    ";\n",
);

// Standalone copies carry the dark-theme green so they read on the site's dark surfaces.
mkdirSync(join(root, "public/glyphs"), { recursive: true });
for (const [name, steps] of Object.entries(data)) {
  writeFileSync(
    join(root, "public/glyphs", `${name}.svg`),
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${N} ${N}" fill="#8bdbac" shape-rendering="crispEdges">` +
      steps.map((s) => `<path opacity="${s.opacity}" d="${s.d}"/>`).join("") +
      "</svg>\n",
  );
}
console.log(`Generated ${Object.keys(data).length} glyphs`);
