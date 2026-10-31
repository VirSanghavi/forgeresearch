"use client";
/* Forge's hero art: one three.js (r180) points cloud that morphs through the five steps of
   campaign forge-e84925eb every 5.05 s. All geometry is Forge's own and built from the run's data:
   flagged answers, five agents, the real exp-001 loss curve, and before/after bars. */
import { useEffect, useRef, useState } from "react";
import {
  BufferAttribute,
  BufferGeometry,
  PerspectiveCamera,
  Points,
  Scene,
  ShaderMaterial,
  Vector2,
  Vector3,
  WebGLRenderer,
} from "three";
import { LOSS } from "./data";
import { useSiteTheme } from "./theme";
import s from "./live.module.css";

const N = 4096;
const HOLD = 5050; // ms per state, measured
const MORPH = 600; // ms, out-quart (the reference morph lands inside one 500 ms sample)
const W = 9; // world width the camera always fits

export const STATES = [
  { label: "Traffic", caption: "Your agent's real conversations come in" },
  {
    label: "Find mistakes",
    caption: "Forge spots the answers that need fixing",
  },
  { label: "Agents", caption: "Five agents plan the fix together" },
  { label: "Train", caption: "The model learns from its own mistakes" },
  {
    label: "Result",
    caption: "Before and after: more questions answered right",
  },
];

// Deterministic noise so the art is identical on every load.
function rng(seed: number) {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

// Exponential moving average seeded with the mean of the first 10 steps, as loss plots do.
const SEED = LOSS.slice(0, 10).reduce((p, v) => p + v, 0) / 10;
const SMOOTH: number[] = LOSS.reduce<number[]>((acc, v, i) => {
  const prev = i ? acc[i - 1] : SEED;
  acc.push(prev + 0.035 * (v - prev));
  return acc;
}, []);
// Light centered smoothing of the averaged curve so the ribbon reads as one line.
for (let pass = 0; pass < 3; pass++)
  for (let i = 1; i < SMOOTH.length - 1; i++)
    SMOOTH[i] = (SMOOTH[i - 1] + 2 * SMOOTH[i] + SMOOTH[i + 1]) / 4;
const LO = Math.min(...SMOOTH);
const HI = Math.max(...SMOOTH);

function buildStates() {
  const r = rng(7);
  const g = () => (r() + r() + r() - 1.5) / 1.5; // ~gaussian in [-1, 1]
  const states = Array.from({ length: 5 }, () => new Float32Array(N * 3));
  const flag = new Float32Array(N); // 1 = belongs to a flagged answer
  const ghost = new Float32Array(N); // 1 = faint outline dot in the result bars
  const set = (k: number, i: number, x: number, y: number, z: number) => {
    states[k][i * 3] = x;
    states[k][i * 3 + 1] = y;
    states[k][i * 3 + 2] = z;
  };

  for (let i = 0; i < N; i++) {
    // 0. Traffic: six flowing streams of conversations.
    const lane = i % 6;
    const u = r();
    const x0 = -W / 2 + u * W;
    set(
      0,
      i,
      x0,
      Math.sin(x0 * 0.7 + lane * 1.1) * 0.55 + (lane - 2.5) * 0.36 + g() * 0.12,
      g() * 0.9,
    );

    // 1. Find mistakes: 64 answers as 16 x 4 tiles of 8 x 8 dots; 62 tiles lift and darken.
    const tile = Math.floor(i / 64);
    const d = i % 64;
    const tx = tile % 16;
    const ty = Math.floor(tile / 16);
    const flagged = tile !== 21 && tile !== 46 ? 1 : 0;
    flag[i] = flagged;
    set(
      1,
      i,
      -4.1 + tx * 0.547 + (d % 8) * 0.055,
      -1.0 + ty * 0.62 + Math.floor(d / 8) * 0.055,
      flagged ? 0.35 : -0.15,
    );

    // 2. The team: five clusters with dotted links (ring plus spokes to the director).
    const centers = [
      [0, 0.05],
      [-2.9, 1.1],
      [2.9, 1.1],
      [-1.9, -1.35],
      [1.9, -1.35],
    ];
    if (i < 3300) {
      const c = centers[i % 5];
      const rad = 0.42 * Math.cbrt(r());
      const th = r() * Math.PI * 2;
      const ph = Math.acos(2 * r() - 1);
      set(
        2,
        i,
        c[0] + rad * Math.sin(ph) * Math.cos(th),
        c[1] + rad * Math.cos(ph),
        rad * Math.sin(ph) * Math.sin(th),
      );
    } else {
      const links = [
        [1, 2],
        [2, 4],
        [4, 3],
        [3, 1],
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
      ];
      const l = links[(i - 3300) % links.length];
      const t = (Math.floor((i - 3300) / links.length) + 0.5) / 100;
      const a = centers[l[0]];
      const b = centers[l[1]];
      set(2, i, a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, 0);
    }

    // 3. Train: the exp-001 loss curve as a ribbon, smoothed the way loss plots usually
    // are (exponential moving average, alpha 0.06) and sampled at 256 points along x.
    const col = i % 256;
    const row = Math.floor(i / 256);
    const pos = (col / 255) * 95;
    const lo = Math.floor(pos);
    const hi = Math.min(95, lo + 1);
    const y = SMOOTH[lo] + (SMOOTH[hi] - SMOOTH[lo]) * (pos - lo);
    set(
      3,
      i,
      -4.2 + (col / 255) * 8.4,
      -1.4 + ((y - LO) / (HI - LO)) * 2.8 + (row / 15 - 0.5) * 0.14,
      g() * 0.08,
    );

    // 4. Result: before and after, as two bars in the same frame. Heights follow the
    // final test of the recorded run (right answers before vs after), shown unlabeled.
    const bar = i % 2;
    const bx = bar ? 1.3 : -1.3;
    const slots = 24;
    const slotH = 3.6 / slots;
    const base = -1.8;
    const fillTop = (bar ? 6 : 1) * slotH;
    const isGhost = r() < 0.32;
    ghost[i] = isGhost ? 1 : 0;
    const hy = isGhost ? r() * slots * slotH : r() * fillTop;
    set(4, i, bx + (r() - 0.5) * 1.1, base + hy, (r() - 0.5) * 1.1);
  }
  return { states, flag, ghost };
}

const VERT = /* glsl */ `
  attribute vec3 p0; attribute vec3 p1; attribute vec3 p2; attribute vec3 p3; attribute vec3 p4;
  attribute float aSeed; attribute float aFlag; attribute float aGhost;
  uniform float uFrom; uniform float uTo; uniform float uMix; uniform float uTime;
  uniform vec2 uMouse; uniform float uDpr; uniform float uRise;
  varying float vAlpha;
  vec3 pick(float k) {
    if (k < 0.5) return p0 + vec3(sin(uTime * 0.0004 + aSeed * 6.28) * 0.08, 0.0, 0.0);
    if (k < 1.5) return p1;
    if (k < 2.5) return p2;
    if (k < 3.5) return p3;
    vec3 q = p4;
    if (aGhost < 0.5) q.y = -1.8 + (q.y + 1.8) * uRise;
    return q;
  }
  float shade(float k) {
    if (k > 0.5 && k < 1.5) return aFlag > 0.5 ? 0.9 : 0.25;
    if (k > 3.5) return aGhost > 0.5 ? 0.16 : 0.92;
    return 0.35 + aSeed * 0.5;
  }
  void main() {
    vec3 pos = mix(pick(uFrom), pick(uTo), uMix);
    // Gentle repel from the pointer, in world units.
    vec2 dv = pos.xy - uMouse;
    float dist = length(dv);
    pos.xy += normalize(dv + 0.0001) * 0.22 * smoothstep(0.9, 0.0, dist);
    vec4 mv = modelViewMatrix * vec4(pos, 1.0);
    gl_Position = projectionMatrix * mv;
    float depth = clamp((-mv.z - 9.0) / 6.0, 0.0, 1.0);
    vAlpha = mix(shade(uFrom), shade(uTo), uMix) * (1.0 - depth * 0.55);
    vAlpha += 0.25 * smoothstep(0.9, 0.0, dist);
    gl_PointSize = (3.4 + aSeed * 2.6) * uDpr * (12.0 / -mv.z);
  }
`;
const FRAG = /* glsl */ `
  uniform vec3 uColor;
  varying float vAlpha;
  void main() {
    vec2 c = gl_PointCoord - 0.5;
    float d = length(c);
    if (d > 0.5) discard;
    float core = smoothstep(0.32, 0.2, d);
    float halo = 0.22 * exp(-d * d * 18.0);
    gl_FragColor = vec4(uColor, vAlpha * clamp(core + halo, 0.0, 1.0));
  }
`;

const easeOutQuart = (t: number) => 1 - Math.pow(1 - t, 4);

export function ParticleHero() {
  const wrap = useRef<HTMLDivElement>(null);
  const [state, setState] = useState(0);
  const theme = useSiteTheme();
  const color = useRef<Vector3 | null>(null);
  // #111111 on light, #ededed on dark; the shader reads it every frame.
  useEffect(() => {
    color.current?.setScalar(theme === "dark" ? 0.929 : 0.067);
  }, [theme]);

  useEffect(() => {
    const host = wrap.current;
    if (!host) return;
    const reduced = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    let renderer: WebGLRenderer;
    try {
      renderer = new WebGLRenderer({
        antialias: false,
        alpha: true,
        powerPreference: "low-power",
      });
    } catch {
      return; // No WebGL: the caption and tabs still tell the story.
    }
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    renderer.setPixelRatio(dpr);
    renderer.setClearColor(0xffffff, 0);
    renderer.domElement.setAttribute("aria-hidden", "true");
    host.appendChild(renderer.domElement);

    const scene = new Scene();
    const camera = new PerspectiveCamera(35, 1, 0.1, 100);
    const { states, flag, ghost } = buildStates();
    const geo = new BufferGeometry();
    geo.setAttribute("position", new BufferAttribute(states[0], 3));
    states.forEach((p, k) =>
      geo.setAttribute(`p${k}`, new BufferAttribute(p, 3)),
    );
    const seed = new Float32Array(N);
    const r = rng(11);
    for (let i = 0; i < N; i++) seed[i] = r();
    geo.setAttribute("aSeed", new BufferAttribute(seed, 1));
    geo.setAttribute("aFlag", new BufferAttribute(flag, 1));
    geo.setAttribute("aGhost", new BufferAttribute(ghost, 1));
    const uniforms = {
      uFrom: { value: reduced ? 4 : 0 },
      uTo: { value: reduced ? 4 : 0 },
      uMix: { value: 0 },
      uTime: { value: 0 },
      uMouse: { value: new Vector2(99, 99) },
      uDpr: { value: dpr },
      uRise: { value: 1 },
      uColor: {
        value: new Vector3().setScalar(
          document.documentElement.dataset.siteTheme === "dark" ? 0.929 : 0.067,
        ),
      },
    };
    color.current = uniforms.uColor.value;
    const mat = new ShaderMaterial({
      vertexShader: VERT,
      fragmentShader: FRAG,
      uniforms,
      transparent: true,
      depthWrite: false,
    });
    const points = new Points(geo, mat);
    points.frustumCulled = false;
    scene.add(points);

    const fit = () => {
      const w = host.clientWidth;
      const h = host.clientHeight;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      // Fit W world units across, or 5.6 units down, whichever is tighter.
      const vFov = (camera.fov * Math.PI) / 180;
      const distW = W / 2 / (Math.tan(vFov / 2) * camera.aspect);
      const distH = 5.6 / 2 / Math.tan(vFov / 2);
      camera.position.set(0, 0, Math.max(distW, distH));
      camera.updateProjectionMatrix();
    };
    fit();
    const ro = new ResizeObserver(fit);
    ro.observe(host);

    const pointer = { x: 0, y: 0, active: false };
    const onMove = (e: PointerEvent) => {
      const b = host.getBoundingClientRect();
      const nx = ((e.clientX - b.left) / b.width) * 2 - 1;
      const ny = -(((e.clientY - b.top) / b.height) * 2 - 1);
      pointer.x = nx;
      pointer.y = ny;
      pointer.active = Math.abs(nx) <= 1.1 && Math.abs(ny) <= 1.1;
      const halfH = Math.tan((camera.fov * Math.PI) / 360) * camera.position.z;
      uniforms.uMouse.value.set(
        pointer.active ? nx * halfH * camera.aspect : 99,
        pointer.active ? ny * halfH : 99,
      );
    };
    window.addEventListener("pointermove", onMove, { passive: true });

    if (reduced) {
      renderer.render(scene, camera);
      const id = requestAnimationFrame(() => setState(4));
      return () => {
        cancelAnimationFrame(id);
        ro.disconnect();
        window.removeEventListener("pointermove", onMove);
        geo.dispose();
        mat.dispose();
        renderer.dispose();
        renderer.domElement.remove();
      };
    }

    let visible = false;
    let raf = 0;
    let last = 0;
    let elapsed = 0;
    let shown = -1;
    const total = HOLD * STATES.length;
    const frame = (now: number) => {
      raf = 0;
      if (!visible || document.hidden) return;
      elapsed = (elapsed + (last ? Math.min(now - last, 64) : 0)) % total;
      last = now;
      const k = Math.floor(elapsed / HOLD);
      const into = elapsed - k * HOLD;
      uniforms.uFrom.value = (k + STATES.length - 1) % STATES.length;
      uniforms.uTo.value = k;
      uniforms.uMix.value = easeOutQuart(Math.min(1, into / MORPH));
      uniforms.uRise.value =
        k === 4
          ? easeOutQuart(Math.min(1, Math.max(0, (into - 300) / 1400)))
          : 1;
      uniforms.uTime.value = now;
      // Slow idle turn plus a little pointer parallax.
      points.rotation.y =
        Math.sin(now * 0.00018) * 0.22 +
        (pointer.active ? pointer.x * 0.08 : 0);
      points.rotation.x =
        0.12 +
        Math.sin(now * 0.00013) * 0.05 -
        (pointer.active ? pointer.y * 0.05 : 0);
      renderer.render(scene, camera);
      if (k !== shown) {
        shown = k;
        setState(k);
      }
      raf = requestAnimationFrame(frame);
    };
    const wake = () => {
      if (!raf && visible && !document.hidden) {
        last = 0;
        raf = requestAnimationFrame(frame);
      }
    };
    const io = new IntersectionObserver(([e]) => {
      visible = e.isIntersecting;
      if (visible) wake();
    });
    io.observe(host);
    const onVis = () => wake();
    document.addEventListener("visibilitychange", onVis);

    return () => {
      cancelAnimationFrame(raf);
      io.disconnect();
      ro.disconnect();
      document.removeEventListener("visibilitychange", onVis);
      window.removeEventListener("pointermove", onMove);
      geo.dispose();
      mat.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);

  return (
    <div className={s.art}>
      <div className={s.artTabs} role="list" aria-label="Campaign steps">
        {STATES.map((st, i) => (
          <span
            role="listitem"
            key={st.label}
            className={`${s.stepTab} ${i === state ? s.stepTabOn : ""}`}
            aria-current={i === state ? "step" : undefined}
          >
            {st.label}
          </span>
        ))}
      </div>
      <div
        ref={wrap}
        className={s.artCanvas}
        role="img"
        aria-label="Animated dots show how Forge works: your agent's conversations come in, the answers that need fixing are found, five agents plan the fix, the model trains, and more questions are answered right afterward."
      />
      <p className={s.artCaption} aria-live="polite">
        {STATES[state].caption}
      </p>
    </div>
  );
}
