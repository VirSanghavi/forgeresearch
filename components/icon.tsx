import { GLYPHS, GLYPH_GRID } from "./glyphs";

/**
 * A Forge concept glyph (scripts/gen-glyphs.mjs), drawn on a 12-cell grid. Sizes
 * snap to multiples of 6px so every cell lands on whole device pixels. Below 24px
 * the cells are solid; from 24px up they open into the halftone matrix.
 */
export function Glyph({
  name,
  size = 18,
  className = "",
}: {
  name: string;
  size?: number;
  className?: string;
}) {
  const steps = GLYPHS[name];
  if (!steps) return null;
  const px = Math.max(12, Math.round(size / 6) * 6);
  const solid = px < 24;
  return (
    <svg
      width={px}
      height={px}
      viewBox={`0 0 ${GLYPH_GRID} ${GLYPH_GRID}`}
      fill="currentColor"
      shapeRendering="crispEdges"
      className={`glyph ${className}`}
      aria-hidden="true"
    >
      {steps.map((s) => (
        <path
          key={s.opacity}
          opacity={s.opacity}
          d={solid ? s.d.replaceAll("h0.6667v0.6667h-0.6667", "h1v1h-1") : s.d}
        />
      ))}
    </svg>
  );
}

/** Concepts render as glyphs; small utility marks (arrows, close, check) stay strokes. */
export function Icon({
  name,
  size = 18,
  className = "",
}: {
  name: string;
  size?: number;
  className?: string;
}) {
  if (GLYPHS[name])
    return <Glyph name={name} size={size} className={className} />;
  const paths: Record<string, React.ReactNode> = {
    plus: <path d="M12 5v14M5 12h14" />,
    arrow: <path d="M5 12h14m-5-5l5 5-5 5" />,
    chevron: <path d="M9 5l7 7-7 7" />,
    down: <path d="M6 9l6 6 6-6" />,
    check: <path d="M5 12l4 4L19 6" />,
    close: <path d="M6 6l12 12M6 18L18 6" />,
    download: (
      <>
        <path d="M12 3v12m-5-5l5 5 5-5M4 16v5h16v-5" />
      </>
    ),
    play: <path d="M8 4l12 8-12 8V4z" />,
    stop: <rect x="6" y="6" width="12" height="12" rx="1" />,
    moon: <path d="M20 14a8.5 8.5 0 01-10-10A8.5 8.5 0 1020 14z" />,
    sun: (
      <>
        <circle cx="12" cy="12" r="4" />
        <path d="M12 1v2m0 18v2M1 12h2m18 0h2M4 4l2 2m12 12l2 2M4 20l2-2M18 6l2-2" />
      </>
    ),
    menu: <path d="M4 6h16M4 12h16M4 18h16" />,
    search: (
      <>
        <circle cx="10" cy="10" r="6" />
        <path d="M15 15l6 6" />
      </>
    ),
    external: (
      <>
        <path d="M14 3h7v7m0-7L10 14M9 3H3v18h18v-6" />
      </>
    ),
    copy: (
      <>
        <rect x="8" y="8" width="13" height="13" rx="2" />
        <path d="M16 8V3H3v13h5" />
      </>
    ),
    alert: (
      <>
        <path d="M12 3L2 21h20L12 3zM12 9v5m0 3v1" />
      </>
    ),
    info: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M12 11v6m0-10v1" />
      </>
    ),
    refresh: (
      <>
        <path d="M20 8a8 8 0 10.4 7M20 3v6h-6" />
      </>
    ),
  };
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      {paths[name] ?? paths.info}
    </svg>
  );
}
export function ForgeMark({ size = 30 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 36 36"
      fill="none"
      aria-hidden="true"
    >
      <path d="M4 5h28L22 15H4zm0 15h18L12 30H4z" fill="currentColor" />
    </svg>
  );
}
