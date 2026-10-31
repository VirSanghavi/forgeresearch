// Original line drawings for the three "How it works" columns. Hairline strokes, black nodes.
const stroke = { stroke: "currentColor", strokeOpacity: 0.28, fill: "none" };

export function TracesDrawing() {
  const rows = [0, 1, 2, 3, 4, 5];
  return (
    <svg width="260" height="240" viewBox="0 0 260 240" aria-hidden="true">
      {rows.map((r) => (
        <g key={r} transform={`translate(20 ${20 + r * 36})`}>
          <rect width="220" height="24" rx="4" {...stroke} />
          <path
            d={`M14 12h${r % 2 ? 120 : 150}`}
            {...stroke}
            strokeDasharray="2 4"
          />
          {r === 2 || r === 4 ? (
            <rect
              x="196"
              y="6"
              width="12"
              height="12"
              rx="2"
              fill="currentColor"
            />
          ) : (
            <rect x="196" y="6" width="12" height="12" rx="2" {...stroke} />
          )}
        </g>
      ))}
    </svg>
  );
}

export function TeamDrawing() {
  const nodes = [
    [130, 40],
    [220, 104],
    [186, 206],
    [74, 206],
    [40, 104],
  ];
  return (
    <svg width="260" height="240" viewBox="0 0 260 240" aria-hidden="true">
      <circle cx="130" cy="128" r="96" {...stroke} strokeDasharray="2 4" />
      {nodes.map(([x, y], i) => {
        const [nx, ny] = nodes[(i + 1) % nodes.length];
        return <path key={`e${i}`} d={`M${x} ${y}L${nx} ${ny}`} {...stroke} />;
      })}
      {nodes.map(([x, y], i) => (
        <path
          key={`c${i}`}
          d={`M130 128L${x} ${y}`}
          {...stroke}
          strokeDasharray="2 4"
        />
      ))}
      <rect x="122" y="120" width="16" height="16" rx="3" {...stroke} />
      {nodes.map(([x, y], i) => (
        <rect
          key={i}
          x={x - 5}
          y={y - 5}
          width="10"
          height="10"
          rx="2"
          fill="currentColor"
        />
      ))}
    </svg>
  );
}

export function JournalDrawing() {
  return (
    <svg width="260" height="240" viewBox="0 0 260 240" aria-hidden="true">
      {[0, 1, 2].map((i) => (
        <rect
          key={i}
          x={50 + i * 14}
          y={40 + i * 14}
          width="132"
          height="150"
          rx="6"
          {...stroke}
          strokeDasharray={i < 2 ? "2 4" : undefined}
        />
      ))}
      {[0, 1, 2, 3, 4].map((r) => (
        <g key={r} transform="translate(94 0)">
          <path d={`M0 ${100 + r * 22}h66`} {...stroke} />
          <rect
            x="-16"
            y={95 + r * 22}
            width="10"
            height="10"
            rx="2"
            fill="currentColor"
          />
        </g>
      ))}
    </svg>
  );
}
