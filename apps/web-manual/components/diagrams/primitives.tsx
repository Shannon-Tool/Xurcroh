import { useId } from "react";

export type Tone = "plain" | "accent" | "warn" | "good";

/** Frame shared by every chapter diagram: scalable, themed through CSS tokens, described for screen readers. */
export function Diagram({ title, description, width, height, caption, children }: {
  title: string;
  description: string;
  width: number;
  height: number;
  caption: string;
  children: (arrowId: string) => React.ReactNode;
}) {
  const id = useId().replace(/[^A-Za-z0-9_-]/g, "");
  const arrowId = `dg-arrow-${id}`;
  return (
    <figure className="figure">
      <svg className="dg" viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby={`dg-t-${id}`} aria-describedby={`dg-d-${id}`}>
        <title id={`dg-t-${id}`}>{title}</title>
        <desc id={`dg-d-${id}`}>{description}</desc>
        <defs>
          <marker id={arrowId} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0 0 L10 5 L0 10 z" className="dg-arrowhead" />
          </marker>
        </defs>
        {children(arrowId)}
      </svg>
      <figcaption className="figure-caption">{caption}</figcaption>
    </figure>
  );
}

export function Zone({ x, y, width, height, label, tone = "plain" }: { x: number; y: number; width: number; height: number; label: string; tone?: Tone }) {
  return (
    <g>
      <rect x={x} y={y} width={width} height={height} rx="12" className={`dg-zone dg-${tone}`} />
      <text x={x + 14} y={y + 24} className="dg-zone-label">{label}</text>
    </g>
  );
}

export function Box({ x, y, width, height, lines, tone = "plain" }: { x: number; y: number; width: number; height: number; lines: string[]; tone?: Tone }) {
  const lineHeight = 17;
  const first = y + height / 2 - ((lines.length - 1) * lineHeight) / 2 + 5;
  return (
    <g>
      <rect x={x} y={y} width={width} height={height} rx="8" className={`dg-box dg-${tone}`} />
      {lines.map((line, index) => (
        <text key={line} x={x + width / 2} y={first + index * lineHeight} textAnchor="middle" className={index === 0 ? "dg-text dg-strong" : "dg-text dg-sub"}>{line}</text>
      ))}
    </g>
  );
}

export function Arrow({ d, arrowId, label, labelX, labelY, tone = "plain", dashed = false, both = false }: {
  d: string;
  arrowId: string;
  label?: string;
  labelX?: number;
  labelY?: number;
  tone?: Tone;
  dashed?: boolean;
  both?: boolean;
}) {
  return (
    <g>
      <path d={d} className={`dg-line dg-${tone}`} strokeDasharray={dashed ? "6 5" : undefined} markerEnd={`url(#${arrowId})`} markerStart={both ? `url(#${arrowId})` : undefined} />
      {label && labelX !== undefined && labelY !== undefined ? <text x={labelX} y={labelY} textAnchor="middle" className="dg-label">{label}</text> : null}
    </g>
  );
}

export function Note({ x, y, lines }: { x: number; y: number; lines: string[] }) {
  return <>{lines.map((line, index) => <text key={line} x={x} y={y + index * 16} className="dg-note">{line}</text>)}</>;
}
