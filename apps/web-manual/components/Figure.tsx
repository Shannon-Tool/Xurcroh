"use client";

import { useState } from "react";

export function Figure({ caption, children }: { caption: string; children?: React.ReactNode }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <figure className="figure">
      <div className="diagram" role="img" aria-label={caption}>
        {children ?? <svg viewBox="0 0 640 120" aria-hidden="true">
          <path d="M20 60 H620" stroke="currentColor" strokeWidth="2" strokeDasharray="6 4" />
          {[70, 220, 370, 520].map((x) => <rect key={x} x={x - 40} y={35} width="80" height="50" rx="5" className="flow-node" />)}
          <path d="M110 60 H180 M260 60 H330 M410 60 H480" stroke="currentColor" strokeWidth="3" />
        </svg>}
      </div>
      <figcaption className="figure-caption">{caption}</figcaption>
      {children ? <button onClick={() => setExpanded(!expanded)}>{expanded ? "Hide details" : "Inspect diagram"}</button> : null}
      {expanded ? <pre className="state-inspector">{caption}</pre> : null}
    </figure>
  );
}
