"use client";

import { useState } from "react";

export function Figure({ caption, children }: { caption: string; children?: React.ReactNode }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <figure className="figure">
      {children ? <div className="diagram" role="img" aria-label={caption}>{children}</div> : null}
      <figcaption className="figure-caption">{caption}</figcaption>
      {children ? <button onClick={() => setExpanded(!expanded)}>{expanded ? "Hide details" : "Inspect diagram"}</button> : null}
      {expanded ? <pre className="state-inspector">{caption}</pre> : null}
    </figure>
  );
}
