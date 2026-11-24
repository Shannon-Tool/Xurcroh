"use client";

import { useEffect, useId, useRef } from "react";
import mermaid from "mermaid";

export function Mermaid({ chart, caption }: { chart: string; caption: string }) {
  const ref = useRef<HTMLDivElement>(null);
  // Not crypto.randomUUID: it exists only on HTTPS and localhost pages.
  const id = `mermaid-${useId().replace(/[^A-Za-z0-9_-]/g, "")}`;

  useEffect(() => {
    let cancelled = false;
    mermaid.initialize({ startOnLoad: false, securityLevel: "strict", theme: "base" });
    // Chapter source writes line breaks as the two characters "\n".
    void mermaid.render(id, chart.replaceAll("\\n", "\n")).then(({ svg }) => {
      if (!cancelled && ref.current) ref.current.innerHTML = svg;
    }).catch((error: unknown) => {
      if (!cancelled && ref.current) ref.current.textContent = `Mermaid error: ${String(error)}`;
    });
    return () => { cancelled = true; };
  }, [chart, id]);

  return <figure className="figure"><div ref={ref} className="mermaid" /><figcaption className="figure-caption">{caption}</figcaption></figure>;
}
