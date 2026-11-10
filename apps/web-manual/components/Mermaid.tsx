"use client";

import { useEffect, useRef } from "react";
import mermaid from "mermaid";

export function Mermaid({ chart, caption }: { chart: string; caption: string }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    mermaid.initialize({ startOnLoad: false, securityLevel: "strict", theme: "base" });
    void mermaid.render(`mermaid-${crypto.randomUUID()}`, chart).then(({ svg }) => {
      if (!cancelled && ref.current) ref.current.innerHTML = svg;
    }).catch((error: unknown) => {
      if (!cancelled && ref.current) ref.current.textContent = `Mermaid error: ${String(error)}`;
    });
    return () => { cancelled = true; };
  }, [chart]);

  return <figure className="figure"><div ref={ref} className="mermaid" /><figcaption className="figure-caption">{caption}</figcaption></figure>;
}
