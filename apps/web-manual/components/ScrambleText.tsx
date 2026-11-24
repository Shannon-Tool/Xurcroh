"use client";

import { useEffect, useRef, useState } from "react";
import { scrambleFrame } from "@/lib/show/scramble";

const DURATION_MS = 520;

/**
 * Text that visibly turns into its next value: used wherever readable data becomes pieces or sealed
 * bytes, and where results become readable again. Under reduced motion it simply shows the new value.
 */
export function ScrambleText({ text, className }: { text: string; className?: string }) {
  const [shown, setShown] = useState(text);
  const previous = useRef(text);

  useEffect(() => {
    const from = previous.current;
    previous.current = text;
    const reduced = typeof window.matchMedia !== "function" || window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduced || from === text) {
      setShown(text);
      return;
    }
    const started = performance.now();
    let frame = 0;
    let tick = 0;
    const step = (now: number) => {
      const progress = (now - started) / DURATION_MS;
      tick += 1;
      setShown(scrambleFrame(from, text, progress, tick));
      if (progress < 1) frame = window.requestAnimationFrame(step);
    };
    frame = window.requestAnimationFrame(step);
    return () => window.cancelAnimationFrame(frame);
  }, [text]);

  // Assistive technology gets the real value, not the frames in between.
  return <span className={className}><span className="sr-only">{text}</span><span aria-hidden="true">{shown}</span></span>;
}
