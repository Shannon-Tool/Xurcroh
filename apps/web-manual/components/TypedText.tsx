"use client";

import { useEffect, useState } from "react";

const WORD_MS = 85;
const HOLD_MS = 2600;

/**
 * Text that arrives word by word and, after a pause, arrives again: an answer that keeps coming.
 * Readers who prefer reduced motion, and assistive technology, get the whole text at once.
 */
export function TypedText({ text, loop = true }: { text: string; loop?: boolean }) {
  const words = text.split(" ");
  const [count, setCount] = useState(words.length);
  const [moving, setMoving] = useState(false);

  useEffect(() => {
    const allowed = typeof window.matchMedia === "function" && !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    setMoving(allowed);
    if (!allowed) {
      setCount(text.split(" ").length);
      return;
    }
    const total = text.split(" ").length;
    let shown = 0;
    let timer = 0;
    setCount(0);
    const step = () => {
      shown += 1;
      setCount(shown);
      if (shown < total) timer = window.setTimeout(step, WORD_MS);
      else if (loop) timer = window.setTimeout(() => { shown = 0; setCount(0); timer = window.setTimeout(step, WORD_MS); }, HOLD_MS);
    };
    timer = window.setTimeout(step, WORD_MS);
    return () => window.clearTimeout(timer);
  }, [text, loop]);

  if (!moving) return <>{text}</>;
  return <><span className="sr-only">{text}</span><span aria-hidden="true">{words.slice(0, count).join(" ")}<span className="show-caret">▌</span></span></>;
}
