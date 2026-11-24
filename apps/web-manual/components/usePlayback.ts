"use client";

import { useEffect, useState } from "react";

export function usePlayback(maxIndex: number) {
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);

  useEffect(() => {
    if (!playing || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const timer = window.setInterval(() => {
      setIndex((current) => (current >= maxIndex ? 0 : current + 1));
    }, 1000 / speed);
    return () => window.clearInterval(timer);
  }, [playing, speed, maxIndex]);

  function step(delta: number) {
    setPlaying(false);
    setIndex((current) => Math.max(0, Math.min(maxIndex, current + delta)));
  }

  function reset() {
    setPlaying(false);
    setIndex(0);
  }

  /** For a lab's Watch button: plays from the start when the lab is already at its end. */
  const watch = {
    playing,
    onToggle: () => {
      if (!playing && index >= maxIndex) setIndex(0);
      setPlaying(!playing);
    },
  };

  return { index: Math.max(0, Math.min(maxIndex, index)), playing, setPlaying, speed, onSpeed: setSpeed, onStep: step, onReset: reset, watch };
}
