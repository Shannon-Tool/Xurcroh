"use client";

import { useEffect, useMemo, useState } from "react";
import { Scene3DView } from "@/components/Scene3DView";
import { ARCHITECTURE_PHASES, architectureScene } from "@/lib/scene/builders";

const PHASE_MS = 3600;

/**
 * The design as one 3D scene that steps through split, send, compute, return, verify and combine.
 * An illustration built from the same scene model as the laboratories; it measures nothing.
 */
export function ArchitectureExplorer({ autoPlay = false, variant = "lab" }: { autoPlay?: boolean; variant?: "lab" | "hero" }) {
  const [phase, setPhase] = useState(0);
  const [playing, setPlaying] = useState(false);

  // Start by itself only where asked, and never for readers who prefer reduced motion.
  useEffect(() => {
    const reduced = typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (autoPlay && !reduced) setPlaying(true);
  }, [autoPlay]);

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => setPhase((current) => (current + 1) % ARCHITECTURE_PHASES.length), PHASE_MS);
    return () => window.clearInterval(timer);
  }, [playing]);

  const scene = useMemo(() => architectureScene({ phase }), [phase]);
  const go = (next: number) => {
    setPlaying(false);
    setPhase((next + ARCHITECTURE_PHASES.length) % ARCHITECTURE_PHASES.length);
  };

  return (
    <div className="explorer">
      <div className="explorer-steps" role="group" aria-label="Steps of one protected computation">
        <button onClick={() => setPlaying(!playing)}>{playing ? "Pause" : "Play"}</button>
        {ARCHITECTURE_PHASES.map((entry, index) => (
          <button key={entry.title} className={index === phase ? "explorer-step current" : "explorer-step"} aria-pressed={index === phase} onClick={() => go(index)}>
            <span aria-hidden="true">{index + 1}</span> {entry.title}
          </button>
        ))}
      </div>
      <Scene3DView scene={scene} variant={variant} />
    </div>
  );
}
