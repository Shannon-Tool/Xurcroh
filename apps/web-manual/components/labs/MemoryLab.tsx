"use client";

import { useMemo, useState } from "react";
import { simulateMemory } from "@/lib/simulation/memory";
import { LabShell } from "@/components/LabShell";
import { PlaybackControls } from "@/components/PlaybackControls";
import { usePlayback } from "@/components/usePlayback";
import { memoryScene } from "@/lib/scene/builders";

export function MemoryLab() {
  const [synchronized, setSynchronized] = useState(true);
  const playback = usePlayback(5);
  const steps = useMemo(() => simulateMemory(synchronized), [synchronized]);
  const scene = useMemo(() => memoryScene({ steps, index: playback.index }), [steps, playback.index]);
  const step = steps[playback.index]!;
  const residual = step.bytes.some((value) => value !== 0);
  return (
    <LabShell id="memory" scene={scene} title="GPU memory lifecycle" caption="Logical allocator cleanup is distinct from physical sanitization. Toggle synchronization to see how queued—not completed—wipes can expose sensitive residual bytes on reallocation.">
      <div className="lab-controls">
        <label><input type="checkbox" checked={synchronized} onChange={(event) => setSynchronized(event.target.checked)} /> Complete wipe before reuse</label>
        <PlaybackControls {...playback} canStepBackward={playback.index > 0} onToggle={() => playback.setPlaying(!playback.playing)} />
      </div>
      <div className="home-grid">
        {steps.map((candidate, index) => <div key={candidate.operation} className={index === playback.index ? "flow-node active" : "flow-node"}>
          <strong>{candidate.operation}</strong>
          <div aria-label={candidate.note} style={{ display: "flex", gap: 4, padding: 8 }}>
            {candidate.bytes.map((value, cell) => <span key={cell} className={value === 0 ? "memory-cell" : "memory-cell residual"} style={{ padding: 8 }}>{value.toString(16).padStart(2, "0")}</span>)}
          </div>
        </div>)}
      </div>
      <p className={residual ? "error" : "success"}>{step.note}</p>
      <p className="small">Actual allocators additionally require stream ordering and pool-specific wipes. A sanitizer kernel does not prove complete physical register/SRAM erasure.</p>
    </LabShell>
  );
}
