"use client";

import { useMemo, useState } from "react";
import { simulateMemory } from "@/lib/simulation/memory";
import { LabShell } from "@/components/LabShell";
import { PlaybackControls } from "@/components/PlaybackControls";
import { usePlayback } from "@/components/usePlayback";
import { memoryScene } from "@/lib/scene/builders";
import type { LiveStory } from "@/components/LiveIO";

const hex = (bytes: number[]) => bytes.map((value) => value.toString(16).padStart(2, "0")).join(" ");
import { useJourney } from "@/lib/show/question";

export function MemoryLab() {
  const [synchronized, setSynchronized] = useState(true);
  const playback = usePlayback(5);
  // While the model works, the question sits in memory. Its first four characters stand for it here.
  const journey = useJourney();
  const steps = useMemo(() => simulateMemory(synchronized, journey.bytes), [synchronized, journey]);
  const scene = useMemo(() => memoryScene({ steps, index: playback.index }), [steps, playback.index]);
  const step = steps[playback.index]!;
  const residual = step.bytes.some((value) => value !== 0);
  const left = steps[steps.length - 1]!.bytes;
  const stale = left.some((value) => value !== 0);
  const story: LiveStory = {
    plain: "While the model works on your question, the question sits in the graphics card's memory. When the answer is done, that memory is wiped before anyone else can get it. Skip the wipe, and the next user can read what was left behind.",
    stage: "wipe",
    input: <p>Its first four characters, “{journey.question.slice(0, 4)}”, are stored as the bytes <code>{hex(journey.bytes)}</code>.</p>,
    seenTitle: "What the next user of that memory reads",
    seen: <p><code>{hex(left)}</code> {stale ? "— the start of your question, still there." : "— nothing of yours."}</p>,
    output: stale ? <p>You got your answer, but part of your question leaked to the next request.</p> : <p>Afterwards the memory reads zeros. The next request cannot recover anything.</p>,
    outputTone: stale ? "warn" : "good",
    why: "Memory is handed to the next request only after the wipe has actually finished, so there is nothing left to read.",
    relies: "The wipe completes before reuse, on every pool that held your data.",
    limit: "A software wipe does not prove that every physical register and cache inside the chip is cleared.",
    attack: {
      label: "skip the wipe",
      on: !synchronized,
      onToggle: () => setSynchronized(!synchronized),
      outcome: stale ? "It worked: the next request reads your bytes. This is the failure the wipe exists to prevent." : "With the wipe in place, the next request reads only zeros.",
    },
    screenSeen: hex(step.bytes),
  };
  return (
    <LabShell id="memory" plainTitle="Wiping your question from memory afterwards" scene={scene} story={story} watch={playback.watch} title="GPU memory lifecycle" caption="Logical allocator cleanup is distinct from physical sanitization. Toggle synchronization to see how queued—not completed—wipes can expose sensitive residual bytes on reallocation.">
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
