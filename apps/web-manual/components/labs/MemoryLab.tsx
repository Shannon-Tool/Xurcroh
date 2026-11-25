"use client";

import { useMemo, useState } from "react";
import { simulateMemory } from "@/lib/simulation/memory";
import { LabShell } from "@/components/LabShell";
import { PlaybackControls } from "@/components/PlaybackControls";
import { usePlayback } from "@/components/usePlayback";
import { memoryScene } from "@/lib/scene/builders";
import type { LiveStory } from "@/components/LiveIO";

const hex = (bytes: number[]) => bytes.map((value) => value.toString(16).padStart(2, "0")).join(" ");
import { writingTime } from "@/lib/show/generation";
import { stepped, type ProcessStep } from "@/lib/show/process";
import { useJourney } from "@/lib/show/question";

export function MemoryLab() {
  const [synchronized, setSynchronized] = useState(true);
  // While the model works, the question sits in memory. Its first four characters stand for it here.
  const journey = useJourney();
  const playback = usePlayback(5, writingTime(journey.tokens.length));
  const steps = useMemo(() => simulateMemory(synchronized, journey.bytes), [synchronized, journey]);
  const scene = useMemo(() => memoryScene({ steps, index: playback.index }), [steps, playback.index]);
  const step = steps[playback.index]!;
  const residual = step.bytes.some((value) => value !== 0);
  const left = steps[steps.length - 1]!.bytes;
  const stale = left.some((value) => value !== 0);
  const last = steps.length - 1;
  // The engine's six operations in everyday words.
  const PLAIN: Record<string, { title: string; does: string }> = {
    allocate: { title: "Set aside", does: "Memory is set aside for your request, and the start of your question is written into it." },
    execute: { title: "In use", does: "The model works on your question. The bytes stay in memory while it does." },
    release: { title: "Handed back", does: "The work is done and the memory is handed back to the pool. Handing it back does not erase it." },
    synchronize: synchronized ? { title: "Wipe put first", does: "The wipe is put ahead of any reuse: nobody gets this memory until the wipe has finished." } : { title: "Wipe only queued", does: "The wipe is only queued. Nothing waits for it to finish." },
    zero: synchronized ? { title: "Wiped", does: "The wipe has run, and the memory was read back to confirm it." } : { title: "Not wiped yet", does: "The wipe has not run yet, so your bytes are still there." },
    reuse: synchronized ? { title: "Next user", does: "The memory goes to the next request, which finds nothing of yours." } : { title: "Next user", does: "The memory goes to the next request, which can read the start of your question." },
  };
  const track: ProcessStep[] = steps.map((entry, at): ProcessStep => {
    const holds = entry.bytes.some((value) => value !== 0);
    const words = PLAIN[entry.operation] ?? { title: entry.operation, does: entry.note };
    return {
      id: `memory-${at}`, title: words.title, tone: at === last ? (holds ? "warn" : "good") : holds ? "accent" : "plain",
      does: `${words.does} The four bytes now read ${hex(entry.bytes)}.`,
      data: [{ kind: "bytes", bytes: entry.bytes, show: 4, readable: holds ? String.fromCharCode(...entry.bytes.map((value) => (value >= 32 && value < 127 ? value : 46))) : undefined }],
    };
  });
  const story: LiveStory = {
    plain: "While the model works on your question, the question sits in the graphics card's memory. When the answer is done, that memory is wiped before anyone else can get it. Skip the wipe, and the next user can read what was left behind.",
    stage: "wipe",
    steps: track,
    ...stepped(playback, last),
    input: <p>Its first four characters, “{journey.question.slice(0, 4)}”, are stored as the bytes <code>{hex(journey.bytes)}</code>.</p>,
    seenTitle: "What the next user of that memory reads",
    seen: playback.index < last ? <p>Nothing yet: the memory is still in use for your request.</p> : <p><code>{hex(left)}</code> {stale ? "— the start of your question, still there." : "— nothing of yours."}</p>,
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
    strength: { kind: "wipe", before: journey.bytes, after: left },
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
