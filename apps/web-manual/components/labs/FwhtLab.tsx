"use client";

import { useMemo, useState } from "react";
import { fwht } from "@/lib/simulation/algebra";
import { LabShell } from "@/components/LabShell";
import { PlaybackControls } from "@/components/PlaybackControls";
import { usePlayback } from "@/components/usePlayback";
import { fwhtScene } from "@/lib/scene/builders";
import type { LiveStory } from "@/components/LiveIO";

const short = (values: number[]) => values.map((value) => (Number.isInteger(value) ? String(value) : value.toFixed(2))).join(", ");

export function FwhtLab() {
  const [dimension, setDimension] = useState(8);
  const [text, setText] = useState("1,2,3,4,5,6,7,8");
  const playback = usePlayback(Math.log2(dimension));
  const result = useMemo<{ error?: string; stages: number[][]; output: number[] }>(() => {
    const values = text.split(",").map((value) => Number(value.trim()));
    if (values.length !== dimension || values.some((value) => !Number.isFinite(value))) {
      return { error: `Enter ${dimension} finite numbers`, stages: [], output: [] };
    }
    try {
      return fwht(values);
    } catch (error) {
      return { error: error instanceof Error ? error.message : String(error), stages: [], output: [] };
    }
  }, [text, dimension]);
  const stage = result.stages[playback.index] ?? [];
  const scene = useMemo(() => fwhtScene({ stages: result.stages, step: playback.index }), [result, playback.index]);
  const story: LiveStory = {
    plain: "A fast way to mix a list of numbers so that every output depends on every input, using only additions and subtractions. Your device uses it to apply its transforms quickly.",
    input: (
      <>
        <label>Your numbers <input value={text} onChange={(event) => setText(event.target.value)} /></label>
        {result.error ? <p className="error">That is not a list of {dimension} numbers.</p> : <p className="small">{dimension} numbers, mixed in {Math.log2(dimension)} rounds.</p>}
      </>
    ),
    seenTitle: `In between: round ${playback.index} of ${Math.log2(dimension)}`,
    seen: stage.length > 0 ? <p><code>{short(stage)}</code></p> : <p>Nothing yet.</p>,
    output: result.output.length > 0 ? <p><code>{short(result.output)}</code>: every number now carries a part of every input.</p> : <p>Nothing yet.</p>,
    why: "After mixing, no single position holds a single one of your numbers, which is what the secret transforms are built on.",
    relies: "It is combined with secret choices on your device. The mixing step itself is public.",
    limit: "A public mixing step hides nothing by itself. It is a building block, not a protection.",
    screens: { input: short(result.stages[0] ?? []), seen: short(stage), output: short(result.output) },
  };
  return (
    <LabShell id="fwht" scene={scene} story={story} watch={playback.watch} title="Animated Fast Walsh–Hadamard Transform" caption="Every butterfly stage is computed from your vector. The final stage is orthonormal; complexity is O(d log d). Non-power-of-two dimensions are rejected, not silently padded.">
      <div className="lab-controls">
        <label>Dimension
          <select value={dimension} onChange={(event) => {
            const next = Number(event.target.value);
            setDimension(next);
            setText(Array.from({ length: next }, (_, i) => i + 1).join(","));
            playback.onReset();
          }}>
            {[4, 8, 16].map((value) => <option key={value} value={value}>{value}</option>)}
          </select>
        </label>
        <label>Input vector <input value={text} onChange={(event) => setText(event.target.value)} size={Math.max(16, dimension * 2)} /></label>
        <PlaybackControls {...playback} canStepBackward={playback.index > 0} onToggle={() => playback.setPlaying(!playback.playing)} />
      </div>
      {result.error ? <p className="error">{result.error}</p> : null}
      <div className="home-grid" aria-live="polite">
        {result.stages.map((values, index) => <div key={index} className={index === playback.index ? "flow-node active" : "flow-node"}>
          <strong>Stage {index}</strong>
          <pre className="state-inspector">[{values.map((value) => value.toFixed(3)).join(", ")}]</pre>
        </div>)}
      </div>
      {stage.length > 0 ? <p>Current butterfly values: <code>[{stage.map((value) => value.toFixed(4)).join(", ")}]</code></p> : null}
      <p className="small">Direct reference: H{subscript(dimension)}x / √{dimension}. Each stage performs additions and subtractions only.</p>
    </LabShell>
  );
}

function subscript(value: number): string {
  return String(value).replace(/\d/g, (digit) => "₀₁₂₃₄₅₆₇₈₉"[Number(digit)]!);
}
