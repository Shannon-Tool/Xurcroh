"use client";

import { useEffect, useMemo, useState } from "react";
import { fwht } from "@/lib/simulation/algebra";
import { LabShell } from "@/components/LabShell";
import { PlaybackControls } from "@/components/PlaybackControls";
import { usePlayback } from "@/components/usePlayback";
import { fwhtScene } from "@/lib/scene/builders";
import type { LiveStory } from "@/components/LiveIO";
import { writingTime } from "@/lib/show/generation";
import { stepped, type ProcessStep } from "@/lib/show/process";
import { useJourney } from "@/lib/show/question";

const short = (values: number[]) => values.map((value) => (Number.isInteger(value) ? String(value) : value.toFixed(2))).join(", ");

export function FwhtLab() {
  const [dimension, setDimension] = useState(8);
  // The first letters of the chosen question, as positions in the alphabet; they can still be edited below.
  const journey = useJourney();
  const lettersFor = (count: number) => Array.from({ length: count }, (_, index) => journey.letters[index % journey.letters.length]!).join(",");
  const [text, setText] = useState(lettersFor(8));
  useEffect(() => setText(Array.from({ length: dimension }, (_, index) => journey.letters[index % journey.letters.length]!).join(",")), [journey, dimension]);
  const rounds = Math.log2(dimension);
  const playback = usePlayback(rounds, writingTime(journey.tokens.length));
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
  const tidy = (values: number[]) => values.map((value) => (Number.isInteger(value) ? value : Number(value.toFixed(2))));
  const steps: ProcessStep[] | undefined = result.error ? undefined : result.stages.map((values, round): ProcessStep => {
    if (round === 0) {
      return { id: "numbers", title: "Your numbers", tone: "good", does: `The first ${dimension} letters of your question as numbers (a = 1, b = 2, …). Each position holds one letter.`, data: [{ kind: "numbers", values: tidy(values), show: 8 }] };
    }
    const apart = 2 ** (round - 1);
    const before = result.stages[round - 1]!;
    const last = round === rounds;
    // The last round is also divided by √d, so that the overall size of the list stays the same.
    const scale = last ? Math.sqrt(dimension) : 1;
    return {
      id: `round-${round}`, title: `Round ${round}`, tone: last ? "good" : "accent",
      does: `Numbers ${apart} apart are replaced by their sum and their difference: ${short([before[0]!])} and ${short([before[apart]!])} become ${short([before[0]! + before[apart]!])} and ${short([before[0]! - before[apart]!])}${last ? `, and everything is divided by √${dimension}` : ""}. After this round every number carries ${2 ** round} of your letters.`,
      data: [{ kind: "numbers", values: tidy(values), show: 8 }, { kind: "sum", text: `${short([before[0]!])} ± ${short([before[apart]!])}${last ? ` ÷ √${dimension}` : ""} → ${short([(before[0]! + before[apart]!) / scale])}, ${short([(before[0]! - before[apart]!) / scale])}` }],
    };
  });
  const story: LiveStory = {
    plain: "A fast way to mix the numbers of your question so that every output depends on every input, using only additions and subtractions. Your device uses it to apply its disguises quickly.",
    stage: "prepare",
    steps,
    ...stepped(playback, rounds),
    released: !result.error,
    input: result.error
      ? <p className="error">The list below is not {dimension} numbers.</p>
      : <p>Its first {dimension} letters as numbers (a = 1, b = 2, …): <code>{short(result.stages[0] ?? [])}</code>. They are mixed in {Math.log2(dimension)} rounds.</p>,
    seenTitle: `In between: round ${playback.index} of ${Math.log2(dimension)}`,
    seen: stage.length > 0 ? <p><code>{short(stage)}</code></p> : <p>Nothing yet.</p>,
    output: result.output.length > 0 ? <p><code>{short(result.output)}</code>: every number now carries a part of every letter.</p> : <p>Nothing yet.</p>,
    why: "After mixing, no single position holds a single one of your letters, which is what the secret disguises are built on.",
    relies: "It is combined with secret choices on your device. The mixing step itself is public.",
    limit: "A public mixing step hides nothing by itself. It is a building block, not a protection.",
    screenSeen: short(stage),
    strength: { kind: "disguise", what: "The mixing step is public: anyone who has the mixed list can run the same rounds again and get your numbers back. On its own it hides nothing; your device combines it with secret choices to apply its disguises quickly." },
  };
  return (
    <LabShell id="fwht" plainTitle="Mixing the numbers quickly" scene={scene} story={story} watch={playback.watch} title="Animated Fast Walsh–Hadamard Transform" caption="Every butterfly stage is computed from your vector. The final stage is orthonormal; complexity is O(d log d). Non-power-of-two dimensions are rejected, not silently padded.">
      <div className="lab-controls">
        <label>Dimension
          <select value={dimension} onChange={(event) => {
            const next = Number(event.target.value);
            setDimension(next);
            setText(lettersFor(next));
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
