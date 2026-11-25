"use client";

import { useMemo, useState } from "react";
import { conjugateLinear, matMul, maxError, transpose } from "@/lib/simulation/algebra";
import { LabShell } from "@/components/LabShell";
import { MatrixView } from "@/components/MatrixView";
import { PlaybackControls } from "@/components/PlaybackControls";
import { usePlayback } from "@/components/usePlayback";
import { conjugationScene } from "@/lib/scene/builders";
import type { LiveStory } from "@/components/LiveIO";
import { writingTime } from "@/lib/show/generation";
import { stepped, type ProcessStep } from "@/lib/show/process";
import { useJourney } from "@/lib/show/question";

const short = (values: number[]) => values.map((value) => value.toFixed(2)).join(", ");

const weights: Record<number, number[][]> = {
  2: [[1, 2], [3, 4]],
  3: [[1, 2, 0], [0, 1, 3], [2, 0, 1]],
  4: [[1, 1, 0, 2], [0, 2, 1, 0], [3, 0, 1, 1], [1, 2, 0, 1]],
};

function rotation(dimension: number, angle: number): number[][] {
  return Array.from({ length: dimension }, (_, i) =>
    Array.from({ length: dimension }, (_, j) => {
      if (i === j) return i < 2 ? (i === 0 ? Math.cos(angle) : Math.cos(angle)) : 1;
      if (i === 0 && j === 1) return -Math.sin(angle);
      if (i === 1 && j === 0) return Math.sin(angle);
      return 0;
    }),
  );
}

export function ConjugationLab() {
  const [dimension, setDimension] = useState(2);
  const [angle, setAngle] = useState(Math.PI / 4);
  // The first letters of the chosen question, as positions in the alphabet.
  const journey = useJourney();
  // Numbers, turned, computed on, turned back, compared.
  const playback = usePlayback(4, writingTime(journey.tokens.length));
  const index = playback.index;
  const view = Math.min(index, 3);
  const state = useMemo(() => {
    const weight = weights[dimension]!;
    const basis = rotation(dimension, angle);
    const transformed = conjugateLinear(weight, basis, basis);
    const x = [Array.from({ length: dimension }, (_, i) => journey.letters[i % journey.letters.length]!)];
    const expected = matMul(x, weight);
    const rotatedInput = matMul(x, transpose(basis));
    const transformedOutput = matMul(rotatedInput, transformed);
    const recovered = matMul(transformedOutput, basis);
    const length = (vector: number[]) => Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
    const norm = length(x[0]!);
    const rotatedNorm = length(rotatedInput[0]!);
    return { weight, basis, transformed, x, expected, rotatedInput, transformedOutput, recovered, error: maxError(recovered, expected), norm, rotatedNorm };
  }, [dimension, angle, journey]);

  const scene = useMemo(() => conjugationScene({ x: state.x[0]!, rotatedInput: state.rotatedInput[0]!, expected: state.expected[0]!, transformedOutput: state.transformedOutput[0]!, recovered: state.recovered[0]!, step: view }), [state, view]);
  const stages = [
    { title: "Original computation", matrix: state.expected },
    { title: "Secret basis", matrix: state.basis },
    { title: "Conjugated computation", matrix: state.transformedOutput },
    { title: "Recovered output", matrix: state.recovered },
  ];
  const two = (values: number[]) => values.map((value) => Number(value.toFixed(2)));
  const same = state.error < 1e-9;
  const steps: ProcessStep[] = [
    {
      id: "numbers", title: "Your numbers", tone: "good",
      does: `The first ${dimension} letters of your question as numbers (a = 1, b = 2, …).`,
      data: [{ kind: "numbers", values: two(state.x[0]!), show: 4 }],
    },
    {
      id: "turned", title: "Turned in secret", tone: "accent",
      does: `Your device turns them by an angle only it knows (${(angle / Math.PI).toFixed(2)}π here). The values change; their overall size, ${state.rotatedNorm.toFixed(3)}, does not, which is why this is only a disguise.`,
      data: [{ kind: "numbers", values: two(state.rotatedInput[0]!), show: 4 }],
    },
    {
      id: "computed", title: "Worker computes", tone: "accent",
      does: "The worker multiplies the turned numbers by a model table that was turned the same way. It never handles your numbers the right way round.",
      data: [{ kind: "numbers", values: two(state.transformedOutput[0]!), show: 4 }],
    },
    {
      id: "back", title: "Turned back", tone: "good",
      does: "Your device undoes the rotation on what the worker returned.",
      data: [{ kind: "numbers", values: two(state.recovered[0]!), show: 4 }],
    },
    {
      id: "same", title: same ? "Same as direct" : "Not the same", tone: same ? "good" : "warn",
      does: same ? `Computing directly, without any disguise, gives ${short(state.expected[0]!)}: the same result, to within ${state.error.toExponential(1)}.` : `Computing directly gives ${short(state.expected[0]!)}; the difference is ${state.error.toExponential(1)}.`,
      data: [{ kind: "rows", rows: [{ name: "direct", value: short(state.expected[0]!), state: "done" }, { name: "disguised", value: short(state.recovered[0]!), state: same ? "done" : "bad" }] }],
    },
  ];
  const story: LiveStory = {
    plain: "Before the numbers of your question leave your device they are turned by a secret rotation. The worker computes on the turned version, and your device turns the result back.",
    stage: "prepare",
    steps,
    ...stepped(playback, 4),
    input: <p>Its first letters as numbers (a = 1, b = 2, …): <code>{short(state.x[0]!)}</code>. Your device turns them by a secret angle before sending.</p>,
    seenTitle: "What the worker sees",
    seen: index < 1 ? <p>Nothing has left your device yet.</p> : <p><code>{short(state.rotatedInput[0]!)}</code>: your numbers after the rotation. Their overall size, {state.rotatedNorm.toFixed(3)}, is unchanged.</p>,
    output: <p><code>{short(state.recovered[0]!)}</code> after turning back: the same as computing directly (<code>{short(state.expected[0]!)}</code>).</p>,
    why: "The worker never handles your numbers in their original orientation, which removes the easy reading of them.",
    relies: "The rotation stays secret on your device.",
    limit: "This is disguise, not encryption. The overall size of the numbers is still visible, so it is only an inner layer underneath the random pieces.",
    screenSeen: short(state.rotatedInput[0]!),
    strength: { kind: "disguise", what: "The worker gets your numbers turned by an angle it does not know." },
  };
  return (
    <LabShell id="conjugation" plainTitle="Disguising the numbers before they leave" scene={scene} story={story} watch={playback.watch} title="Orthogonal conjugation laboratory" caption="Adjust basis and dimension. Error measures recovered output against the original computation; equal norms demonstrate structural leakage, not semantic secrecy.">
      <div className="lab-controls">
        <label>Dimension
          <select value={dimension} onChange={(event) => { setDimension(Number(event.target.value)); playback.onReset(); }}>
            {[2, 3, 4].map((value) => <option key={value} value={value}>{value}</option>)}
          </select>
        </label>
        <label>Rotation θ = {(angle / Math.PI).toFixed(2)}π
          <input type="range" min="0" max={2 * Math.PI} step={Math.PI / 16} value={angle} onChange={(event) => setAngle(Number(event.target.value))} />
        </label>
        <PlaybackControls {...playback} canStepBackward={playback.index > 0} onToggle={() => playback.setPlaying(!playback.playing)} />
      </div>
      <div className="home-grid">
        <MatrixView title="Weight W" matrix={state.weight} />
        <MatrixView title="Conjugated weight" matrix={state.transformed} />
      </div>
      <MatrixView title={stages[view]!.title} matrix={stages[view]!.matrix} />
      <p><span className="metric">Equivalence error: <strong>{state.error.toExponential(3)}</strong></span><span className="metric">Input norm: <strong>{state.norm.toFixed(6)}</strong></span><span className="metric">Rotated norm: <strong>{state.rotatedNorm.toFixed(6)}</strong></span></p>
      <p className="small">Functional equivalence is measured. Norm preservation is a counterexample to treating rotation as encryption.</p>
    </LabShell>
  );
}
