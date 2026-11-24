"use client";

import { useMemo, useState } from "react";
import { conjugateLinear, matMul, maxError, transpose } from "@/lib/simulation/algebra";
import { LabShell } from "@/components/LabShell";
import { MatrixView } from "@/components/MatrixView";
import { PlaybackControls } from "@/components/PlaybackControls";
import { usePlayback } from "@/components/usePlayback";
import { conjugationScene } from "@/lib/scene/builders";

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
  const playback = usePlayback(3);
  const state = useMemo(() => {
    const weight = weights[dimension]!;
    const basis = rotation(dimension, angle);
    const transformed = conjugateLinear(weight, basis, basis);
    const x = [Array.from({ length: dimension }, (_, i) => i + 1)];
    const expected = matMul(x, weight);
    const rotatedInput = matMul(x, transpose(basis));
    const transformedOutput = matMul(rotatedInput, transformed);
    const recovered = matMul(transformedOutput, basis);
    const length = (vector: number[]) => Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
    const norm = length(x[0]!);
    const rotatedNorm = length(rotatedInput[0]!);
    return { weight, basis, transformed, x, expected, rotatedInput, transformedOutput, recovered, error: maxError(recovered, expected), norm, rotatedNorm };
  }, [dimension, angle]);

  const scene = useMemo(() => conjugationScene({ x: state.x[0]!, rotatedInput: state.rotatedInput[0]!, expected: state.expected[0]!, transformedOutput: state.transformedOutput[0]!, recovered: state.recovered[0]!, step: playback.index }), [state, playback.index]);
  const stages = [
    { title: "Original computation", matrix: state.expected },
    { title: "Secret basis", matrix: state.basis },
    { title: "Conjugated computation", matrix: state.transformedOutput },
    { title: "Recovered output", matrix: state.recovered },
  ];
  return (
    <LabShell id="conjugation" scene={scene} title="Orthogonal conjugation laboratory" caption="Adjust basis and dimension. Error measures recovered output against the original computation; equal norms demonstrate structural leakage, not semantic secrecy.">
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
      <MatrixView title={stages[playback.index]!.title} matrix={stages[playback.index]!.matrix} />
      <p><span className="metric">Equivalence error: <strong>{state.error.toExponential(3)}</strong></span><span className="metric">Input norm: <strong>{state.norm.toFixed(6)}</strong></span><span className="metric">Rotated norm: <strong>{state.rotatedNorm.toFixed(6)}</strong></span></p>
      <p className="small">Functional equivalence is measured. Norm preservation is a counterexample to treating rotation as encryption.</p>
    </LabShell>
  );
}
