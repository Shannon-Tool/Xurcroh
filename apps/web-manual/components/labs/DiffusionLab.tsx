"use client";

import { useMemo, useState } from "react";
import { eulerStep, matMul, maxError, rotate2d } from "@/lib/simulation/algebra";
import { LabShell } from "@/components/LabShell";
import { MatrixView } from "@/components/MatrixView";
import { PlaybackControls } from "@/components/PlaybackControls";
import { usePlayback } from "@/components/usePlayback";
import { diffusionScene } from "@/lib/scene/builders";

const initial = [[0.2, -0.4], [0.8, 0.1], [-0.5, 0.7], [1.0, -0.2]];
const velocity = [[0.1, 0.3], [-0.2, 0.4], [0.5, -0.1], [-0.3, 0.2]];
const permutation = [2, 0, 3, 1];

export function DiffusionLab() {
  const [angle, setAngle] = useState(Math.PI / 6);
  const [steps, setSteps] = useState(2);
  const playback = usePlayback(3);
  const state = useMemo(() => {
    const basis = rotate2d(angle);
    const permute = (matrix: number[][]) => permutation.map((index) => matrix[index]!);
    const transformedInitial = matMul(permute(initial), basis[0]!.map((_, i) => basis.map((row) => row[i]!)));
    const transformedVelocity = matMul(permute(velocity), basis[0]!.map((_, i) => basis.map((row) => row[i]!)));
    const transformed = eulerStep(transformedInitial, transformedVelocity, 0.25 * steps);
    const unrotated = matMul(transformed, basis);
    const inversePermutation = permutation.map((_, index) => permutation.indexOf(index));
    const recovered = inversePermutation.map((index) => unrotated[index]!);
    const expected = eulerStep(initial, velocity, 0.25 * steps);
    return { basis, transformedInitial, transformed, recovered, expected, error: maxError(recovered, expected) };
  }, [angle, steps]);
  const stages = [
    { title: "Initial latent", matrix: initial },
    { title: "Rotated and permuted latent", matrix: state.transformedInitial },
    { title: "Denoised transformed latent", matrix: state.transformed },
    { title: "Recovered client latent", matrix: state.recovered },
  ];
  const scene = diffusionScene({ stages: stages.map((stage) => ({ title: stage.title, points: stage.matrix })), step: playback.index });
  return (
    <LabShell id="diffusion" scene={scene} title="Diffusion and video inference" caption="Synthetic latent demonstration. Secret channel rotation and spatial permutation precede Euler denoising; client recovery is checked against the original latent. No model-quality image or video is implied.">
      <div className="lab-controls">
        <label>Rotation {(angle / Math.PI).toFixed(2)}π <input type="range" min="0" max={2 * Math.PI} step={Math.PI / 16} value={angle} onChange={(event) => setAngle(Number(event.target.value))} /></label>
        <label>Euler steps {steps} <input type="range" min="1" max="4" value={steps} onChange={(event) => setSteps(Number(event.target.value))} /></label>
        <PlaybackControls {...playback} canStepBackward={playback.index > 0} onToggle={() => playback.setPlaying(!playback.playing)} />
      </div>
      <div className="home-grid">
        <MatrixView title={stages[playback.index]!.title} matrix={stages[playback.index]!.matrix} />
        <MatrixView title="Original Euler result" matrix={state.expected} />
      </div>
      <p>Recovery error: <strong>{state.error.toExponential(3)}</strong></p>
      <p className="small">Spatial order: [{permutation.join(", ")}]. Flow step z′ = z − Δt·v. Cube structure and recurrent state remain synthetic and separate from official checkpoint quality.</p>
    </LabShell>
  );
}
