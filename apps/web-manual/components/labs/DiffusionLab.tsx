"use client";

import { useMemo, useState } from "react";
import { eulerStep, matMul, maxError, rotate2d } from "@/lib/simulation/algebra";
import { LabShell } from "@/components/LabShell";
import { MatrixView } from "@/components/MatrixView";
import { PlaybackControls } from "@/components/PlaybackControls";
import { usePlayback } from "@/components/usePlayback";
import { diffusionScene } from "@/lib/scene/builders";
import type { LiveStory } from "@/components/LiveIO";
import { paintingTime } from "@/lib/show/generation";
import type { ProcessStep } from "@/lib/show/process";
import { usePictureJourney } from "@/lib/show/question";
import { getModel } from "@/lib/simulation/show";

const short = (rows: number[][]) => rows.slice(0, 2).map((row) => row.map((value) => value.toFixed(2)).join(", ")).join(" | ");

const initial = [[0.2, -0.4], [0.8, 0.1], [-0.5, 0.7], [1.0, -0.2]];
const velocity = [[0.1, 0.3], [-0.2, 0.4], [0.5, -0.1], [-0.3, 0.2]];
const permutation = [2, 0, 3, 1];

export function DiffusionLab() {
  const [angle, setAngle] = useState(Math.PI / 6);
  const [steps, setSteps] = useState(2);
  const picture = usePictureJourney();
  // Random grid, turned and shuffled, cleaned up by the worker, shuffle undone, picture decoded.
  const playback = usePlayback(4, paintingTime(getModel(picture.id).units));
  const index = playback.index;
  const view = Math.min(index, 3);
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
  const scene = diffusionScene({ stages: stages.map((stage) => ({ title: stage.title, points: stage.matrix })), step: view });
  const flat = (rows: number[][]) => rows.flat().map((value) => Number(value.toFixed(2)));
  const same = state.error < 1e-9;
  const track: ProcessStep[] = [
    {
      id: "noise", title: "Random grid", tone: "good",
      does: "Every picture starts as a grid of random numbers on your device. Four points of that grid, two numbers each.",
      data: [{ kind: "numbers", values: flat(initial), show: 4 }],
    },
    {
      id: "disguised", title: "Turned and shuffled", tone: "accent",
      does: `Your device turns the grid by a secret angle (${(angle / Math.PI).toFixed(2)}π here) and puts its points in a secret order before sending it. A disguise, not encryption.`,
      data: [{ kind: "numbers", values: flat(state.transformedInitial), show: 4 }],
    },
    {
      id: "cleaned", title: "Worker cleans up", tone: "accent",
      does: `The worker runs ${steps} clean-up ${steps === 1 ? "step" : "steps"} on the disguised grid. It does the heavy work without knowing the real arrangement.`,
      data: [{ kind: "numbers", values: flat(state.transformed), show: 4 }],
    },
    {
      id: "undone", title: same ? "Shuffle undone" : "Does not match", tone: same ? "good" : "warn",
      does: `Your device undoes the order and the rotation. The grid equals the one cleaned up directly, to within ${state.error.toExponential(1)}.`,
      data: [{ kind: "numbers", values: flat(state.recovered), show: 4 }, { kind: "sum", text: `direct: ${flat(state.expected).slice(0, 2).join("  ")} …` }],
    },
    {
      id: "decoded", title: "Picture decoded", tone: "good",
      does: "The cleaned grid is turned into colours, one step at a time: each step is one product on two random pieces, and each makes the picture a little cleaner. (The picture itself comes from the manual's toy decoder.)",
      data: [{ kind: "text", text: `“${picture.request}”` }],
    },
  ];
  const story: LiveStory = {
    plain: "A picture is made by cleaning up a grid of random numbers, step by step, until it matches your request. The grid is turned and shuffled before it leaves your device, the worker does the clean-up without knowing the real arrangement, and your device undoes the shuffle.",
    stage: "compute",
    kind: "picture",
    steps: track,
    step: index,
    done: index >= 4,
    speed: playback.speed,
    onStep: playback.goTo,
    input: <p>Every picture starts as a grid of random numbers. A small part of that grid: <code>{short(initial)} …</code></p>,
    seenTitle: "What the worker sees",
    seen: index < 1 ? <p>Nothing has left your device yet.</p> : <p><code>{short(index >= 2 ? state.transformed : state.transformedInitial)} …</code>: the grid turned and in a different order{index >= 2 ? ", after its clean-up" : ""}.</p>,
    output: <p>After {steps} clean-up steps and undoing the shuffle, the grid is the same as cleaning it up directly, to within {state.error.toExponential(1)}. The picture comes from the manual&apos;s toy decoder.</p>,
    why: "The worker does the heavy step on a grid whose arrangement it does not know.",
    relies: "The rotation and the shuffle order stay secret on your device.",
    limit: "Turning and shuffling are disguise, not encryption; the random pieces are the real protection. No picture model runs here.",
    screenSeen: short(state.transformedInitial),
  };
  return (
    <LabShell id="diffusion" plainTitle="Making a picture without showing the worker the real grid" scene={scene} story={story} watch={playback.watch} title="Diffusion and video inference" caption="Synthetic latent demonstration. Secret channel rotation and spatial permutation precede Euler denoising; client recovery is checked against the original latent. No model-quality image or video is implied.">
      <div className="lab-controls">
        <label>Rotation {(angle / Math.PI).toFixed(2)}π <input type="range" min="0" max={2 * Math.PI} step={Math.PI / 16} value={angle} onChange={(event) => setAngle(Number(event.target.value))} /></label>
        <label>Euler steps {steps} <input type="range" min="1" max="4" value={steps} onChange={(event) => setSteps(Number(event.target.value))} /></label>
        <PlaybackControls {...playback} canStepBackward={index > 0} onToggle={() => playback.setPlaying(!playback.playing)} />
      </div>
      <div className="home-grid">
        <MatrixView title={stages[view]!.title} matrix={stages[view]!.matrix} />
        <MatrixView title="Original Euler result" matrix={state.expected} />
      </div>
      <p>Recovery error: <strong>{state.error.toExponential(3)}</strong></p>
      <p className="small">Spatial order: [{permutation.join(", ")}]. Flow step z′ = z − Δt·v. Cube structure and recurrent state remain synthetic and separate from official checkpoint quality.</p>
    </LabShell>
  );
}
