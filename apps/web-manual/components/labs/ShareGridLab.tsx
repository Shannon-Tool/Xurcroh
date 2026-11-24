"use client";

import { useMemo, useState } from "react";
import { LabShell } from "@/components/LabShell";
import { PlaybackControls } from "@/components/PlaybackControls";
import { usePlayback } from "@/components/usePlayback";
import { executeShareGrid } from "@/lib/simulation/fortress";
import { shareGridScene } from "@/lib/scene/builders";
import type { LiveStory } from "@/components/LiveIO";

const WEIGHT = [[1, 2], [3, 4]];

function parseNumbers(text: string): number[] | null {
  const values = text.split(",").map((value) => Number(value.trim()));
  return values.length === 2 && values.every((value) => Number.isInteger(value) && value >= 0 && value <= 9999) ? values : null;
}

export function ShareGridLab() {
  const [epoch, setEpoch] = useState(0);
  const [tamper, setTamper] = useState(false);
  const [numbers, setNumbers] = useState("5, 7");
  const playback = usePlayback(4);
  const parsed = useMemo(() => parseNumbers(numbers), [numbers]);
  const activation = useMemo(() => parsed ?? [5, 7], [parsed]);
  const expected = WEIGHT.map((row) => row[0]! * activation[0]! + row[1]! * activation[1]!);
  const state = useMemo(() => {
    const honest = executeShareGrid(WEIGHT, activation, epoch);
    if (!tamper) return { grid: honest, error: null };
    try {
      executeShareGrid(WEIGHT, activation, epoch, "0:0");
      return { grid: honest, error: null };
    } catch (error) {
      return { grid: honest, error: error instanceof Error ? error.message : String(error) };
    }
  }, [activation, epoch, tamper]);
  const role = state.grid.roles[Math.min(playback.index, 3)]!;
  const finished = playback.index >= state.grid.roles.length;
  const scene = useMemo(() => shareGridScene({ roles: state.grid.roles, step: playback.index, tampered: tamper }), [state, playback.index, tamper]);
  const piece = state.grid.activationShares[role.activationShare]!.map(String);
  const combined = state.grid.combined.map(String).join(", ");
  const story: LiveStory = {
    plain: "Your numbers are cut into two random pieces. Four workers each multiply one piece, and only your device can add the four answers back together.",
    input: (
      <>
        <label>Your two numbers <input value={numbers} onChange={(event) => { setNumbers(event.target.value); playback.onReset(); }} /></label>
        {parsed ? <p className="small">They are multiplied by a fixed table of weights.</p> : <p className="error">Enter two whole numbers from 0 to 9999, separated by a comma.</p>}
      </>
    ),
    seenTitle: `What worker (${role.activationShare},${role.weightShare}) sees`,
    seen: (
      <>
        <p>Its piece of your numbers: <code>{piece.join(", ")}</code></p>
        <p>The product it sends back: <code>{role.product.map(String).join(", ")}</code></p>
      </>
    ),
    output: state.error
      ? <p>Nothing. A product failed its check, so no result was released.</p>
      : <p>Result: {combined}. The same as multiplying your numbers directly ({expected.join(", ")}).</p>,
    outputTone: state.error ? "warn" : "good",
    why: "One piece on its own is a uniformly random number, so a single worker learns nothing about your numbers from it.",
    relies: "Workers that hold different pieces do not pool them, and the pieces are freshly random every time. This lab uses repeatable teaching pieces.",
    limit: "Two workers holding the two matching pieces can rebuild your numbers together.",
    attack: {
      label: "tamper with worker (0,0)",
      on: tamper,
      onToggle: () => setTamper(!tamper),
      outcome: state.error ? "Caught: that worker's product failed its check, and the step was rejected." : "No tampering: all four products pass their checks.",
    },
    screens: { input: activation.join(", "), seen: piece.join("\n"), output: state.error ? "rejected" : combined },
  };
  return (
    <LabShell id="share-grid" scene={scene} story={story} watch={playback.watch} title="Additive share-grid laboratory" caption="Four independent role computations reconstruct xWᵀ in the prime field 2³²−5 (the prime just below 2³²). Epoch changes refresh deterministic teaching shares; production uses independent random shares.">
      <div className="lab-controls">
        <label>Share epoch
          <input type="number" min="0" max="20" value={epoch} onChange={(event) => { setEpoch(Number(event.target.value)); playback.onReset(); }} />
        </label>
        <label><input type="checkbox" checked={tamper} onChange={(event) => setTamper(event.target.checked)} /> Tamper role (0,0)</label>
        <PlaybackControls {...playback} canStepBackward={playback.index > 0} onToggle={() => playback.setPlaying(!playback.playing)} />
      </div>
      {state.error ? <p role="alert">Rejected: {state.error}</p> : null}
      <h3>{finished ? "All four roles returned; last role" : "Role"} ({role.activationShare},{role.weightShare}) exact signed product</h3>
      <pre>{role.product.map((value) => value.toString()).join(", ")}</pre>
      <h3>Combined field output</h3>
      <pre>{state.grid.combined.map((value) => value.toString()).join(", ")}</pre>
      <p className="small">Expected from the original integers: [{expected.join(", ")}]. A role is verified before its result can contribute.</p>
    </LabShell>
  );
}
