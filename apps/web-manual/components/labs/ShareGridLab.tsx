"use client";

import { useMemo, useState } from "react";
import { LabShell } from "@/components/LabShell";
import { PlaybackControls } from "@/components/PlaybackControls";
import { usePlayback } from "@/components/usePlayback";
import { executeShareGrid } from "@/lib/simulation/fortress";
import { shareGridScene } from "@/lib/scene/builders";
import type { LiveStory } from "@/components/LiveIO";
import { useJourney } from "@/lib/show/question";

const few = (values: Array<bigint | number>, count = 3) => `${values.slice(0, count).map(String).join(", ")}${values.length > count ? ", …" : ""}`;

export function ShareGridLab() {
  const [epoch, setEpoch] = useState(0);
  const [tamper, setTamper] = useState(false);
  const playback = usePlayback(4);
  // The real first step of answering: the question's word counts against the model's matching table.
  const journey = useJourney();
  const expected = journey.scores;
  const state = useMemo(() => {
    const honest = executeShareGrid(journey.weightRows, journey.counts, epoch);
    if (!tamper) return { grid: honest, error: null };
    try {
      executeShareGrid(journey.weightRows, journey.counts, epoch, "0:0");
      return { grid: honest, error: null };
    } catch (error) {
      return { grid: honest, error: error instanceof Error ? error.message : String(error) };
    }
  }, [journey, epoch, tamper]);
  const role = state.grid.roles[Math.min(playback.index, 3)]!;
  const finished = playback.index >= state.grid.roles.length;
  const scene = useMemo(() => shareGridScene({ roles: state.grid.roles, step: playback.index, tampered: tamper }), [state, playback.index, tamper]);
  const piece = state.grid.activationShares[role.activationShare]!;
  const combined = state.grid.combined.map(String).join(", ");
  const story: LiveStory = {
    plain: "Your question is turned into numbers and cut into two random pieces. Four workers each work on one piece, and only your device can put their results together to find the reply.",
    stage: "split",
    input: <p>Its {journey.words.length} words become a list of {journey.counts.length} numbers, one for every word the model knows: a 1 for “{journey.words.slice(0, 3).join("”, “")}”… and a 0 for the rest. That list is cut into two random pieces.</p>,
    seenTitle: `What worker (${role.activationShare},${role.weightShare}) sees`,
    seen: (
      <>
        <p>Its piece of your question: <code>{few(piece)}</code></p>
        <p>The numbers it sends back: <code>{few(role.product)}</code></p>
        <p>Nothing in them shows your words.</p>
      </>
    ),
    output: state.error
      ? <p>One worker&apos;s numbers failed their check, so the whole step was thrown away.</p>
      : <p>Added together, the four workers&apos; numbers give one score for each prepared answer: <code>{combined}</code>. The highest score chose the answer above.</p>,
    released: !state.error,
    outputTone: state.error ? "warn" : "good",
    why: "One piece on its own is a list of uniformly random numbers, so a single worker learns nothing about your question from it.",
    relies: "Workers that hold different pieces do not pool them, and the pieces are freshly random every time. This lab uses repeatable teaching pieces.",
    limit: "Two workers holding the two matching pieces can rebuild your question together.",
    attack: {
      label: "tamper with worker (0,0)",
      on: tamper,
      onToggle: () => setTamper(!tamper),
      outcome: state.error ? "Caught: that worker's numbers failed their check, and the step was rejected." : "No tampering: all four workers pass their checks.",
    },
    screenSeen: piece.slice(0, 2).map(String).join("\n"),
  };
  return (
    <LabShell id="share-grid" plainTitle="Splitting your question so no worker can read it" scene={scene} story={story} watch={playback.watch} title="Additive share-grid laboratory" caption="Four independent role computations reconstruct xWᵀ in the prime field 2³²−5 (the prime just below 2³²). Epoch changes refresh deterministic teaching shares; production uses independent random shares.">
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
