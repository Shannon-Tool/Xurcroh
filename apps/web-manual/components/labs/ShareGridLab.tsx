"use client";

import { useMemo, useState } from "react";
import { LabShell } from "@/components/LabShell";
import { PlaybackControls } from "@/components/PlaybackControls";
import { usePlayback } from "@/components/usePlayback";
import { executeShareGrid } from "@/lib/simulation/fortress";
import { shareGridScene } from "@/lib/scene/builders";

export function ShareGridLab() {
  const [epoch, setEpoch] = useState(0);
  const [tamper, setTamper] = useState(false);
  const playback = usePlayback(4);
  const state = useMemo(() => {
    const honest = executeShareGrid([[1, 2], [3, 4]], [5, 7], epoch);
    if (!tamper) return { grid: honest, error: null };
    try {
      executeShareGrid([[1, 2], [3, 4]], [5, 7], epoch, "0:0");
      return { grid: honest, error: null };
    } catch (error) {
      return { grid: honest, error: error instanceof Error ? error.message : String(error) };
    }
  }, [epoch, tamper]);
  const role = state.grid.roles[Math.min(playback.index, 3)]!;
  const finished = playback.index >= state.grid.roles.length;
  const scene = useMemo(() => shareGridScene({ roles: state.grid.roles, step: playback.index, tampered: tamper }), [state, playback.index, tamper]);
  return (
    <LabShell id="share-grid" scene={scene} title="Additive share-grid laboratory" caption="Four independent role computations reconstruct xWᵀ in the prime field 2³²−5 (the prime just below 2³²). Epoch changes refresh deterministic teaching shares; production uses independent random shares.">
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
      <p className="small">Expected from the original integers: [19, 43]. A role is verified before its result can contribute.</p>
    </LabShell>
  );
}
