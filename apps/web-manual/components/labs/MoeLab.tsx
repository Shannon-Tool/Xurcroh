"use client";

import { useMemo, useState } from "react";
import { combineMoe, routeMoe } from "@/lib/simulation/moe";
import { LabShell } from "@/components/LabShell";
import { PlaybackControls } from "@/components/PlaybackControls";
import { usePlayback } from "@/components/usePlayback";

const hidden = [[1, 0], [0, 1], [1, 1], [2, 0], [0, 2], [2, 2]];
const router = [[1, 0], [0, 1], [1, 1], [2, 0.5]];

export function MoeLab() {
  const [topK, setTopK] = useState(1);
  const [constantLoad, setConstantLoad] = useState(4);
  const [protectedMode, setProtectedMode] = useState(true);
  const playback = usePlayback(3);
  const calculation = useMemo(() => {
    try {
      const plan = routeMoe(hidden, router, topK, constantLoad);
      const outputs = router.map((_, expert) => Array.from({ length: constantLoad }, (_, slot) => [expert + 1, (slot + 1) * 0.5]));
      const combined = combineMoe(plan, outputs);
      const realLoads = plan.dummy.map((row) => row.filter((dummy) => !dummy).length);
      return { state: { plan, combined, realLoads }, error: null };
    } catch (error) {
      return { state: null, error: error instanceof Error ? error.message : String(error) };
    }
  }, [topK, constantLoad]);
  const state = calculation.state;
  const loads = state ? (protectedMode ? state.plan.loads : state.realLoads) : [];
  return (
    <LabShell id="moe" title="Mixture-of-Experts routing simulator" caption="Real router logits, top-k dispatch, dummy padding, and weighted combination. Constant loads hide volume, not timing, cache, TLB, power, or co-residency channels.">
      <div className="lab-controls">
        <label>Top-k <select value={topK} onChange={(event) => { setTopK(Number(event.target.value)); playback.onReset(); }}>{[1, 2, 3].map((value) => <option key={value}>{value}</option>)}</select></label>
        <label>Capacity {constantLoad} <input type="range" min="2" max="8" value={constantLoad} onChange={(event) => setConstantLoad(Number(event.target.value))} /></label>
        <label><input type="checkbox" checked={protectedMode} onChange={(event) => setProtectedMode(event.target.checked)} /> Constant-load protected dispatch</label>
        <PlaybackControls {...playback} canStepBackward={playback.index > 0} onToggle={() => playback.setPlaying(!playback.playing)} />
      </div>
      {calculation.error ? <p role="alert" className="error">{calculation.error}. Increase capacity or reduce top-k.</p> : null}
      {state ? <>
        <div className="home-grid">
          {loads.map((load, expert) => <div key={expert} className={playback.index === expert % 4 ? "flow-node active" : "flow-node"}>
            <strong>Expert {expert}</strong><p>{load} buffer slots</p>
            <p>{state.realLoads[expert]} real / {load - state.realLoads[expert]!} dummy</p>
          </div>)}
        </div>
        <pre className="state-inspector">{JSON.stringify({ tokenIndices: state.plan.tokenIndices, weights: state.plan.weights, dummy: state.plan.dummy }, null, 2)}</pre>
        <p>Combined output: <code>{JSON.stringify(state.combined)}</code></p>
      </> : null}
      <p className="small">Dummy slots have zero combine weight. Padding does not prove elimination of timing or hardware side channels.</p>
    </LabShell>
  );
}
