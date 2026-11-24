"use client";

import { useMemo, useState } from "react";
import { simulatePipeline } from "@/lib/simulation/pipeline";
import { LabShell } from "@/components/LabShell";
import { PlaybackControls } from "@/components/PlaybackControls";
import { usePlayback } from "@/components/usePlayback";
import { pipelineScene } from "@/lib/scene/builders";

export function PipelineLab() {
  const [nodes, setNodes] = useState(4);
  const [latency, setLatency] = useState(3);
  const [loss, setLoss] = useState(0.1);
  const [requests, setRequests] = useState(2);
  const [fail, setFail] = useState(true);
  const events = useMemo(() => simulatePipeline({ nodes, latencyMs: latency, packetLoss: loss, requests, failNode: fail && nodes > 1 ? 1 : null }), [nodes, latency, loss, requests, fail]);
  const playback = usePlayback(events.length - 1);
  const event = events[playback.index]!;
  const scene = useMemo(() => pipelineScene({ nodes, failNode: fail && nodes > 1 ? 1 : null, events, index: playback.index }), [nodes, fail, events, playback.index]);
  return (
    <LabShell id="pipeline" scene={scene} title="Distributed inference pipeline" caption="Discrete-event simulation. Time and retransmission counts are synthetic controls, not hardware measurements. Every event changes with nodes, latency, loss, workload, or failure settings.">
      <div className="lab-controls">
        <label>Nodes <select value={nodes} onChange={(event) => { setNodes(Number(event.target.value)); playback.onReset(); }}>{[2, 3, 4, 5, 6, 8].map((value) => <option key={value}>{value}</option>)}</select></label>
        <label>Latency {latency} ms <input type="range" min="1" max="20" value={latency} onChange={(event) => setLatency(Number(event.target.value))} /></label>
        <label>Loss {(loss * 100).toFixed(0)}% <input type="range" min="0" max="1" step="0.05" value={loss} onChange={(event) => setLoss(Number(event.target.value))} /></label>
        <label>Requests <select value={requests} onChange={(event) => setRequests(Number(event.target.value))}>{[1, 2, 4, 8].map((value) => <option key={value}>{value}</option>)}</select></label>
        <label><input type="checkbox" checked={fail} onChange={(event) => setFail(event.target.checked)} /> Fail worker 1</label>
        <PlaybackControls {...playback} canStepBackward={playback.index > 0} onToggle={() => playback.setPlaying(!playback.playing)} />
      </div>
      <div className="home-grid">
        {Array.from({ length: nodes }, (_, index) => <div key={index} className={event.stageIndex === index ? "flow-node active" : "flow-node"}>
          <strong>Worker {index}</strong><p>Independent trust boundary</p>
        </div>)}
      </div>
      <pre className="state-inspector">{JSON.stringify(event, null, 2)}</pre>
      <p className="small">Event {playback.index + 1}/{events.length}. Packet is {event.state}. Client and workers use encrypted state and scrubbed KV lifecycle.</p>
    </LabShell>
  );
}
