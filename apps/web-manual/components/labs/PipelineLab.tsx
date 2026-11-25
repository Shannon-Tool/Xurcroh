"use client";

import { useMemo, useState } from "react";
import { simulatePipeline } from "@/lib/simulation/pipeline";
import { LabShell } from "@/components/LabShell";
import { PlaybackControls } from "@/components/PlaybackControls";
import { usePlayback } from "@/components/usePlayback";
import { pipelineScene } from "@/lib/scene/builders";
import type { LiveStory } from "@/components/LiveIO";
import { writingTime } from "@/lib/show/generation";
import type { ProcessStep } from "@/lib/show/process";
import { useJourney } from "@/lib/show/question";

export function PipelineLab() {
  const [nodes, setNodes] = useState(4);
  const [latency, setLatency] = useState(3);
  const [loss, setLoss] = useState(0.1);
  const [requests, setRequests] = useState(2);
  const [fail, setFail] = useState(false);
  const events = useMemo(() => simulatePipeline({ nodes, latencyMs: latency, packetLoss: loss, requests, failNode: fail && nodes > 1 ? 1 : null }), [nodes, latency, loss, requests, fail]);
  const journey = useJourney();
  const playback = usePlayback(events.length - 1, writingTime(journey.tokens.length));
  const index = playback.index;
  const event = events[index]!;
  // The reader's own request is the first one; the others follow it through the same chain.
  const mine = events.map((entry, at) => ({ entry, at })).filter(({ entry }) => entry.request === 0);
  const delivered = mine.find(({ entry }) => entry.state === "decoded")!.at;
  const soFar = mine.filter(({ at }) => at <= index).map(({ entry }) => entry);
  const WORDS = { resent: "lost, sent again", recovered: "failed, recovered", computed: "done" } as const;
  const workerRows = Array.from({ length: nodes }, (_, worker) => {
    const seen = soFar.filter((entry) => entry.stageIndex === worker && entry.state !== "scrubbed").map((entry) => WORDS[entry.state as keyof typeof WORDS]);
    const done = seen.includes("done");
    return { name: `worker ${worker}`, value: seen.length === 0 ? "waiting" : seen.join(", "), state: seen.length === 0 ? "todo" as const : done ? (seen.length > 1 ? "bad" as const : "done" as const) : "now" as const };
  });
  const trouble = soFar.filter((entry) => entry.state === "resent" || entry.state === "recovered").length;
  const card = event.request > 0 ? 3 : event.state === "encrypted" ? 0 : index >= delivered ? 2 : 1;
  const jump = [0, mine[1]?.at ?? 0, delivered, mine.at(-1)!.at + 1];
  const scene = useMemo(() => pipelineScene({ nodes, failNode: fail && nodes > 1 ? 1 : null, events, index }), [nodes, fail, events, index]);
  const count = (state: string) => events.filter((entry) => entry.state === state).length;
  const answered = count("decoded");
  const steps: ProcessStep[] = [
    {
      id: "sealed", title: "Sealed and sent", tone: "cipher",
      does: `Your question leaves your device sealed, as request 1 of ${requests}, at ${events[0]!.timeMs.toFixed(1)} ms on this run's clock.`,
      data: [{ kind: "rows", rows: [{ name: "request 1", value: "sealed, on its way", state: "done" }] }],
    },
    {
      id: "chain", title: "Through the workers", tone: trouble > 0 ? "warn" : "accent",
      does: event.request === 0 && event.stageIndex >= 0
        ? `At ${event.timeMs.toFixed(1)} ms, worker ${event.stageIndex}: ${event.state === "resent" ? "the message was lost on the way and is sent again" : event.state === "recovered" ? "this worker failed; its work is recovered and done again" : event.state === "computed" ? "its part of the work is done and passed on" : "its memory is cleared"}.`
        : `Each of the ${nodes} workers does its part and passes the request on. A lost message is sent again; a failed worker's part is recovered.`,
      data: [{ kind: "rows", rows: workerRows }],
    },
    {
      id: "back", title: "Back on your device", tone: "good",
      does: `The result came back and was opened on your device at ${events[delivered]!.timeMs.toFixed(1)} ms${trouble > 0 ? `, after ${trouble} ${trouble === 1 ? "setback" : "setbacks"} on the way` : ""}. Now the answer can be written.`,
      data: [{ kind: "rows", rows: [{ name: "request 1", value: "answered", state: "done" }, { name: "setbacks", value: String(trouble), state: trouble > 0 ? "bad" : "done" }] }],
    },
    ...(requests > 1 ? [{
      id: "others", title: "The other requests", tone: "plain" as const,
      does: event.request > 0 ? `Request ${event.request + 1} of ${requests} is at ${event.stage}: ${event.state}.` : `${requests - 1} more ${requests === 2 ? "request follows" : "requests follow"} through the same workers.`,
      data: [{ kind: "rows" as const, rows: Array.from({ length: requests - 1 }, (_, other) => {
        const last = events.filter((entry, at) => entry.request === other + 1 && at <= index).at(-1);
        return { name: `request ${other + 2}`, value: last ? `${last.stage}: ${last.state}` : "waiting", state: !last ? "todo" as const : events.some((entry, at) => entry.request === other + 1 && entry.state === "decoded" && at <= index) ? "done" as const : "now" as const };
      }) }],
    }] : []),
  ];
  const story: LiveStory = {
    plain: "Your question passes along a chain of workers. When a message is lost or a worker fails, it is sent again or recovered, and you still get your answer.",
    stage: "send",
    steps,
    step: Math.min(card, steps.length - 1),
    done: index >= delivered,
    speed: playback.speed,
    onStep: (target) => playback.goTo(jump[target] ?? 0),
    input: <p>It travels as request 1 of {requests}, through {nodes} workers in a row.</p>,
    seenTitle: "What is happening right now",
    seen: <p>Request {event.request + 1} at <code>{event.stage}</code>: {event.state}. Step {playback.index + 1} of {events.length}.</p>,
    output: <p>{answered} of {requests} requests answered. {count("resent")} messages had to be sent again; {count("recovered")} recoveries after a failure.</p>,
    released: answered > 0,
    outputTone: answered === requests ? "good" : "warn",
    why: "A lost message or a failed worker delays your answer. It does not lose your question.",
    relies: "A retry or another route is available, as the runtime chapter describes.",
    limit: "The times and counts here are made-up controls for teaching, not measurements of real hardware.",
    attack: {
      label: "fail worker 1",
      on: fail,
      onToggle: () => setFail(!fail),
      outcome: fail ? `Worker 1 failed. ${count("recovered")} recoveries later, ${answered} of ${requests} requests were still answered.` : "No worker fails in this run.",
    },
    screenSeen: `request ${event.request + 1}\n${event.stage}\n${event.state}`,
  };
  return (
    <LabShell id="pipeline" plainTitle="Getting your answer even when a worker fails" scene={scene} story={story} watch={playback.watch} title="Distributed inference pipeline" caption="Discrete-event simulation. Time and retransmission counts are synthetic controls, not hardware measurements. Every event changes with nodes, latency, loss, workload, or failure settings.">
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
