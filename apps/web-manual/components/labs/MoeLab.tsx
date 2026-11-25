"use client";

import { useMemo, useState } from "react";
import { combineMoe, routeMoe } from "@/lib/simulation/moe";
import { LabShell } from "@/components/LabShell";
import { PlaybackControls } from "@/components/PlaybackControls";
import { usePlayback } from "@/components/usePlayback";
import { moeScene } from "@/lib/scene/builders";
import type { LiveStory } from "@/components/LiveIO";
import { writingTime } from "@/lib/show/generation";
import { stepped, type ProcessStep } from "@/lib/show/process";
import { useJourney } from "@/lib/show/question";

const HIDDEN = [[1, 0], [0, 1], [1, 1], [2, 0], [0, 2], [2, 2]];
const router = [[1, 0], [0, 1], [1, 1], [2, 0.5]];

export function MoeLab() {
  const [topK, setTopK] = useState(1);
  const [constantLoad, setConstantLoad] = useState(4);
  const [protectedMode, setProtectedMode] = useState(true);
  // The first six words of the chosen question stand for its tokens; which word goes where depends on the question.
  const journey = useJourney();
  // Words, router scores, real work, dummy work, combined.
  const playback = usePlayback(4, writingTime(journey.tokens.length));
  const index = playback.index;
  const hidden = useMemo(() => HIDDEN.map((_, index) => HIDDEN[(index + journey.hash) % HIDDEN.length]!), [journey]);
  const words = HIDDEN.map((_, index) => journey.words[index % journey.words.length]!);
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
  }, [hidden, topK, constantLoad]);
  const state = calculation.state;
  const loads = useMemo(() => (state ? (protectedMode ? state.plan.loads : state.realLoads) : []), [state, protectedMode]);
  const scene = useMemo(() => moeScene({ tokens: hidden.length, plan: state ? state.plan : null, loads, protectedMode, step: index % 4 }), [state, loads, protectedMode, index, hidden.length]);
  const even = loads.every((load) => load === loads[0]);
  // The router's score of the first word for every expert, and where each word was sent.
  const firstScores = router.map((expert) => expert.reduce((sum, value, at) => sum + value * hidden[0]![at]!, 0));
  const sentTo = words.map((_, token) => (state ? state.plan.tokenIndices.flatMap((slots, expert) => (slots.some((entry, slot) => entry === token && !state.plan.dummy[expert]![slot]) ? [expert] : [])) : []));
  const steps: ProcessStep[] | undefined = state ? [
    {
      id: "words", title: "Six words", tone: "good",
      does: "Six words of your question, standing for the pieces of text the model works on.",
      data: [{ kind: "text", text: words.join(" · ") }],
    },
    {
      id: "router", title: "Router scores", tone: "accent",
      does: `A small table inside the model scores every word for every expert. For “${words[0]}” the four scores are ${firstScores.join(", ")}, so it goes to expert ${sentTo[0]!.join(" and ")}. Top-k is ${topK}: each word goes to its ${topK === 1 ? "best expert" : `${topK} best experts`}.`,
      data: [{ kind: "rows", rows: words.map((word, token) => ({ name: word, value: `expert ${sentTo[token]!.join(", ")}`, state: "done" as const })) }],
    },
    {
      id: "real", title: "Real work", tone: "warn",
      does: `Counting the real work shows where your words went: the experts got ${state.realLoads.join(", ")} words. Anyone who can count this learns something about your question.`,
      data: [{ kind: "bars", values: state.realLoads, names: state.realLoads.map((_, expert) => `${expert}`), top: -1 }],
    },
    {
      id: "padded", title: protectedMode ? "Dummy work added" : "No dummy work", tone: even ? "good" : "warn",
      does: protectedMode ? `Dummy work is added until every expert has ${constantLoad} slots: ${state.plan.loads.map((load, expert) => `${load - state.realLoads[expert]!} dummy for expert ${expert}`).join(", ")}. The counts are now ${state.plan.loads.join(", ")}.`
        : `Padding is off, so the counts stay ${loads.join(", ")} and the busy experts stand out.`,
      data: [{ kind: "bars", values: loads, names: loads.map((_, expert) => `${expert}`), top: -1 }],
    },
    {
      id: "combined", title: "Results combined", tone: "good",
      does: `Each expert returns its results. Your device keeps the real ones, weighted by the router's scores, and gives the dummy ones weight zero. For “${words[0]}”: ${state.combined[0]!.map((value) => Number(value.toFixed(2))).join(", ")}.`,
      data: [{ kind: "numbers", label: `“${words[0]}”`, values: state.combined[0]!.map((value) => Number(value.toFixed(2))), show: 4 }],
    },
  ] : undefined;
  const story: LiveStory = {
    plain: "Different words of your question are handled by different experts inside the model. Dummy work is added so that every expert always receives the same amount, and a watcher cannot tell which experts your question needed.",
    stage: "compute",
    steps,
    ...stepped(playback, 4),
    input: <p>Six of its words, “{words.join("”, “")}”, are each sent to the expert that suits them best.</p>,
    seenTitle: "What someone counting each expert's work sees",
    seen: state && index < 2 ? <p>Nothing yet: no work has been handed out.</p> : state ? <p><code>{(index === 2 ? state.realLoads : loads).join(", ")}</code> {index !== 2 && even ? "— the same for every expert." : "— uneven: the busy experts stand out."}</p> : <p>Nothing: this setting does not fit. Raise the capacity below.</p>,
    output: state ? <p>The experts&apos; results for all {state.combined.length} words were combined. Dummy work counts for nothing.{even ? "" : " But the uneven loads gave away which experts your question used."}</p> : <p>Nothing yet.</p>,
    released: state !== null,
    outputTone: state && even ? "good" : "warn",
    why: "Which experts a question uses says something about the question. Equal loads take that signal away.",
    relies: "Dummy work looks the same as real work to whoever is counting.",
    limit: "Padding hides how much each expert does. It does not hide timing, cache or power side channels.",
    attack: {
      label: "turn the padding off",
      on: !protectedMode,
      onToggle: () => setProtectedMode(!protectedMode),
      outcome: protectedMode ? "With padding on, every expert receives the same number of slots." : `Without padding the loads are ${loads.join(", ")}: they show where your words went.`,
    },
    screenSeen: loads.join(", "),
  };
  return (
    <LabShell id="moe" plainTitle="Hiding which parts of the model your question used" scene={scene} story={story} watch={playback.watch} title="Mixture-of-Experts routing simulator" caption="Real router logits, top-k dispatch, dummy padding, and weighted combination. Constant loads hide volume, not timing, cache, TLB, power, or co-residency channels.">
      <div className="lab-controls">
        <label>Top-k <select value={topK} onChange={(event) => { setTopK(Number(event.target.value)); playback.onReset(); }}>{[1, 2, 3].map((value) => <option key={value}>{value}</option>)}</select></label>
        <label>Capacity {constantLoad} <input type="range" min="2" max="8" value={constantLoad} onChange={(event) => setConstantLoad(Number(event.target.value))} /></label>
        <label><input type="checkbox" checked={protectedMode} onChange={(event) => setProtectedMode(event.target.checked)} /> Constant-load protected dispatch</label>
        <PlaybackControls {...playback} canStepBackward={index > 0} onToggle={() => playback.setPlaying(!playback.playing)} />
      </div>
      {calculation.error ? <p role="alert" className="error">{calculation.error}. Increase capacity or reduce top-k.</p> : null}
      {state ? <>
        <div className="home-grid">
          {loads.map((load, expert) => <div key={expert} className={index % 4 === expert % 4 ? "flow-node active" : "flow-node"}>
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
