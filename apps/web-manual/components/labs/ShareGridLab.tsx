"use client";

import { useMemo, useState } from "react";
import { LabShell } from "@/components/LabShell";
import { PlaybackControls } from "@/components/PlaybackControls";
import { usePlayback } from "@/components/usePlayback";
import { executeShareGrid, mod, PRIME } from "@/lib/simulation/fortress";
import { shareGridScene } from "@/lib/scene/builders";
import type { LiveStory } from "@/components/LiveIO";
import { TOKEN_MS } from "@/lib/show/generation";
import { grouped, type ProcessStep } from "@/lib/show/process";
import { useJourney } from "@/lib/show/question";

const few = (values: Array<bigint | number>, count = 3) => `${values.slice(0, count).map(grouped).join(", ")}${values.length > count ? ", …" : ""}`;

/** Playback positions: numbers, piece A, piece B, the four workers one after another, the sum. */
const LAST = 7;
const CARD = [0, 1, 2, 3, 3, 3, 3, 4];
/** Where pressing a card takes the reader. */
const JUMP = [0, 1, 2, 3, LAST];

export function ShareGridLab() {
  const [epoch, setEpoch] = useState(0);
  const [tamper, setTamper] = useState(false);
  // The real first step of answering: the question's word counts against the model's matching table.
  const journey = useJourney();
  // The last step stays up while the answer is written, one token for every product.
  const playback = usePlayback(LAST, journey.tokens.length * TOKEN_MS + 2600);
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
  const index = playback.index;
  // How many workers have answered, and which one is answering now.
  const answered = Math.max(0, Math.min(4, index - 3));
  const working = index >= 3 && index <= 6 ? index - 3 : null;
  const finished = index >= LAST;
  const role = state.grid.roles[working ?? (finished ? 3 : 0)]!;
  const [pieceA, pieceB] = state.grid.activationShares as [bigint[], bigint[]];
  const piece = state.grid.activationShares[role.activationShare]!;
  const combined = state.grid.combined.map(String).join(", ");
  const used = journey.counts.flatMap((count, at) => (count > 0 ? [at] : []));
  const first = used[0] ?? 0;
  const scene = useMemo(() => shareGridScene({
    roles: state.grid.roles, step: index - 3, tampered: tamper,
    values: {
      client: index === 0 ? `Your device: ${journey.counts.slice(0, 8).join(" ")} …` : index <= 2 ? `Your device: pieces ${grouped(pieceA[first]!)} + ${grouped(pieceB[first]!)}` : undefined,
      roles: state.grid.roles.map((entry, at) => (at === working ? `piece ${grouped(state.grid.activationShares[entry.activationShare]![first]!)} …` : at < answered ? `sent ${grouped(mod(entry.product[0]!))} …` : undefined)),
      output: `scores ${state.grid.combined.slice(0, 3).map(String).join(" ")} …`,
    },
  }), [state, index, tamper, journey, pieceA, pieceB, first, working, answered]);
  const failed = Boolean(state.error);
  const steps: ProcessStep[] = [
    {
      id: "numbers", title: "Words to numbers", tone: "good",
      does: `The model knows ${journey.counts.length} words. Each gets a 1 if it is in your question and a 0 if not: ${journey.words.length} ones, lit here.`,
      data: [{ kind: "numbers", values: journey.counts, mark: used, show: journey.counts.length }],
    },
    {
      id: "piece-a", title: "A random piece", tone: "accent",
      does: `Your device takes one number between 0 and ${grouped(PRIME - 1n)} for every position. This is piece A; on its own it is noise.`,
      data: [{ kind: "numbers", label: "piece A", values: pieceA, show: 2 }],
    },
    {
      id: "piece-b", title: "The other piece", tone: "accent",
      does: `Piece B is whatever must be added to piece A to give your numbers back. For the word “${journey.vocabulary[first]}”: ${journey.counts[first]} = ${grouped(pieceA[first]!)} + ${grouped(pieceB[first]!)}, counted round at ${grouped(PRIME)}.`,
      data: [{ kind: "numbers", label: "piece B", values: pieceB, show: 2 }, { kind: "sum", text: `A + B = ${journey.counts[first]} for “${journey.vocabulary[first]}”` }],
    },
    {
      id: "workers", title: "Four workers", tone: failed && answered > 0 ? "warn" : "accent",
      does: working === null ? "Each of the four workers multiplies one piece of your numbers by one piece of the model."
        : `Worker (${role.activationShare},${role.weightShare}) gets piece ${role.activationShare === 0 ? "A" : "B"} and one piece of the model, multiplies them and sends ${state.grid.roles[working]!.product.length} numbers back. ${failed && working === 0 ? "This worker's numbers were changed on the way." : "Your device checks them before using them."}`,
      data: [{
        kind: "rows",
        rows: state.grid.roles.map((entry, at) => ({
          name: `(${entry.activationShare},${entry.weightShare})`,
          value: at < answered || at === working ? `${grouped(mod(entry.product[0]!))} …` : "waiting",
          state: failed && at === 0 && (at < answered || at === working) ? "bad" as const : at < answered ? "done" as const : at === working ? "now" as const : "todo" as const,
        })),
      }],
    },
    {
      id: "sum", title: failed ? "Refused" : "Added up", tone: failed ? "warn" : "good",
      does: failed ? "One worker's numbers failed their check, so nothing is added up and no answer can be written."
        : "Your device adds the four lists. The random pieces cancel out and one score for each prepared answer is left.",
      data: failed ? [{ kind: "text", text: "nothing combined" }] : [{ kind: "numbers", label: "scores", values: state.grid.combined, show: 6 }],
    },
  ];
  const story: LiveStory = {
    plain: "Your question is turned into numbers and cut into two random pieces. Four workers each work on one piece, and only your device can put their results together to find the reply.",
    stage: "split",
    steps,
    step: CARD[index]!,
    done: finished,
    speed: playback.speed,
    onStep: (card) => playback.goTo(JUMP[card] ?? 0),
    result: [{ kind: "bars", label: "Score of each prepared answer", values: state.grid.combined.map(Number), names: state.grid.combined.map((_, at) => `${at + 1}`) }],
    input: <p>Its {journey.words.length} words become a list of {journey.counts.length} numbers, one for every word the model knows: a 1 for “{journey.words.slice(0, 3).join("”, “")}”… and a 0 for the rest. That list is cut into two random pieces.</p>,
    seenTitle: working === null ? "What a worker sees" : `What worker (${role.activationShare},${role.weightShare}) sees`,
    seen: index < 3 ? <p>Nothing has left your device yet.</p> : working === null ? <p>Each worker saw one piece and sent one list back. None of them saw your words or the scores.</p> : (
      <>
        <p>Its piece of your question: <code>{few(piece, 2)}</code></p>
        <p>The numbers it sends back: <code>{few(role.product.map((value) => mod(value)), 2)}</code></p>
        <p>Nothing in them shows your words.</p>
      </>
    ),
    output: state.error
      ? <p>One worker&apos;s numbers failed their check, so the whole step was thrown away.</p>
      : <p>Added together, the four workers&apos; numbers give one score for each prepared answer: <code>{combined}</code>. The highest score chose the answer, which is then written one token at a time.</p>,
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
    strength: { kind: "pieces", piece: pieceA[first]!, position: first, word: journey.vocabulary[first]!, holder: "Worker (0,0)" },
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
      <h3>{finished ? "All four roles returned; last role" : working === null ? "First role" : "Role"} ({role.activationShare},{role.weightShare}) exact signed product</h3>
      <pre>{role.product.map((value) => value.toString()).join(", ")}</pre>
      <h3>Combined field output</h3>
      <pre>{state.grid.combined.map((value) => value.toString()).join(", ")}</pre>
      <p className="small">Expected from the original integers: [{expected.join(", ")}]. A role is verified before its result can contribute.</p>
    </LabShell>
  );
}
