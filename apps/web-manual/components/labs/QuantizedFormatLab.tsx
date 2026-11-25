"use client";

import { useEffect, useMemo, useState } from "react";
import {
  additiveShares,
  combineShares,
  decodeExl2Mixed,
  decodeQ4_0,
  procedural3inst,
  quantizeFortress,
  quantizationError,
  signedField,
  type QuantizedEncoding,
} from "@/lib/simulation/quantized";
import { LabShell } from "@/components/LabShell";
import { MatrixView } from "@/components/MatrixView";
import { PlaybackControls } from "@/components/PlaybackControls";
import { usePlayback } from "@/components/usePlayback";
import { quantizedScene } from "@/lib/scene/builders";
import type { LiveStory } from "@/components/LiveIO";
import { writingTime } from "@/lib/show/generation";
import { stepped, type ProcessStep } from "@/lib/show/process";
import { useJourney } from "@/lib/show/question";

/** Which of the lab's six exact views goes with each step: codes and decoded numbers share the first. */
const VIEW = [0, 0, 1, 2, 3, 5];

export function QuantizedFormatLab() {
  const [encoding, setEncoding] = useState<QuantizedEncoding>("gguf-q4_0");
  // Four stored codes of the model that answers the chosen question; they can still be edited below.
  const journey = useJourney();
  const [codes, setCodes] = useState(journey.codes.join(", "));
  useEffect(() => setCodes(journey.codes.join(", ")), [journey]);
  const playback = usePlayback(5, writingTime(journey.tokens.length));
  const index = playback.index;
  const view = VIEW[index]!;
  const state = useMemo(() => {
    try {
      const values = codes.split(",").map((value) => Number(value.trim()));
      if (values.length < 2 || values.some((value) => !Number.isInteger(value) || value < 0)) return null;
      const decoded = encoding === "gguf-q4_0"
        ? decodeQ4_0(0.5, values)
        : encoding === "exl2-mixed"
          ? decodeExl2Mixed([
              { bits: 2, scale: 0.25, codes: values.slice(0, 2) },
              { bits: 4, scale: 0.5, codes: values.slice(2) },
            ])
          : values.map(procedural3inst);
      const quantized = quantizeFortress(decoded);
      const recovered = quantized.integers.map((value) => value * quantized.scale);
      const shares = additiveShares(quantized.integers, 2, 731);
      const combined = combineShares(shares).map(signedField);
      return {
        decoded,
        quantized,
        recovered,
        shares: shares.map((share) => share.map(Number)),
        combined,
        error: quantizationError(decoded, recovered),
        exact: combined.every((value, index) => value === quantized.integers[index]),
      };
    } catch {
      return null;
    }
  }, [encoding, codes]);

  const stages = useMemo(() => (state ? [
    { title: "Source-decoded weights", matrix: [state.decoded] },
    { title: "Independent Fortress quantization", matrix: [state.quantized.integers] },
    { title: "Random weight share A", matrix: [state.shares[0]!] },
    { title: "Complementary weight share B", matrix: [state.shares[1]!] },
    { title: "Recovered dequantized weights", matrix: [state.recovered] },
    { title: "Exact modular share recombination", matrix: [state.combined] },
  ] : []), [state]);
  const scene = useMemo(() => quantizedScene({ stages: stages.map((stage) => ({ title: stage.title, values: stage.matrix[0]! })), step: view }), [stages, view]);
  const round = (values: number[]) => values.map((value) => Number(value.toFixed(3)));
  const steps: ProcessStep[] | undefined = state ? [
    {
      id: "codes", title: "Stored codes", tone: "plain",
      does: "A model file does not keep its numbers at full size. It keeps a small code for each one; these are four of them, taken from the model that answers your question.",
      data: [{ kind: "numbers", label: "codes", values: codes.split(",").map((value) => value.trim()), show: 6 }],
    },
    {
      id: "decoded", title: "Decoded", tone: "plain",
      does: "The file format says how a code becomes a number. Decoding gives the model's numbers back as fractions.",
      data: [{ kind: "numbers", values: round(state.decoded), show: 4 }],
    },
    {
      id: "whole", title: "Whole numbers", tone: "accent",
      does: `Pieces can only be made of whole numbers, so each fraction is scaled to one (one unit is ${state.quantized.scale}). From here on the arithmetic is exact.`,
      data: [{ kind: "numbers", values: state.quantized.integers, show: 4 }],
    },
    {
      id: "piece-a", title: "A random piece", tone: "accent",
      does: "One random number is taken for every whole number. This is piece A, kept by one operator. It has no link to the model.",
      data: [{ kind: "numbers", label: "piece A", values: state.shares[0]!, show: 2 }],
    },
    {
      id: "piece-b", title: "The other piece", tone: "accent",
      does: `Piece B is what must be added to piece A to give the whole numbers back, kept by a different operator. For the first number: ${state.shares[0]![0]} + ${state.shares[1]![0]} gives ${state.quantized.integers[0]}, counted round at 4294967291.`,
      data: [{ kind: "numbers", label: "piece B", values: state.shares[1]!, show: 2 }],
    },
    {
      id: "back", title: state.exact ? "Added back" : "Does not add back", tone: state.exact ? "good" : "warn",
      does: state.exact ? "Adding the two pieces gives exactly the whole numbers that went in, so the model can be used without ever being stored whole." : "The pieces do not add back to what went in.",
      data: [{ kind: "numbers", values: state.combined, show: 4 }, { kind: "sum", text: state.exact ? "= the whole numbers of step 3" : "≠ the whole numbers of step 3" }],
    },
  ] : undefined;
  const story: LiveStory = {
    plain: "The model that answers your question is not stored in one place. Its numbers are kept as small codes, turned into whole numbers and cut into two random pieces. Adding the pieces gives exactly the same numbers back.",
    stage: "prepare",
    steps,
    ...stepped(playback, 5),
    released: Boolean(state?.exact),
    input: state
      ? <p>Four of the codes of the model that answers it: <code>{codes}</code>. As whole numbers: <code>{state.quantized.integers.join(", ")}</code>.</p>
      : <p className="error">The codes below are not whole numbers.</p>,
    seenTitle: "What the holder of one piece sees",
    seen: !state || index < 3 ? <p>Nothing yet: the model has not been cut into pieces.</p> : <p><code>{state.shares[0]!.join(", ")}</code>: random numbers with no visible link to the model.</p>,
    output: state ? <p>Adding both pieces gives <code>{state.combined.join(", ")}</code>: {state.exact ? "exactly the whole numbers that went in." : "not what went in."}</p> : <p>Nothing yet.</p>,
    outputTone: state && !state.exact ? "warn" : "good",
    why: "One piece on its own is random, so a machine that stores one piece does not learn the model, and cannot answer for it either.",
    relies: "The two pieces are stored by different operators, and each split uses fresh randomness. This lab uses a fixed seed so that it repeats.",
    limit: "Getting the same numbers back shows the arithmetic is exact. It says nothing about how good the model is.",
    screenSeen: state ? state.shares[0]!.slice(0, 2).join("\n") : "",
  };
  return (
    <LabShell id="quantized-formats" plainTitle="Storing the model in pieces" scene={state ? scene : undefined} story={story} watch={playback.watch} title="Quantized format and share-grid laboratory" caption="Synthetic numeric examples compute source decoding, independent Fortress quantization, and exact modular shares. They do not execute a checkpoint or establish model quality.">
      <div className="lab-controls">
        <label>Encoding
          <select value={encoding} onChange={(event) => { setEncoding(event.target.value as QuantizedEncoding); playback.onReset(); }}>
            <option value="gguf-q4_0">GGUF Q4_0</option>
            <option value="exl2-mixed">EXL2 mixed widths</option>
            <option value="exl3-3inst">EXL3 procedural 3INST</option>
          </select>
        </label>
        <label>Integer codes
          <input value={codes} onChange={(event) => setCodes(event.target.value)} />
        </label>
        <PlaybackControls {...playback} canStepBackward={index > 0} onToggle={() => playback.setPlaying(!playback.playing)} />
      </div>
      {state ? <>
        <MatrixView title={stages[view]!.title} matrix={stages[view]!.matrix} />
        <p><span className="metric">Source quality error: <strong>{state.error.toExponential(3)}</strong></span><span className="metric">Exact modular recombination: <strong>{state.exact ? "yes" : "no"}</strong></span></p>
      </> : <p role="alert" className="error">Enter at least two non-negative integer codes, separated by commas.</p>}
      <p className="small">The two quantization stages are deliberately separate. Exact recombination proves arithmetic, not semantic secrecy or source-model equivalence.</p>
    </LabShell>
  );
}
