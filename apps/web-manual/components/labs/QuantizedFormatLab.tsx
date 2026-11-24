"use client";

import { useMemo, useState } from "react";
import {
  additiveShares,
  combineShares,
  decodeExl2Mixed,
  decodeQ4_0,
  procedural3inst,
  quantizeFortress,
  quantizationError,
  type QuantizedEncoding,
} from "@/lib/simulation/quantized";
import { LabShell } from "@/components/LabShell";
import { MatrixView } from "@/components/MatrixView";
import { PlaybackControls } from "@/components/PlaybackControls";
import { usePlayback } from "@/components/usePlayback";
import { quantizedScene } from "@/lib/scene/builders";

export function QuantizedFormatLab() {
  const [encoding, setEncoding] = useState<QuantizedEncoding>("gguf-q4_0");
  const [codes, setCodes] = useState("0, 1, 2, 3");
  const playback = usePlayback(5);
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
      const combined = combineShares(shares).map(Number);
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
  const scene = useMemo(() => quantizedScene({ stages: stages.map((stage) => ({ title: stage.title, values: stage.matrix[0]! })), step: playback.index }), [stages, playback.index]);
  return (
    <LabShell id="quantized-formats" scene={state ? scene : undefined} title="Quantized format and share-grid laboratory" caption="Synthetic numeric examples compute source decoding, independent Fortress quantization, and exact modular shares. They do not execute a checkpoint or establish model quality.">
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
        <PlaybackControls {...playback} canStepBackward={playback.index > 0} onToggle={() => playback.setPlaying(!playback.playing)} />
      </div>
      {state ? <>
        <MatrixView title={stages[playback.index]!.title} matrix={stages[playback.index]!.matrix} />
        <p><span className="metric">Source quality error: <strong>{state.error.toExponential(3)}</strong></span><span className="metric">Exact modular recombination: <strong>{state.exact ? "yes" : "no"}</strong></span></p>
      </> : <p role="alert" className="error">Enter at least two non-negative integer codes, separated by commas.</p>}
      <p className="small">The two quantization stages are deliberately separate. Exact recombination proves arithmetic, not semantic secrecy or source-model equivalence.</p>
    </LabShell>
  );
}
