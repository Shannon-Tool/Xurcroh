"use client";

import { useEffect, useMemo, useState } from "react";
import { openOnion, ReplayWindow, sealOnion, type OnionLayer } from "@/lib/simulation/onion";
import { LabShell } from "@/components/LabShell";
import { PlaybackControls } from "@/components/PlaybackControls";
import { usePlayback } from "@/components/usePlayback";

export function OnionLab() {
  const [layers, setLayers] = useState<OnionLayer[]>([]);
  const [tamper, setTamper] = useState(false);
  const [compromised, setCompromised] = useState(false);
  const [colluding, setColluding] = useState(false);
  const [result, setResult] = useState("Not verified");
  const playback = usePlayback(3);
  const replay = useMemo(() => new ReplayWindow(), []);
  const layer = useMemo(() => {
    const original = layers[playback.index];
    if (!original || !tamper) return original;
    const ciphertext = new Uint8Array(original.ciphertext);
    ciphertext[0]! ^= 1;
    return { ...original, ciphertext };
  }, [layers, playback.index, tamper]);

  useEffect(() => {
    void sealOnion([{ nextHop: "worker-1" }, { nextHop: "worker-2" }, { nextHop: "client" }], new TextEncoder().encode("private tensor"))
      .then(setLayers);
  }, []);

  async function verify() {
    if (!layer) return;
    if (!replay.accept(playback.index)) {
      setResult("Replay rejected");
      return;
    }
    try {
      const opened = await openOnion(layer);
      setResult(new TextDecoder().decode(opened));
    } catch {
      setResult("Authentication failed");
    }
  }

  return (
    <LabShell id="onion" title="Interactive onion-routing simulation" caption="Actual ECDH/AES-GCM layers with per-hop visibility, compromised-relay controls, authentication failure, and replay rejection. This is a protocol demonstration, not a traffic-analysis guarantee.">
      <div className="lab-controls">
        <label><input type="checkbox" checked={tamper} onChange={(event) => setTamper(event.target.checked)} /> Tamper current ciphertext</label>
        <label><input type="checkbox" checked={compromised} onChange={(event) => setCompromised(event.target.checked)} /> Compromised relay</label>
        <label><input type="checkbox" checked={colluding} onChange={(event) => setColluding(event.target.checked)} /> Colluding hops</label>
        <button onClick={verify}>Decrypt / verify</button>
        <PlaybackControls {...playback} canStepBackward={playback.index > 0} onToggle={() => playback.setPlaying(!playback.playing)} />
      </div>
      <div className="home-grid">
        {layers.map((candidate, index) => <div key={index} className={index === playback.index ? "flow-node active" : "flow-node"}>
          <strong>Hop {index}</strong><p>Next: {candidate.nextHop}</p>
          <p>{candidate.returnSecret ? "Inner return secret present" : "Only next-hop metadata"}</p>
        </div>)}
      </div>
      {layer ? <pre className="state-inspector">{JSON.stringify({ nextHop: layer.nextHop, nonce: Array.from(layer.nonce), ciphertext: Array.from(layer.ciphertext).slice(0, 24), compromised, colluding, result }, null, 2)}</pre> : <p>Generating encryption layers…</p>}
      <p className="small">A relay observes ciphertext and routing metadata. Collusion can reveal circuit linkage; encryption does not prove anonymity or protect against malicious computation.</p>
    </LabShell>
  );
}
