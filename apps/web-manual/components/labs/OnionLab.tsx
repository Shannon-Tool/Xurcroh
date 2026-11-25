"use client";

import { useEffect, useMemo, useState } from "react";
import { openOnion, ReplayWindow, sealOnion, type OnionLayer } from "@/lib/simulation/onion";
import { LabShell } from "@/components/LabShell";
import { PlaybackControls } from "@/components/PlaybackControls";
import { usePlayback } from "@/components/usePlayback";
import { onionScene } from "@/lib/scene/builders";
import type { LiveStory } from "@/components/LiveIO";
import { useJourney } from "@/lib/show/question";

export function OnionLab() {
  const [layers, setLayers] = useState<OnionLayer[]>([]);
  const [tamper, setTamper] = useState(false);
  const [compromised, setCompromised] = useState(false);
  const [colluding, setColluding] = useState(false);
  const [result, setResult] = useState("Not verified");
  // The chosen question is the message that gets wrapped.
  const message = useJourney().question;
  const [exit, setExit] = useState<string | null>(null);
  const [opens, setOpens] = useState(true);
  // One step for each of the three relays.
  const playback = usePlayback(2);
  const replay = useMemo(() => new ReplayWindow(), []);
  const layer = useMemo(() => {
    const original = layers[playback.index];
    if (!original || !tamper) return original;
    const ciphertext = new Uint8Array(original.ciphertext);
    ciphertext[0]! ^= 1;
    return { ...original, ciphertext };
  }, [layers, playback.index, tamper]);

  useEffect(() => {
    let current = true;
    void sealOnion([{ nextHop: "worker-1" }, { nextHop: "worker-2" }, { nextHop: "client" }], new TextEncoder().encode(message))
      .then((sealed) => {
        if (current) setLayers(sealed);
      });
    return () => {
      current = false;
    };
  }, [message]);

  // What arrives at the far end, and whether the layer in front of the reader still authenticates.
  useEffect(() => {
    const last = layers[layers.length - 1];
    if (!last) return;
    let current = true;
    void openOnion(last).then((opened) => {
      if (current) setExit(new TextDecoder().decode(opened));
    });
    return () => {
      current = false;
    };
  }, [layers]);
  useEffect(() => {
    if (!layer) return;
    let current = true;
    void openOnion(layer).then(() => current && setOpens(true)).catch(() => current && setOpens(false));
    return () => {
      current = false;
    };
  }, [layer]);

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

  const scene = useMemo(() => onionScene({ hops: layers.map((candidate) => candidate.nextHop), step: playback.index, tampered: tamper, compromised, colluding }), [layers, playback.index, tamper, compromised, colluding]);

  const sealed = layer ? Array.from(layer.ciphertext.slice(0, 10), (value) => value.toString(16).padStart(2, "0")).join(" ") : "";
  const story: LiveStory = {
    plain: "Your question is wrapped in layers of encryption, one for each relay on the way. A relay can remove only its own layer, which tells it where to send the rest and nothing more.",
    stage: "send",
    input: <p>The whole question is wrapped three times, once for each relay it will pass.</p>,
    seenTitle: `What relay ${playback.index + 1} sees`,
    seen: layer
      ? <><p>Next stop: <code>{layer.nextHop}</code></p><p>Sealed bytes: <code>{sealed} …</code> ({layer.ciphertext.length} bytes it cannot read)</p></>
      : <p>Wrapping the question…</p>,
    output: opens
      ? <p>{exit === null ? "…" : `“${exit}” reaches the far end unchanged, and the answer comes back the same way.`}</p>
      : <p>Refused. The altered layer failed its authenticity check, so your question is not passed on.</p>,
    released: opens,
    outputTone: opens ? "good" : "warn",
    why: "Each relay learns only the next stop. It cannot read your question or see the whole route.",
    relies: "Relays do not share what they see with one another. The layers here use real ECDH and AES-GCM.",
    limit: "Relays that work together can link the route, and encryption does not hide timing or how much is sent.",
    attack: {
      label: "alter the sealed bytes",
      on: tamper,
      onToggle: () => setTamper(!tamper),
      outcome: opens ? "Untouched: every layer authenticates." : "Caught: one flipped bit and the layer no longer authenticates.",
    },
    screenSeen: layer ? `to ${layer.nextHop}\n${sealed.slice(0, 17)}` : "",
  };
  return (
    <LabShell id="onion" plainTitle="Hiding the route your question takes" scene={layers.length > 0 ? scene : undefined} story={story} watch={playback.watch} title="Interactive onion-routing simulation" caption="Actual ECDH/AES-GCM layers with per-hop visibility, compromised-relay controls, authentication failure, and replay rejection. This is a protocol demonstration, not a traffic-analysis guarantee.">
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
