"use client";

import { useEffect, useMemo, useState } from "react";
import { openOnion, ReplayWindow, sealOnion, type OnionLayer } from "@/lib/simulation/onion";
import { LabShell } from "@/components/LabShell";
import { PlaybackControls } from "@/components/PlaybackControls";
import { usePlayback } from "@/components/usePlayback";
import { onionScene } from "@/lib/scene/builders";
import type { LiveStory } from "@/components/LiveIO";
import { TOKEN_MS } from "@/lib/show/generation";
import { hexText, type ProcessStep } from "@/lib/show/process";
import { useJourney } from "@/lib/show/question";

/** Playback positions: readable; three layers put on; three relays; arrival. */
const LAST = 7;
const CARD = [0, 1, 1, 1, 2, 3, 4, 5];
const JUMP = [0, 1, 4, 5, 6, LAST];
/** The relay whose packet the reader can alter: the middle one, so that one relay passes it and one never gets it. */
const ALTERED = 1;
const HOPS = [{ nextHop: "worker-1" }, { nextHop: "worker-2" }, { nextHop: "client" }];

export function OnionLab() {
  const [layers, setLayers] = useState<OnionLayer[]>([]);
  const [failure, setFailure] = useState<string | null>(null);
  const [tamper, setTamper] = useState(false);
  const [compromised, setCompromised] = useState(false);
  const [colluding, setColluding] = useState(false);
  const [result, setResult] = useState("Not verified");
  // The chosen question is the message that gets wrapped.
  const journey = useJourney();
  const message = journey.question;
  const plain = useMemo(() => new TextEncoder().encode(message), [message]);
  const [exit, setExit] = useState<string | null>(null);
  const [opens, setOpens] = useState(true);
  const playback = usePlayback(LAST, journey.tokens.length * TOKEN_MS + 2600);
  const index = playback.index;
  const replay = useMemo(() => new ReplayWindow(), []);
  // Which relay the packet is at (or was last at), for the exact view below the stage.
  const hop = Math.max(0, Math.min(2, index - 4));
  const layer = useMemo(() => {
    const original = layers[hop];
    if (!original || !tamper || hop !== ALTERED) return original;
    const ciphertext = new Uint8Array(original.ciphertext);
    ciphertext[0]! ^= 1;
    return { ...original, ciphertext };
  }, [layers, hop, tamper]);

  useEffect(() => {
    let current = true;
    setFailure(null);
    void sealOnion(HOPS, plain)
      .then((sealed) => {
        if (current) setLayers(sealed);
      })
      .catch((error: unknown) => {
        if (current) setFailure(error instanceof Error ? error.message : String(error));
      });
    return () => {
      current = false;
    };
  }, [plain]);

  // What arrives at the far end, and whether the altered packet still authenticates at its relay.
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
    const target = layers[ALTERED];
    if (!target || !tamper) {
      setOpens(true);
      return;
    }
    let current = true;
    const ciphertext = new Uint8Array(target.ciphertext);
    ciphertext[0]! ^= 1;
    void openOnion({ ...target, ciphertext }).then(() => current && setOpens(true)).catch(() => current && setOpens(false));
    return () => {
      current = false;
    };
  }, [layers, tamper]);

  async function verify() {
    if (!layer) return;
    if (!replay.accept(hop)) {
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

  const ready = layers.length === HOPS.length;
  const wrapped = Math.max(0, Math.min(3, index));
  const atRelay = index >= 4 && index <= 6;
  const refused = !opens;
  // With an altered packet, relay 2 refuses: relay 3 and the far end never receive anything.
  const reaches = (relay: number) => !refused || relay <= ALTERED;
  const packet = (relay: number) => (relay === ALTERED && tamper && layer && hop === ALTERED ? layer.ciphertext : layers[relay]?.ciphertext ?? new Uint8Array());
  const phase = index <= 3 ? "wrap" as const : index >= LAST ? "arrived" as const : "relay" as const;
  const note = !ready ? undefined
    : index === 0 ? `readable: ${hexText(plain, 5)}`
      : index <= 3 ? `sealed ×${wrapped}: ${hexText(layers[3 - wrapped]!.ciphertext, 5)}`
        : atRelay ? `sealed: ${hexText(packet(Math.min(hop, refused ? ALTERED : 2)), 5)}`
          : refused ? `sealed: ${hexText(packet(ALTERED), 5)}` : `readable again: ${hexText(plain, 5)}`;
  const scene = useMemo(
    () => onionScene({ hops: layers.map((candidate) => candidate.nextHop), step: hop, tampered: tamper, compromised, colluding, phase, wrapped, note, stoppedAt: refused ? ALTERED : null }),
    [layers, hop, tamper, compromised, colluding, phase, wrapped, note, refused],
  );

  const sealed = layer ? Array.from(layer.ciphertext.slice(0, 10), (value) => value.toString(16).padStart(2, "0")).join(" ") : "";
  const relayStep = (relay: number): ProcessStep => {
    const got = reaches(relay);
    const bad = refused && relay === ALTERED;
    return {
      id: `relay-${relay}`, title: `Relay ${relay + 1}`, tone: bad ? "warn" : "cipher",
      does: !got ? `Relay ${relay + 1} never receives the packet: relay ${ALTERED + 1} refused it.`
        : bad ? `Relay ${relay + 1} is handed bytes with one bit changed. Its authenticity check fails, so it removes nothing and passes nothing on.`
          : relay === 2 ? `Relay 3 removes the last layer and hands what is inside to ${HOPS[2]!.nextHop}. So the last relay holds the payload itself: in the pipeline that payload is one random piece (see the splitting lab), here it is the readable question so that you can compare the bytes.`
            : `Relay ${relay + 1} removes its own layer with its own key. Inside it finds one thing it can read, the next stop (${HOPS[relay]!.nextHop}), and bytes it cannot read.`,
      data: !got ? [{ kind: "text", text: "nothing received" }]
        : [{ kind: "bytes", label: "gets", bytes: packet(relay), show: 6 }, { kind: "rows", rows: [{ name: "to", value: bad ? "refused" : HOPS[relay]!.nextHop, state: bad ? "bad" : "done" }] }],
    };
  };
  const steps: ProcessStep[] = !ready ? [] : [
    {
      id: "plain", title: "Readable", tone: "good",
      does: `Your question as ${plain.length} bytes, one for each character. Anyone holding these bytes can read it.`,
      data: [{ kind: "bytes", bytes: plain, show: 6, readable: message.length > 14 ? `${message.slice(0, 13)}…` : message }],
    },
    {
      id: "wrap", title: "Sealed three times", tone: "cipher",
      does: wrapped === 0 ? "Your device seals the bytes three times, the layer for the last relay first."
        : `Layer ${wrapped} of 3 is on: sealed with a key shared with relay ${4 - wrapped} (AES-256-GCM). The packet is now ${layers[3 - wrapped]!.ciphertext.length} bytes and none of them is readable.`,
      data: [
        { kind: "rows", rows: [2, 1, 0].map((relay, order) => ({ name: `relay ${relay + 1}`, value: order < wrapped ? `${layers[relay]!.ciphertext.length} b` : "wait", state: order < wrapped - 1 ? "done" as const : order === wrapped - 1 ? "now" as const : "todo" as const })) },
        ...(wrapped > 0 ? [{ kind: "bytes" as const, bytes: layers[3 - wrapped]!.ciphertext, show: 6 }] : []),
      ],
    },
    relayStep(0), relayStep(1), relayStep(2),
    {
      id: "arrive", title: refused ? "Nothing arrives" : "Readable again", tone: refused ? "warn" : "good",
      does: refused ? "The altered packet was stopped on the way, so your question is not delivered and no answer comes back."
        : `The last layer is off. The ${plain.length} bytes that come out are the ones that went in${exit === message ? "" : " (checking…)"}; the first two relays never held anything readable.`,
      data: refused ? [{ kind: "text", text: "not delivered" }] : [{ kind: "bytes", bytes: exit === null ? plain : new TextEncoder().encode(exit), show: 6, readable: message.length > 14 ? `${message.slice(0, 13)}…` : message }],
    },
  ];
  const story: LiveStory = {
    plain: "Your question is wrapped in layers of encryption, one for each relay on the way. A relay can remove only its own layer, which tells it where to send the rest and nothing more.",
    stage: "send",
    steps: ready ? steps : undefined,
    step: CARD[index]!,
    done: index >= LAST,
    speed: playback.speed,
    onStep: (card) => playback.goTo(JUMP[card] ?? 0),
    input: <p>The whole question is wrapped three times, once for each relay it will pass.</p>,
    seenTitle: atRelay ? `What relay ${hop + 1} sees` : "What a relay sees",
    seen: failure ? <p role="alert">The layers could not be made in this browser: {failure}</p>
      : !ready ? <p>Wrapping the question…</p>
        : index < 4 ? <p>No relay has received anything yet.</p>
          : !atRelay ? <p>Relays 1 and 2 saw a next stop and sealed bytes, nothing else. The last relay hands on the payload, which in the pipeline is a random piece. None of them saw the whole route.</p>
            : !reaches(hop) ? <p>Nothing: the packet was refused before it got here.</p>
              : <><p>Next stop: <code>{refused && hop === ALTERED ? "unknown, the layer would not open" : layer!.nextHop}</code></p><p>Sealed bytes: <code>{sealed} …</code> ({layer!.ciphertext.length} bytes it cannot read)</p></>,
    output: opens
      ? <p>{exit === null ? "…" : `“${exit}” reaches the far end unchanged, and the answer comes back the same way.`}</p>
      : <p>Refused. The altered layer failed its authenticity check, so your question is not passed on.</p>,
    released: opens,
    outputTone: opens ? "good" : "warn",
    why: "Each relay learns only the next stop and cannot see the whole route. Only the last one holds the payload, and in the pipeline that payload is a random piece.",
    relies: "Relays do not share what they see with one another. The layers here use real ECDH and AES-GCM.",
    limit: "Relays that work together can link the route, encryption does not hide timing or how much is sent, and the last relay holds whatever was wrapped: it must not be readable on its own.",
    attack: {
      label: "alter the sealed bytes",
      on: tamper,
      onToggle: () => setTamper(!tamper),
      outcome: opens ? "Untouched: every layer authenticates." : "Caught: one flipped bit and the layer no longer authenticates.",
    },
    screenSeen: layer ? `to ${layer.nextHop}\n${sealed.slice(0, 17)}` : "",
  };
  return (
    <LabShell id="onion" plainTitle="Hiding the route your question takes" scene={ready ? scene : undefined} story={story} watch={playback.watch} title="Interactive onion-routing simulation" caption="Actual ECDH/AES-GCM layers with per-hop visibility, compromised-relay controls, authentication failure, and replay rejection. This is a protocol demonstration, not a traffic-analysis guarantee.">
      <div className="lab-controls">
        <label><input type="checkbox" checked={tamper} onChange={(event) => setTamper(event.target.checked)} /> Alter the packet on its way to relay 2</label>
        <label><input type="checkbox" checked={compromised} onChange={(event) => setCompromised(event.target.checked)} /> Compromised relay</label>
        <label><input type="checkbox" checked={colluding} onChange={(event) => setColluding(event.target.checked)} /> Colluding hops</label>
        <button onClick={verify}>Decrypt / verify</button>
        <PlaybackControls {...playback} canStepBackward={index > 0} onToggle={() => playback.setPlaying(!playback.playing)} />
      </div>
      <div className="home-grid">
        {layers.map((candidate, at) => <div key={at} className={atRelay && at === hop ? "flow-node active" : "flow-node"}>
          <strong>Hop {at}</strong><p>Next: {candidate.nextHop}</p>
          <p>{candidate.returnSecret ? "Inner return secret present" : "Only next-hop metadata"}</p>
        </div>)}
      </div>
      {failure ? <p role="alert">The encryption layers could not be made: {failure}</p> : layer ? <pre className="state-inspector">{JSON.stringify({ nextHop: layer.nextHop, nonce: Array.from(layer.nonce), ciphertext: Array.from(layer.ciphertext).slice(0, 24), compromised, colluding, result }, null, 2)}</pre> : <p>Generating encryption layers…</p>}
      <p className="small">A relay observes ciphertext and routing metadata. Collusion can reveal circuit linkage; encryption does not prove anonymity or protect against malicious computation.</p>
    </LabShell>
  );
}
