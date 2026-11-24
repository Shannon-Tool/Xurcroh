import type { Direction } from "@/lib/scene/orbit";
import type { Scene3D, SceneLink, SceneNode, Vec3 } from "@/lib/scene/types";
import type { Attack, Beat } from "@/lib/show/timeline";
import type { ShowKind } from "@/lib/simulation/show";

export interface ShowSceneInput {
  kind: ShowKind;
  beat: Beat;
  /** Units whose output is already on the device's screen. */
  revealed: number;
  units: number;
  rejections: number;
  /** Role (2·activation share + weight share) hit by the attack on stage, while a rejection is shown. */
  tamperRole: number | null;
  /** Which attack caused that rejection. */
  rejectKind?: Attack | null;
  /** Everyday wording, or the terms the chapters use (the default). */
  words?: "plain" | "technical";
}

const THING: Record<ShowKind, string> = { text: "token", picture: "step of the picture", clip: "frame of the clip" };
const STEPS = 8;

export function workerName(role: number): string {
  return `Worker (${role >> 1},${role & 1})`;
}

/** The relay an altered packet is stopped at, for a given attacked role. */
export function relayOf(role: number): number {
  return role % 3;
}

function progress(input: ShowSceneInput): string {
  if (input.kind === "text") return `Token ${input.revealed} of ${input.units}`;
  if (input.kind === "picture") return `Step ${input.revealed} of ${input.units}`;
  return `Frame ${Math.ceil(input.revealed / 2)} of ${input.units / 2}`;
}

function rejection(input: ShowSceneInput, plain: boolean): string {
  const role = input.tamperRole ?? 0;
  if (input.rejectKind === "software") {
    return plain
      ? `${workerName(role)} is not running the expected software. Its report does not match, so your device sends it nothing and uses a checked worker instead.`
      : `${workerName(role)}'s attestation report carries the wrong measurement. The client refuses to release shares to it and the step is re-run.`;
  }
  if (input.rejectKind === "packet") {
    return plain
      ? `Someone altered a sealed packet. Relay ${relayOf(role) + 1} noticed, dropped it, and your device sent it again.`
      : `A sealed packet was altered in transit. Relay ${relayOf(role) + 1}'s AEAD check failed, the packet was dropped and the step is re-sent.`;
  }
  return plain
    ? `${workerName(role)} sent back a wrong answer. Your device caught it, threw the step away and asked again.`
    : `${workerName(role)} returned a product that fails the check. The step is rejected and sent again.`;
}

function plainCaption(input: ShowSceneInput): string {
  const thing = THING[input.kind];
  switch (input.beat) {
    case "split": return `Step 1 of ${STEPS} · Split. Your device shuffles its numbers for the first ${thing} and cuts them into two random pieces.`;
    case "attest": return `Step 2 of ${STEPS} · Attest. Each worker shows a signed report of the software it runs. Your device checks it before sending anything.`;
    case "seal": return `Step 3 of ${STEPS} · Seal. Each piece is locked in three layers of encryption, one for each relay.`;
    case "route": return `Step 4 of ${STEPS} · Route. The sealed pieces pass through three relays. Each relay removes one layer and sees only where to send the rest.`;
    case "compute": return `Step 5 of ${STEPS} · Compute. Inside its enclave, each worker multiplies the two pieces it holds. Its screen shows all it sees: noise.`;
    case "return": return `Step 6 of ${STEPS} · Return. The four results travel back, sealed, through the relays.`;
    case "verify": return `Step 7 of ${STEPS} · Check. Your device tests every result. A wrong one would be thrown out here.`;
    case "combine": return `Step 8 of ${STEPS} · Combine. The four checked results add up to the first ${thing}, which appears only on your device.`;
    case "stream": return `${progress(input)}. The same eight steps repeat for every ${thing}.`;
    case "rejected": return rejection(input, true);
    case "done": return `Complete. Every ${thing} was split, sealed, computed in enclaves, checked, and put together on your device.`;
  }
}

/** One line on why the step on stage matters for protection. */
export function showWhy(beat: Beat, rejectKind?: Attack | null): string {
  switch (beat) {
    case "split": return "Shuffling is a disguise; the random pieces are what hide your data. One piece alone is noise.";
    case "attest": return "A worker running anything else is refused. An enclave is one more layer, not proof: it rests on the hardware and its maker.";
    case "seal": return "From here on, everything on the wire is unreadable without the keys.";
    case "route": return "No relay knows both who is asking and who is answering.";
    case "compute": return "The heavy work is done by machines that cannot read what they are working on.";
    case "return": return "Each result is only one of four parts, sealed, and useless on its own.";
    case "verify": return "A worker that cheats or makes a mistake is caught before its result is used.";
    case "combine": return "The readable result exists in one place only: your device.";
    case "stream": return "This holds while workers with different pieces do not pool them. “Whose eyes?” shows what each party gets.";
    case "rejected": return rejectKind === "software" ? "Nothing of yours ever reaches a worker whose software cannot be confirmed."
      : rejectKind === "packet" ? "Sealed data cannot be changed on the way without the change being noticed."
        : "Every result is checked, so a single wrong number cannot slip into the answer.";
    case "done": return "The relays and enclaves on this stage are stand-ins inside the page; the arithmetic, the sealing and the signature checks are real.";
  }
}

function caption(input: ShowSceneInput): string {
  if (input.words === "plain") return plainCaption(input);
  const thing = THING[input.kind];
  switch (input.beat) {
    case "split": return `Split: the client permutes its input for the first ${thing} with a secret order and splits it into two additive shares.`;
    case "attest": return "Attest: each worker's enclave signs a report over its code measurement and a fresh nonce; the client verifies it before releasing shares.";
    case "seal": return "Seal: each share is wrapped in three AEAD layers, one per relay of the Tor circuit.";
    case "route": return "Route: the sealed shares cross three relays. Each strips one layer and learns only the next hop.";
    case "compute": return "Compute: inside its enclave each worker multiplies the activation share and the weight share it holds.";
    case "return": return "Return: the four products travel back through the circuit.";
    case "verify": return "Verify: the client tests every product against a random challenge (Freivalds).";
    case "combine": return `Combine: the four verified products add up to the first ${thing}, which appears only on the client.`;
    case "stream": return `${progress(input)}. The same eight steps repeat for every ${thing}.`;
    case "rejected": return rejection(input, false);
    case "done": return `Complete. Every ${thing} was computed on shares inside attested enclaves, verified and combined on the client.`;
  }
}

const DEVICE: Vec3 = [-8.2, 0.75, 0];
const RELAYS: Vec3[] = [[-4.6, 0.5, 1.3], [-2, 0.5, -1.3], [0.6, 0.5, 1.3]];

function workerAt(role: number): Vec3 {
  return [4.6 + (role >> 1) * 3.1, 0.55, -1.55 + (role & 1) * 3.1];
}

/**
 * The stage for the live demonstration: your device, a three-relay Tor circuit, and four workers each
 * inside a hardware enclave. The relays and enclaves are stand-ins; what they are shown doing is computed.
 */
export function showScene(input: ShowSceneInput): Scene3D {
  const { beat } = input;
  const kind = beat === "rejected" ? input.rejectKind ?? "result" : null;
  const role = input.tamperRole ?? 0;
  const stream = beat === "stream";
  const routing = beat === "route" || stream;
  const returning = beat === "return" || stream;
  const sealed = beat === "verify" || beat === "combine" || stream || beat === "done" || kind === "result";
  const stopRelay = kind === "packet" ? relayOf(role) : -1;

  const nodes: SceneNode[] = [
    {
      id: "device", kind: "box", position: DEVICE, size: [1.7, 1.5, 1.7], tone: "good", flash: input.revealed,
      detail: "The only place where the question, both pieces together, and the answer exist in the clear.",
      active: beat === "split" || beat === "seal" || beat === "verify" || beat === "combine" || stream || beat === "done",
    },
    {
      id: "screen-device", kind: "screen", screen: "device", position: [DEVICE[0], 3.3, 0], size: [3.6, 2.25, 0], tone: "good", flash: input.revealed, label: "Your device",
      detail: "The output as it forms. It is decoded from the four recombined results.",
    },
    { id: "circuit", kind: "slab", position: [-2, 0.04, 0], size: [7.6, 0.08, 5], tone: "cipher", opacity: 0.14, label: "Tor circuit: three relays (stand-ins)" },
    { id: "zone", kind: "slab", position: [6.15, 0.04, 0], size: [7, 0.08, 7], tone: "accent", opacity: 0.14, label: "Four operators, each in a hardware enclave (simulated)" },
  ];
  const links: SceneLink[] = [];

  RELAYS.forEach((at, index) => {
    const stopped = stopRelay === index;
    nodes.push({
      id: `relay-${index}`, kind: "cylinder", position: at, size: [1.1, 1, 1.1], tone: stopped ? "warn" : "cipher",
      active: beat === "route" || stopped, pulse: routing || returning || stopped, flash: stopped ? input.rejections : 0,
      detail: "Removes one layer of encryption and learns only the next stop.",
    });
    nodes.push({
      id: `screen-relay-${index}`, kind: "screen", screen: `relay-${index}`, position: [at[0], 1.85, at[2]], size: [1.5, 0.94, 0], tone: stopped ? "warn" : "cipher", active: stopped,
      label: stopped ? `Relay ${index + 1}: packet refused` : `Relay ${index + 1}`,
      detail: "What this relay sees: sealed bytes it cannot read.",
    });
    const from = index === 0 ? "device" : `relay-${index - 1}`;
    const cut = stopRelay >= 0 && index > stopRelay;
    links.push({ id: `hop-${index}`, from, to: `relay-${index}`, arc: 1.5, tone: stopped ? "warn" : "cipher", dashed: stopped, active: (routing && !cut) || stopped, flow: routing && !cut ? "forward" : undefined });
    links.push({ id: `back-${index}`, from: `relay-${index}`, to: from, arc: 0.35, tone: "cipher", active: returning, flow: returning ? "forward" : undefined });
  });

  for (let worker = 0; worker < 4; worker += 1) {
    const id = `worker-${worker}`;
    const at = workerAt(worker);
    const hit = beat === "rejected" && input.tamperRole === worker;
    const wrongResult = hit && kind === "result";
    const wrongSoftware = hit && kind === "software";
    nodes.push({
      id, kind: "box", position: at, size: [1.2, 1.1, 1.2], tone: wrongResult ? "warn" : "accent",
      detail: `Holds activation share ${worker >> 1} and weight share ${worker & 1}.`,
      active: beat === "compute" || wrongResult, pulse: beat === "compute" || stream || wrongResult, flash: wrongResult ? input.rejections : 0,
    });
    nodes.push({
      id: `enclave-${worker}`, kind: "shell", position: [at[0], 0.85, at[2]], size: [2.5, 2.5, 2.5], tone: wrongSoftware ? "warn" : "good",
      opacity: beat === "attest" || wrongSoftware ? 0.42 : 0.2, pulse: beat === "attest" || wrongSoftware,
      detail: wrongSoftware ? "This enclave's report does not match the expected software." : "A hardware enclave (simulated here): the operator of the machine cannot look inside.",
    });
    nodes.push({
      id: `screen-${id}`, kind: "screen", screen: id, position: [at[0], 2.6, at[2]], size: [1.3, 0.81, 0], tone: hit ? "warn" : "accent", active: hit,
      label: wrongResult ? `${workerName(worker)}: failed the check` : wrongSoftware ? `${workerName(worker)}: report refused` : workerName(worker),
      detail: `What ${workerName(worker)} sees: its pieces and its result, which are random numbers.`,
    });
    const cut = stopRelay >= 0 || wrongSoftware;
    links.push({ id: `fan-${worker}`, from: "relay-2", to: id, arc: 1.3 + worker * 0.2, tone: "cipher", active: routing && !cut, flow: routing && !cut ? "forward" : undefined });
    links.push({ id: `ret-${worker}`, from: id, to: "relay-2", arc: 0.3 + worker * 0.1, tone: wrongResult ? "warn" : "cipher", dashed: wrongResult, active: returning || wrongResult, flow: returning ? "forward" : undefined });
    if (beat === "attest" || wrongSoftware) {
      links.push({ id: `report-${worker}`, from: id, to: "device", arc: 4.2 + worker * 0.3, tone: wrongSoftware ? "warn" : "good", dashed: true, active: true, flow: wrongSoftware ? undefined : "forward" });
    }
    if (sealed) nodes.push({ id: `seal-${worker}`, kind: "ring", position: [at[0], 0.1, at[2]], size: [2.9, 0, 0], tone: wrongResult ? "warn" : "good", detail: wrongResult ? "Result failed its check." : "Result verified." });
  }

  if (beat === "split") {
    nodes.push({ id: "share-0", kind: "orb", position: [DEVICE[0] - 0.5, 1.9, 1], size: [0.34, 0.34, 0.34], tone: "accent", label: "piece 0", detail: "One random piece of the input." });
    nodes.push({ id: "share-1", kind: "orb", position: [DEVICE[0] + 0.5, 1.9, 1], size: [0.34, 0.34, 0.34], tone: "accent", label: "piece 1", detail: "The other random piece of the input." });
  }
  if (beat === "seal") {
    // A piece with three rings round it: one layer of encryption per relay.
    const at: Vec3 = [DEVICE[0] + 1.9, 1.5, 0.9];
    nodes.push({ id: "packet", kind: "orb", position: at, size: [0.4, 0.4, 0.4], tone: "cipher", label: "sealed three times", detail: "A piece inside three layers of encryption.", pulse: true });
    [0, 1, 2].forEach((layer) => nodes.push({ id: `layer-${layer}`, kind: "ring", position: [at[0], at[1] - 0.25 + layer * 0.25, at[2]], size: [0.75 + layer * 0.3, 0, 0], tone: "cipher" }));
  }
  return {
    caption: caption(input),
    nodes, links,
    legend: [
      { tone: "good", text: "readable: your device, verified results, checked enclaves" },
      { tone: "accent", text: "random pieces and the workers that hold them" },
      { tone: "cipher", text: "sealed: encrypted on the way through the relays" },
      { tone: "warn", text: "something that failed a check" },
    ],
  };
}

const WIDE: Direction = { theta: 0.28, phi: 1.06, zoom: 0.6, focusId: null, sway: 0.2 };

/** Where the camera goes for each beat while the reader is not steering it. */
export function showCamera(beat: Beat, tamperRole: number | null, rejectKind?: Attack | null): Direction {
  switch (beat) {
    case "split": return { theta: 0.1, phi: 1.15, zoom: 1, focusId: "device", sway: 0 };
    case "attest": return { ...WIDE, theta: 0.5, sway: 0 };
    case "seal": return { theta: 0.3, phi: 1.15, zoom: 1, focusId: "device", sway: 0 };
    case "route": return { theta: 0.1, phi: 1.02, zoom: 1.15, focusId: "relay-1", sway: 0 };
    case "compute": return { theta: 0.7, phi: 1.02, zoom: 1.25, focusId: "worker-1", sway: 0 };
    case "return": return { ...WIDE, theta: 0.2, sway: 0 };
    case "verify": return { ...WIDE, theta: 0.3, sway: 0 };
    case "combine": return { theta: 0.12, phi: 1.15, zoom: 1, focusId: "device", sway: 0 };
    case "rejected": return { theta: 0.6, phi: 1.05, zoom: 1.1, focusId: rejectKind === "packet" ? `relay-${relayOf(tamperRole ?? 0)}` : `worker-${tamperRole ?? 0}`, sway: 0 };
    case "done": return { ...WIDE, theta: 0.3, sway: 0.3 };
    case "stream": return WIDE;
  }
}
