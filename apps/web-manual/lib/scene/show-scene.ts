import type { Direction } from "@/lib/scene/orbit";
import type { Scene3D, SceneLink, SceneNode, Vec3 } from "@/lib/scene/types";
import type { Beat } from "@/lib/show/timeline";
import type { ShowKind } from "@/lib/simulation/show";

export interface ShowSceneInput {
  kind: ShowKind;
  beat: Beat;
  /** Units whose output is already on the device's screen. */
  revealed: number;
  units: number;
  rejections: number;
  /** Role (2·activation share + weight share) whose product was tampered with, while a rejection is on stage. */
  tamperRole: number | null;
  /** Everyday wording, or the terms the chapters use (the default). */
  words?: "plain" | "technical";
}

const THING: Record<ShowKind, string> = { text: "token", picture: "step of the picture", clip: "frame of the clip" };

export function workerName(role: number): string {
  return `Worker (${role >> 1},${role & 1})`;
}

function progress(input: ShowSceneInput): string {
  if (input.kind === "text") return `Token ${input.revealed} of ${input.units}`;
  if (input.kind === "picture") return `Step ${input.revealed} of ${input.units}`;
  return `Frame ${Math.ceil(input.revealed / 2)} of ${input.units / 2}`;
}

function plainCaption(input: ShowSceneInput): string {
  const thing = THING[input.kind];
  switch (input.beat) {
    case "split": return `Step 1 of 6 · Split. Your device cuts its data for the first ${thing} into two random pieces.`;
    case "send": return "Step 2 of 6 · Send. Each of the four workers gets one piece. It already holds one piece of the model.";
    case "compute": return "Step 3 of 6 · Compute. Each worker multiplies the two pieces it holds. Its screen shows all it sees: noise.";
    case "return": return "Step 4 of 6 · Return. The four answers travel back to your device.";
    case "verify": return "Step 5 of 6 · Check. Your device tests every answer. A wrong one would be thrown out here.";
    case "combine": return `Step 6 of 6 · Combine. The four checked answers add up to the first ${thing}, which appears only on your device.`;
    case "stream": return `${progress(input)}. The same six steps repeat for every ${thing}.`;
    case "rejected": return `${workerName(input.tamperRole ?? 0)} sent back a wrong answer. Your device caught it, threw the step away and asked again.`;
    case "done": return `Complete. Every ${thing} was computed on pieces, checked, and put together on your device.`;
  }
}

/** One line on why the step on stage matters for protection. */
export function showWhy(beat: Beat): string {
  switch (beat) {
    case "split": return "One piece alone is random noise. It says nothing about your data.";
    case "send": return "No worker ever receives both pieces.";
    case "compute": return "The heavy work is done by machines that cannot read what they are working on.";
    case "return": return "Each answer is only one of four parts, and useless on its own.";
    case "verify": return "A worker that cheats or makes a mistake is caught before its answer is used.";
    case "combine": return "The readable result exists in one place only: your device.";
    case "stream": return "This holds while workers with different pieces do not pool them. “Whose eyes?” shows what each party gets.";
    case "rejected": return "Every answer is checked, so a single wrong number cannot slip into the result.";
    case "done": return "Now try your own text or drawing, or look through another party’s eyes.";
  }
}

function caption(input: ShowSceneInput): string {
  if (input.words === "plain") return plainCaption(input);
  const thing = THING[input.kind];
  switch (input.beat) {
    case "split": return `Split: your device splits its input for the first ${thing} into two random shares.`;
    case "send": return "Send: each of the four workers receives one share. It already holds one share of the weights.";
    case "compute": return "Compute: each worker multiplies the two shares it holds. Its screen shows what it sees: random numbers.";
    case "return": return "Return: the four products travel back to your device.";
    case "verify": return "Verify: your device tests every product against a random challenge.";
    case "combine": return `Combine: the four verified products add up to the first ${thing}, which appears only on your device.`;
    case "stream": return `${progress(input)}. The same six steps repeat for every ${thing}.`;
    case "rejected": return `${workerName(input.tamperRole ?? 0)} returned a product that fails the check. The step is rejected and sent again.`;
    case "done": return `Complete. Every ${thing} was computed on shares, checked and combined on your device.`;
  }
}

/** The stage for the live demonstration: the architecture overview with a screen on every machine. */
export function showScene(input: ShowSceneInput): Scene3D {
  const { beat } = input;
  const sending = beat === "send" || beat === "stream";
  const returning = beat === "return" || beat === "stream";
  const sealed = beat === "verify" || beat === "combine" || beat === "stream" || beat === "done" || beat === "rejected";
  const nodes: SceneNode[] = [
    {
      id: "device", kind: "box", position: [-4.6, 0.75, 0], size: [1.7, 1.5, 1.7], tone: "good", flash: input.revealed,
      detail: "The only place where the input, both shares together, and the output exist in the clear.",
      active: beat === "split" || beat === "verify" || beat === "combine" || beat === "stream" || beat === "done",
    },
    {
      id: "screen-device", kind: "screen", screen: "device", position: [-4.6, 3.3, 0], size: [3.6, 2.25, 0], tone: "good", flash: input.revealed, label: "Your device",
      detail: "The output as it forms. It is decoded from the four recombined products.",
    },
    { id: "controller", kind: "cylinder", position: [0, 0.35, -4.6], size: [1.7, 0.7, 1.7], tone: "plain", label: "Controller: sees ciphertext only", detail: "A signed directory of workers. It routes encrypted messages and cannot read them." },
    { id: "zone", kind: "slab", position: [4.4, 0.04, 0], size: [6.6, 0.08, 6.6], tone: "accent", opacity: 0.16 },
  ];
  const links: SceneLink[] = [{ id: "lookup", from: "device", to: "controller", tone: "plain", dashed: true, arc: 0.6 }];
  for (let role = 0; role < 4; role += 1) {
    const id = `worker-${role}`;
    const at: Vec3 = [2.9 + (role >> 1) * 3, 0.55, -1.5 + (role & 1) * 3];
    const bad = beat === "rejected" && input.tamperRole === role;
    nodes.push({
      id, kind: "box", position: at, size: [1.2, 1.1, 1.2], tone: bad ? "warn" : "accent",
      detail: `Holds activation share ${role >> 1} and weight share ${role & 1}.`,
      active: beat === "compute" || bad, pulse: beat === "compute" || beat === "stream" || bad, flash: bad ? input.rejections : 0,
    });
    nodes.push({
      id: `screen-${id}`, kind: "screen", screen: id, position: [at[0], 1.95, at[2]], size: [1.3, 0.81, 0], tone: bad ? "warn" : "accent", active: bad,
      label: bad ? `${workerName(role)}: failed the check` : workerName(role),
      detail: `What ${workerName(role)} sees: its shares and its product, which are random numbers.`,
    });
    links.push({ id: `out-${role}`, from: "device", to: id, arc: 2.3 + role * 0.25, tone: "accent", active: sending, flow: sending ? "forward" : undefined });
    links.push({ id: `back-${role}`, from: id, to: "device", arc: 0.7 + role * 0.14, tone: bad ? "warn" : "good", dashed: bad, active: returning || bad, flow: returning ? "forward" : undefined });
    if (sealed) nodes.push({ id: `seal-${role}`, kind: "ring", position: [at[0], 0.1, at[2]], size: [2.1, 0, 0], tone: bad ? "warn" : "good", detail: bad ? "Product failed its check." : "Product verified." });
  }
  if (beat === "split" || beat === "send") {
    nodes.push({ id: "share-0", kind: "orb", position: [-5.1, 1.85, 0.9], size: [0.34, 0.34, 0.34], tone: "accent", label: beat === "split" ? "share 0" : undefined, detail: "One additive share of the input." });
    nodes.push({ id: "share-1", kind: "orb", position: [-4.1, 1.85, 0.9], size: [0.34, 0.34, 0.34], tone: "accent", label: beat === "split" ? "share 1" : undefined, detail: "The other additive share of the input." });
  }
  return {
    caption: caption(input),
    nodes, links,
    legend: [
      { tone: "good", text: "your device and verified products" },
      { tone: "accent", text: "shares and the workers that hold them" },
      { tone: "warn", text: "a product that failed its check" },
      { tone: "plain", text: "controller (ciphertext only)" },
    ],
  };
}

const WIDE: Direction = { theta: 0.5, phi: 1.1, zoom: 0.66, focusId: null, sway: 0.22 };

/** Where the camera goes for each beat while the reader is not steering it. */
export function showCamera(beat: Beat, tamperRole: number | null): Direction {
  switch (beat) {
    case "split": return { theta: 0.2, phi: 1.15, zoom: 1, focusId: "device", sway: 0 };
    case "send": return { ...WIDE, sway: 0 };
    case "compute": return { ...WIDE, theta: 0.9, zoom: 0.6, sway: 0 };
    case "return": return { ...WIDE, theta: 0.4, sway: 0 };
    case "verify": return { ...WIDE, theta: 0.3, zoom: 0.62, sway: 0 };
    case "combine": return { theta: 0.15, phi: 1.15, zoom: 1, focusId: "device", sway: 0 };
    case "rejected": return { theta: 0.9, phi: 1.05, zoom: 1.1, focusId: `worker-${tamperRole ?? 0}`, sway: 0 };
    case "done": return { ...WIDE, theta: 0.3, sway: 0.3 };
    case "stream": return WIDE;
  }
}
