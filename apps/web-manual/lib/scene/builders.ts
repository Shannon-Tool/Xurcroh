import type { Scene3D, SceneLink, SceneNode, ScenePacket, SceneTone } from "@/lib/scene/types";

/** Bar standing on the floor; height follows |value| relative to the largest value shown. */
function bar(id: string, x: number, z: number, value: number, largest: number, tone: SceneTone, extra: Partial<SceneNode> = {}): SceneNode {
  const height = 0.08 + 2.2 * (largest > 0 ? Math.abs(value) / largest : 0);
  return { id, kind: "box", position: [x, height / 2, z], size: [0.6, height, 0.6], tone, ...extra };
}

function largestOf(values: number[]): number {
  return values.reduce((most, value) => Math.max(most, Math.abs(value)), 0);
}

function short(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(2);
}

// ---------------------------------------------------------------- share grid

export function shareGridScene(input: { roles: { activationShare: number; weightShare: number }[]; step: number; tampered: boolean }): Scene3D {
  const nodes: SceneNode[] = [];
  const links: SceneLink[] = [];
  const packets: ScenePacket[] = [];
  const finished = input.step >= input.roles.length;
  nodes.push({ id: "client", kind: "box", position: [0, 0.6, 0], size: [1.4, 1.2, 1.4], tone: "good", label: finished ? "Client: combined" : "Client", active: finished });
  input.roles.forEach((role, index) => {
    const bad = input.tampered && index === 0;
    const current = index === input.step;
    const done = index < input.step;
    const id = `role-${index}`;
    nodes.push({
      id,
      kind: "box",
      position: [(role.activationShare * 2 - 1) * 3, 0.5, (role.weightShare * 2 - 1) * 3],
      size: [1.1, 1, 1.1],
      tone: bad ? "warn" : done || finished ? "accent" : "plain",
      label: `Worker (${role.activationShare},${role.weightShare})${bad ? " tampered" : ""}`,
      active: current,
    });
    links.push({ id: `link-${index}`, from: "client", to: id, tone: bad ? "warn" : "accent", dashed: bad, active: current });
    if (current) packets.push({ id: `packet-${index}`, link: `link-${index}`, progress: 0.5, tone: bad ? "warn" : "accent" });
  });
  const caption = finished
    ? (input.tampered ? "The client refuses to combine: the tampered worker's result failed verification." : "All four products verified; the client combines them into the output.")
    : `Worker ${input.step + 1} of ${input.roles.length} multiplies its activation share by its weight share and returns the product.`;
  return { caption, nodes, links, packets };
}

// ---------------------------------------------------------------- Freivalds

export function freivaldsScene(input: { x: number[]; product: number[]; challenge: number[]; expected: number[]; valid: boolean }): Scene3D {
  const largest = largestOf([...input.x, ...input.product, ...input.challenge]);
  const nodes: SceneNode[] = [];
  input.x.forEach((value, index) => nodes.push(bar(`x-${index}`, index, 0, value, largest, "plain", { label: `x${index} = ${short(value)}` })));
  input.challenge.forEach((value, index) => nodes.push(bar(`r-${index}`, index, 1.6, value, largest, "plain", { label: `r${index} = ${short(value)}` })));
  input.product.forEach((value, index) => {
    const wrong = value !== input.expected[index];
    nodes.push(bar(`p-${index}`, index, 3.2, value, largest, wrong ? "warn" : "accent", { label: `claimed ${short(value)}`, active: wrong }));
  });
  const tone: SceneTone = input.valid ? "good" : "warn";
  nodes.push({ id: "left", kind: "sphere", position: [3.4, 0.6, 0.8], size: [0.9, 0.9, 0.9], tone, label: "x·(Wᵀr)" });
  nodes.push({ id: "right", kind: "sphere", position: [3.4, 0.6, 2.6], size: [0.9, 0.9, 0.9], tone, label: "(claimed)·r", active: !input.valid });
  return {
    caption: input.valid ? "Both sides of the check agree, so the claimed product is accepted." : "The two sides of the check differ, so the claimed product is rejected.",
    nodes,
    links: [{ id: "check", from: "left", to: "right", tone, dashed: !input.valid, active: input.valid }],
    packets: [],
  };
}

// ---------------------------------------------------------------- quantized formats

export function quantizedScene(input: { stages: { title: string; values: number[] }[]; step: number }): Scene3D {
  const nodes: SceneNode[] = [];
  input.stages.forEach((stage, row) => {
    const largest = largestOf(stage.values);
    stage.values.forEach((value, column) => {
      nodes.push(bar(`q-${row}-${column}`, column, row * 1.3, value, largest, row === input.step ? "accent" : "plain", {
        active: row === input.step,
        label: column === 0 ? stage.title : undefined,
      }));
    });
  });
  const current = input.stages[input.step];
  return { caption: current ? `${current.title}: ${current.values.map(short).join(", ")}. Each row is scaled to its own largest value.` : "No values to show.", nodes, links: [], packets: [] };
}

// ---------------------------------------------------------------- placement

const PROVIDER_TONES: SceneTone[] = ["accent", "good", "warn", "plain"];

export function placementScene(input: { roles: { operation: string; activationShare: number; weightShare: number; provider: string; country: string }[] }): Scene3D {
  const providers = [...new Set(input.roles.map((role) => role.provider))].sort();
  const operations = [...new Set(input.roles.map((role) => role.operation))];
  const nodes = input.roles.map((role, index): SceneNode => {
    const provider = providers.indexOf(role.provider);
    return {
      id: `place-${index}`,
      kind: "box",
      position: [(role.activationShare * 2 + role.weightShare) * 1.5, 0.3 + provider * 0.15, operations.indexOf(role.operation) * 1.5],
      size: [1.1, 0.6 + provider * 0.3, 1.1],
      tone: PROVIDER_TONES[provider % PROVIDER_TONES.length]!,
      label: `${role.provider} · ${role.country}`,
    };
  });
  return {
    caption: `${operations.length} operations (rows) across 4 roles (columns); colour and height show which of ${providers.length} providers runs each one.`,
    nodes,
    links: [],
    packets: [],
  };
}

// ---------------------------------------------------------------- conjugation

function tip(vector: number[], scale: number): [number, number, number] {
  return [(vector[0] ?? 0) * scale, (vector[2] ?? 0) * scale + 0.1, (vector[1] ?? 0) * scale];
}

export function conjugationScene(input: { x: number[]; rotatedInput: number[]; expected: number[]; transformedOutput: number[]; recovered: number[]; step: number }): Scene3D {
  const norm = (vector: number[]) => Math.sqrt(vector.slice(0, 3).reduce((sum, value) => sum + value * value, 0));
  const inputNorm = Math.max(norm(input.x), norm(input.rotatedInput), 1e-9);
  const outputNorm = Math.max(norm(input.expected), norm(input.transformedOutput), norm(input.recovered), 1e-9);
  const inScale = 2 / inputNorm;
  const outScale = 3 / outputNorm;
  const vectors: { id: string; label: string; at: [number, number, number]; tone: SceneTone; steps: number[] }[] = [
    { id: "x", label: "input x", at: tip(input.x, inScale), tone: "plain", steps: [0] },
    { id: "rx", label: "rotated input", at: tip(input.rotatedInput, inScale), tone: "accent", steps: [1] },
    { id: "ty", label: "conjugated output", at: tip(input.transformedOutput, outScale), tone: "accent", steps: [2] },
    { id: "y", label: "original output", at: tip(input.expected, outScale), tone: "good", steps: [0, 3] },
    { id: "ry", label: "recovered output", at: [tip(input.recovered, outScale)[0], tip(input.recovered, outScale)[1] + 0.25, tip(input.recovered, outScale)[2]], tone: "good", steps: [3] },
  ];
  const nodes: SceneNode[] = [
    { id: "origin", kind: "sphere", position: [0, 0.1, 0], size: [0.25, 0.25, 0.25], tone: "plain" },
    { id: "norm", kind: "shell", position: [0, 0.1, 0], size: [4, 4, 4], tone: "plain", label: "same length", active: input.step === 1 },
    ...vectors.map((vector): SceneNode => ({ id: vector.id, kind: "sphere", position: vector.at, size: [0.34, 0.34, 0.34], tone: vector.tone, label: vector.label, active: vector.steps.includes(input.step) })),
  ];
  const captions = [
    "The original computation: input x and its output.",
    "The input is rotated by the secret basis. It stays on the same sphere, so its length still leaks.",
    "The worker computes with the conjugated weight and returns a rotated output.",
    "Rotating back recovers the original output exactly.",
  ];
  return {
    caption: `${captions[input.step] ?? captions[0]!} Only the first three coordinates are drawn.`,
    nodes,
    links: vectors.map((vector) => ({ id: `v-${vector.id}`, from: "origin", to: vector.id, tone: vector.tone, active: vector.steps.includes(input.step) })),
    packets: [],
  };
}

// ---------------------------------------------------------------- diffusion

export function diffusionScene(input: { stages: { title: string; points: number[][] }[]; step: number }): Scene3D {
  const nodes: SceneNode[] = [];
  const links: SceneLink[] = [];
  input.stages.forEach((stage, layer) => {
    stage.points.forEach((point, index) => {
      nodes.push({
        id: `d-${layer}-${index}`,
        kind: "sphere",
        position: [(point[0] ?? 0) * 2.5, layer * 1.4 + 0.2, (point[1] ?? 0) * 2.5],
        size: [0.36, 0.36, 0.36],
        tone: layer === 0 || layer === input.stages.length - 1 ? "good" : "accent",
        active: layer === input.step,
        label: index === 0 ? stage.title : undefined,
      });
      if (layer > 0) links.push({ id: `dl-${layer}-${index}`, from: `d-${layer - 1}-${index}`, to: `d-${layer}-${index}`, tone: "plain", dashed: true });
    });
  });
  return {
    caption: `${input.stages[input.step]?.title ?? "Latent"}: four latent points; each level up is the next stage, and only the first and last levels are in the client's own coordinates.`,
    nodes,
    links,
    packets: [],
  };
}

// ---------------------------------------------------------------- FWHT

export function fwhtScene(input: { stages: number[][]; step: number }): Scene3D {
  const nodes: SceneNode[] = [];
  const links: SceneLink[] = [];
  const width = input.stages[0]?.length ?? 0;
  const spread = width > 8 ? 0.75 : 1;
  input.stages.forEach((values, stage) => {
    const largest = largestOf(values);
    values.forEach((value, index) => {
      nodes.push(bar(`f-${stage}-${index}`, index * spread, stage * 2, value, largest, value < 0 ? "warn" : stage === input.step ? "accent" : "plain", {
        active: stage === input.step,
        label: index === 0 ? `Stage ${stage}` : undefined,
        size: [0.6 * spread, 0.08 + 2.2 * (largest > 0 ? Math.abs(value) / largest : 0), 0.6 * spread],
      }));
    });
  });
  if (input.step > 0 && input.stages[input.step]) {
    const block = 2 ** (input.step - 1);
    for (let index = 0; index < width; index += 1) {
      const partner = index ^ block;
      links.push({ id: `fs-${index}`, from: `f-${input.step - 1}-${index}`, to: `f-${input.step}-${index}`, tone: "plain" });
      links.push({ id: `fx-${index}`, from: `f-${input.step - 1}-${partner}`, to: `f-${input.step}-${index}`, tone: "accent", dashed: true });
    }
  }
  return {
    caption: input.stages.length === 0
      ? "Enter a valid vector to draw the transform."
      : input.step === 0
        ? "Stage 0 is your input vector. Orange bars are negative values."
        : `Stage ${input.step}: every entry is the sum or difference of two entries ${2 ** (input.step - 1)} apart in the previous stage.`,
    nodes,
    links,
    packets: [],
  };
}

// ---------------------------------------------------------------- pipeline

export function pipelineScene(input: { nodes: number; failNode: number | null; event: { request: number; state: string; stageIndex: number } }): Scene3D {
  const nodes: SceneNode[] = [{ id: "client", kind: "box", position: [-2.4, 0.6, 0], size: [1.2, 1.2, 1.2], tone: "good", label: "Client", active: input.event.stageIndex < 0 }];
  const links: SceneLink[] = [];
  for (let index = 0; index < input.nodes; index += 1) {
    const failed = input.failNode === index;
    nodes.push({ id: `w-${index}`, kind: "box", position: [index * 2, 0.5, 0], size: [1, 1, 1], tone: failed ? "warn" : "accent", label: `Worker ${index}${failed ? " (fails)" : ""}`, active: input.event.stageIndex === index });
    links.push({ id: `hop-${index}`, from: index === 0 ? "client" : `w-${index - 1}`, to: `w-${index}`, tone: "plain", active: input.event.stageIndex === index });
  }
  links.push({ id: "return", from: `w-${input.nodes - 1}`, to: "client", tone: "good", dashed: true });
  const trouble = input.event.state === "resent" || input.event.state === "recovered";
  const atClient = input.event.stageIndex < 0;
  return {
    caption: `Request ${input.event.request} is ${input.event.state} at ${atClient ? "the client" : `worker ${input.event.stageIndex}`}.`,
    nodes,
    links,
    packets: [{ id: "packet", link: atClient ? "return" : `hop-${input.event.stageIndex}`, progress: atClient ? 1 : 0.6, tone: trouble ? "warn" : "accent" }],
  };
}

// ---------------------------------------------------------------- onion

export function onionScene(input: { hops: string[]; step: number; tampered: boolean; compromised: boolean; colluding: boolean }): Scene3D {
  const count = input.hops.length;
  const nodes: SceneNode[] = [{ id: "sender", kind: "box", position: [-2.5, 0.5, 0], size: [1, 1, 1], tone: "good", label: "Sender" }];
  const links: SceneLink[] = [];
  for (let index = 0; index < count; index += 1) {
    const current = index === input.step;
    const exposed = input.colluding || (input.compromised && current);
    nodes.push({ id: `hop-${index}`, kind: "box", position: [index * 2.5, 0.5, 0], size: [1, 1, 1], tone: exposed ? "warn" : "plain", label: `Hop ${index} → ${input.hops[index]}${exposed ? " (compromised)" : ""}`, active: current });
    links.push({ id: `route-${index}`, from: index === 0 ? "sender" : `hop-${index - 1}`, to: `hop-${index}`, tone: input.colluding && index > 0 ? "warn" : "plain", dashed: input.colluding && index > 0, active: current });
  }
  const at = Math.min(input.step, count - 1);
  const remaining = Math.max(1, count - at);
  for (let layer = 0; layer < remaining; layer += 1) {
    const outer = layer === 0;
    nodes.push({ id: `shell-${layer}`, kind: "shell", position: [at * 2.5, 2.4, 0], size: [0.7 + (remaining - layer) * 0.5, 0, 0], tone: outer && input.tampered ? "warn" : "accent", active: outer, label: outer ? `${remaining} layer${remaining === 1 ? "" : "s"} left${input.tampered ? ", tampered" : ""}` : undefined });
  }
  nodes.push({ id: "payload", kind: "sphere", position: [at * 2.5, 2.4, 0], size: [0.45, 0.45, 0.45], tone: "good" });
  const risk = input.colluding ? " Colluding hops can link the two ends of the route." : input.compromised ? " A compromised hop sees only its own layer and the next hop." : "";
  return {
    caption: `At hop ${at}, ${remaining} layer${remaining === 1 ? "" : "s"} still wrap${remaining === 1 ? "s" : ""} the payload; this hop removes one.${input.tampered ? " The ciphertext was altered, so this hop's check fails." : ""}${risk}`,
    nodes,
    links,
    packets: [],
  };
}

// ---------------------------------------------------------------- mixture of experts

export function moeScene(input: { tokens: number; plan: { tokenIndices: number[][]; dummy: boolean[][] } | null; loads: number[]; protectedMode: boolean; step: number }): Scene3D {
  if (!input.plan) return { caption: "Nothing can be dispatched: an expert would receive more tokens than it has slots.", nodes: [], links: [], packets: [] };
  const nodes: SceneNode[] = [];
  const links: SceneLink[] = [];
  const largest = largestOf(input.loads);
  const experts = input.loads.length;
  const offset = ((input.tokens - 1) * 1.1 - (experts - 1) * 2) / 2;
  for (let token = 0; token < input.tokens; token += 1) {
    nodes.push({ id: `t-${token}`, kind: "sphere", position: [token * 1.1, 4, 0], size: [0.4, 0.4, 0.4], tone: "good", label: token === 0 ? "Tokens" : undefined });
  }
  let real = 0;
  let dummy = 0;
  input.loads.forEach((load, expert) => {
    const active = expert === input.step;
    nodes.push(bar(`e-${expert}`, expert * 2 + offset, 0, load, largest, "accent", { label: `Expert ${expert}: ${load} slots`, active, size: [1, 0.08 + 2.2 * (largest > 0 ? load / largest : 0), 1] }));
    input.plan!.tokenIndices[expert]!.forEach((token, slot) => {
      const isDummy = input.plan!.dummy[expert]![slot]!;
      if (isDummy) dummy += 1;
      else real += 1;
      if (isDummy && !input.protectedMode) return;
      links.push({ id: `r-${expert}-${slot}`, from: `t-${token}`, to: `e-${expert}`, tone: isDummy ? "plain" : "accent", dashed: isDummy, active: active && !isDummy });
    });
  });
  return {
    caption: input.protectedMode
      ? `${real} real routes (solid) and ${dummy} dummy routes (dashed) give every expert the same load.`
      : `${real} real routes only: the bar heights reveal which experts the prompt used.`,
    nodes,
    links,
    packets: [],
  };
}

// ---------------------------------------------------------------- memory

export function memoryScene(input: { steps: { operation: string; bytes: number[] }[]; index: number }): Scene3D {
  const nodes: SceneNode[] = [];
  input.steps.forEach((step, column) => {
    const afterWipe = step.operation === "zero" || step.operation === "reuse";
    step.bytes.forEach((value, cell) => {
      nodes.push(bar(`m-${column}-${cell}`, column * 1.5, cell * 0.8, value, 255, value === 0 ? "good" : afterWipe ? "warn" : "accent", {
        active: column === input.index,
        label: cell === 0 ? step.operation : undefined,
      }));
    });
  });
  const current = input.steps[input.index];
  const stale = current ? current.bytes.some((value) => value !== 0) && (current.operation === "zero" || current.operation === "reuse") : false;
  return {
    caption: current ? `${current.operation}: ${stale ? "the secret bytes are still there and the next owner can read them." : current.bytes.every((value) => value === 0) ? "all four bytes read as zero." : "the buffer holds the secret bytes."}` : "No step selected.",
    nodes,
    links: [],
    packets: [],
  };
}
