import type { Scene3D, SceneLink, SceneNode, SceneTone, Vec3 } from "@/lib/scene/types";

/** Bar standing on the floor; height follows |value| relative to the largest value shown. */
function bar(id: string, x: number, z: number, value: number, largest: number, tone: SceneTone, extra: Partial<SceneNode> = {}): SceneNode {
  const height = 0.08 + 2.2 * (largest > 0 ? Math.abs(value) / largest : 0);
  const width = extra.size?.[0] ?? 0.6;
  return { id, kind: "box", tone, ...extra, position: [x, (extra.position?.[1] ?? 0) + height / 2, z], size: [width, height, width] };
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
  const finished = input.step >= input.roles.length;
  const refused = finished && input.tampered;
  nodes.push({
    id: "client", kind: "box", position: [0, 0.65, 0], size: [1.5, 1.3, 1.5], tone: refused ? "warn" : "good", active: finished,
    label: finished ? (refused ? "Client: refuses" : "Client: combined") : "Client",
    detail: "Splits the activation into two additive shares, checks every product that comes back, and only then adds them up.",
  });
  input.roles.forEach((role, index) => {
    const bad = input.tampered && index === 0;
    const current = index === input.step;
    const done = index < input.step || finished;
    const id = `role-${index}`;
    const at: Vec3 = [(role.activationShare * 2 - 1) * 3.4, 0.5, (role.weightShare * 2 - 1) * 3.4];
    nodes.push({
      id, kind: "box", position: at, size: [1.15, 1, 1.15],
      tone: bad ? "warn" : done ? "accent" : "plain",
      label: `Worker (${role.activationShare},${role.weightShare})${bad ? " tampered" : ""}`,
      detail: `Holds activation share ${role.activationShare} and weight share ${role.weightShare}, and multiplies them. It never sees the other share of either.`,
      active: current, pulse: bad && (done || current),
    });
    links.push({
      id: `link-${index}`, from: "client", to: id, arc: 1.5,
      tone: bad ? "warn" : done ? "good" : "accent",
      dashed: bad, active: current,
      flow: bad && done ? undefined : current ? "forward" : done ? "back" : undefined,
    });
    if (done && !bad) {
      nodes.push({ id: `seal-${index}`, kind: "ring", position: [at[0], 0.06, at[2]], size: [2.1, 0, 0], tone: "good", detail: "This worker's product passed verification." });
    }
  });
  if (finished && !refused) {
    nodes.push({ id: "output", kind: "orb", position: [0, 2.3, 0], size: [0.55, 0.55, 0.55], tone: "good", label: "xWᵀ", detail: "The sum of the four verified products is the true output.", pulse: true });
  }
  const caption = finished
    ? (input.tampered ? "The client refuses to combine: the tampered worker's result failed verification." : "All four products verified; the client combines them into the output.")
    : `Worker ${input.step + 1} of ${input.roles.length} multiplies its activation share by its weight share and returns the product.`;
  return {
    caption, nodes, links,
    legend: [{ tone: "accent", text: "shares going out" }, { tone: "good", text: "verified product coming back" }, { tone: "warn", text: "tampered" }],
  };
}

// ---------------------------------------------------------------- Freivalds

export function freivaldsScene(input: { x: number[]; product: number[]; challenge: number[]; expected: number[]; valid: boolean }): Scene3D {
  const largest = largestOf([...input.x, ...input.product, ...input.challenge]);
  const nodes: SceneNode[] = [];
  input.x.forEach((value, index) => nodes.push(bar(`x-${index}`, index, 0, value, largest, "plain", { label: `x${index} = ${short(value)}`, detail: "Input entry." })));
  input.challenge.forEach((value, index) => nodes.push(bar(`r-${index}`, index, 1.5, value, largest, "accent", { label: `r${index} = ${short(value)}`, detail: "Secret random challenge chosen by the verifier." })));
  input.product.forEach((value, index) => {
    const wrong = value !== input.expected[index];
    nodes.push(bar(`p-${index}`, index, 3, value, largest, wrong ? "warn" : "good", { label: `claimed ${short(value)}`, detail: wrong ? `The worker claimed ${short(value)}; the true entry is ${short(input.expected[index] ?? 0)}.` : "Product entry claimed by the worker.", active: wrong, pulse: wrong }));
  });
  const dot = (values: number[]) => values.reduce((sum, value, index) => sum + value * (input.challenge[index] ?? 0), 0);
  const left = dot(input.expected);
  const right = dot(input.product);
  // A level beam means the two sides agree. When they differ it tips towards the heavier side.
  const tilt = input.valid ? 0 : (right > left ? -1 : 1) * 0.3;
  const tone: SceneTone = input.valid ? "good" : "warn";
  const middle = (Math.max(input.x.length, 1) - 1) / 2;
  const arm = 1.7;
  const pan = (side: -1 | 1): Vec3 => [middle + side * arm * Math.cos(tilt), 1.95 + side * arm * Math.sin(tilt) + 0.5, -2.2];
  nodes.push({ id: "pivot", kind: "cylinder", position: [middle, 0.95, -2.2], size: [0.3, 1.9, 0.3], tone: "plain" });
  nodes.push({ id: "beam", kind: "box", position: [middle, 1.95, -2.2], size: [arm * 2 + 0.5, 0.12, 0.45], tone, tilt, active: !input.valid, label: input.valid ? "balanced: accept" : "unbalanced: reject" });
  nodes.push({ id: "left", kind: "sphere", position: pan(-1), size: [0.85, 0.85, 0.85], tone, label: "x·(Wᵀr)", detail: `What the verifier computes itself: ${short(left)}.` });
  nodes.push({ id: "right", kind: "sphere", position: pan(1), size: [0.85, 0.85, 0.85], tone, label: "(claimed)·r", detail: `What the claimed product gives: ${short(right)}.`, pulse: !input.valid });
  return {
    caption: input.valid ? "Both sides of the check agree, so the claimed product is accepted." : "The two sides of the check differ, so the claimed product is rejected.",
    nodes,
    links: [],
    legend: [{ tone: "plain", text: "input" }, { tone: "accent", text: "random challenge" }, { tone: "good", text: "agrees" }, { tone: "warn", text: "wrong" }],
  };
}

// ---------------------------------------------------------------- quantized formats

export function quantizedScene(input: { stages: { title: string; values: number[] }[]; step: number }): Scene3D {
  const nodes: SceneNode[] = [];
  const links: SceneLink[] = [];
  input.stages.forEach((stage, row) => {
    const current = row === input.step;
    const largest = largestOf(stage.values);
    const width = Math.max(stage.values.length, 1);
    nodes.push({ id: `stage-${row}`, kind: "slab", position: [(width - 1) / 2, 0.05, row * 1.6], size: [width + 0.5, 0.1, 1.15], tone: current ? "accent" : "muted", opacity: current ? 0.55 : 0.3, label: stage.title });
    stage.values.forEach((value, column) => {
      nodes.push(bar(`q-${row}-${column}`, column, row * 1.6, value, largest, current ? "accent" : value < 0 ? "warn" : "plain", { position: [0, 0.1, 0], active: current, detail: `${stage.title}: ${short(value)}` }));
    });
    if (row > 0) links.push({ id: `flow-${row}`, from: `stage-${row - 1}`, to: `stage-${row}`, tone: row === input.step ? "accent" : "plain", arc: 1.2, active: row === input.step, flow: row === input.step ? "forward" : undefined });
  });
  const current = input.stages[input.step];
  return {
    caption: current ? `${current.title}: ${current.values.map(short).join(", ")}. Each row is scaled to its own largest value.` : "No values to show.",
    nodes, links,
    legend: [{ tone: "accent", text: "current stage" }, { tone: "warn", text: "negative value" }],
  };
}

// ---------------------------------------------------------------- placement

const OPERATION_TONES: SceneTone[] = ["accent", "good", "plain", "muted"];

export function placementScene(input: { roles: { operation: string; activationShare: number; weightShare: number; provider: string; country: string }[] }): Scene3D {
  const providers = [...new Set(input.roles.map((role) => role.provider))].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  const countries = [...new Set(input.roles.map((role) => role.country))].sort();
  const operations = [...new Set(input.roles.map((role) => role.operation))];
  // Each provider owns one pad on its country's platform; the pad's place never depends on what runs there.
  const padOf = new Map<string, Vec3>();
  const perCountry = new Map<string, number>();
  for (const provider of providers) {
    const country = input.roles.find((role) => role.provider === provider)!.country;
    const slot = perCountry.get(country) ?? 0;
    perCountry.set(country, slot + 1);
    padOf.set(provider, [countries.indexOf(country) * 5 + (slot % 2) * 1.5, 0, Math.floor(slot / 2) * 1.5]);
  }
  const nodes: SceneNode[] = countries.map((country, index): SceneNode => {
    const rows = Math.ceil((perCountry.get(country) ?? 1) / 2);
    return { id: `country-${country}`, kind: "slab", position: [index * 5 + 0.75, 0.05, (rows - 1) * 0.75], size: [3.4, 0.1, rows * 1.5 + 0.4], tone: "muted", opacity: 0.4, label: `Jurisdiction ${country.toUpperCase()}` };
  });
  input.roles.forEach((role, index) => {
    const pad = padOf.get(role.provider)!;
    const operation = operations.indexOf(role.operation);
    nodes.push({
      id: `place-${index}`, kind: "box", position: [pad[0], 0.1 + 0.45, pad[2]], size: [1.05, 0.9, 1.05],
      tone: OPERATION_TONES[operation % OPERATION_TONES.length]!,
      label: `${role.operation} (${role.activationShare},${role.weightShare})`,
      detail: `Runs at ${role.provider} in ${role.country.toUpperCase()}: activation share ${role.activationShare}, weight share ${role.weightShare}.`,
    });
  });
  return {
    caption: `${operations.length} operations × 4 roles placed on ${providers.length} providers in ${countries.length} jurisdictions. No provider holds two roles of the same operation; reshuffling moves every block.`,
    nodes,
    links: [],
    legend: operations.map((operation, index) => ({ tone: OPERATION_TONES[index % OPERATION_TONES.length]!, text: operation })),
  };
}

// ---------------------------------------------------------------- conjugation

function tip(vector: number[], scale: number): Vec3 {
  return [(vector[0] ?? 0) * scale, (vector[2] ?? 0) * scale + 0.1, (vector[1] ?? 0) * scale];
}

export function conjugationScene(input: { x: number[]; rotatedInput: number[]; expected: number[]; transformedOutput: number[]; recovered: number[]; step: number }): Scene3D {
  const norm = (vector: number[]) => Math.sqrt(vector.slice(0, 3).reduce((sum, value) => sum + value * value, 0));
  const inputNorm = Math.max(norm(input.x), norm(input.rotatedInput), 1e-9);
  const outputNorm = Math.max(norm(input.expected), norm(input.transformedOutput), norm(input.recovered), 1e-9);
  const inScale = 2 / inputNorm;
  const outScale = 3.2 / outputNorm;
  const origin: Vec3 = [0, 0.1, 0];
  const recovered = tip(input.recovered, outScale);
  const vectors: { id: string; label: string; detail: string; at: Vec3; tone: SceneTone; steps: number[] }[] = [
    { id: "x", label: "input x", detail: "The client's real input.", at: tip(input.x, inScale), tone: "plain", steps: [0] },
    { id: "rx", label: "rotated input", detail: "What the worker receives. Same length as x, different direction.", at: tip(input.rotatedInput, inScale), tone: "accent", steps: [1] },
    { id: "ty", label: "conjugated output", detail: "What the worker returns, still in the rotated basis.", at: tip(input.transformedOutput, outScale), tone: "accent", steps: [2] },
    { id: "y", label: "original output", detail: "The output of the unprotected computation, for comparison.", at: tip(input.expected, outScale), tone: "good", steps: [0, 3] },
    { id: "ry", label: "recovered output", detail: "The returned output rotated back by the client.", at: [recovered[0], recovered[1] + 0.25, recovered[2]], tone: "good", steps: [3] },
  ];
  const nodes: SceneNode[] = [
    { id: "origin", kind: "sphere", position: origin, size: [0.22, 0.22, 0.22], tone: "plain" },
    { id: "norm", kind: "shell", position: origin, size: [4, 4, 4], tone: "accent", label: "same length", detail: "A rotation cannot change length, so the input's length still leaks to the worker.", active: input.step === 1, opacity: input.step === 1 ? 0.3 : 0.14 },
  ];
  for (const vector of vectors) {
    const active = vector.steps.includes(input.step);
    nodes.push({ id: `arrow-${vector.id}`, kind: "arrow", position: origin, tip: vector.at, size: [active ? 0.09 : 0.055, 0, 0], tone: vector.tone, active, opacity: active ? 1 : 0.55 });
    nodes.push({ id: vector.id, kind: "orb", position: vector.at, size: [0.2, 0.2, 0.2], tone: vector.tone, label: vector.label, detail: vector.detail, active });
  }
  const links: SceneLink[] = [
    { id: "rotate", from: "x", to: "rx", tone: "accent", arc: 0.9, dashed: true, active: input.step === 1, flow: input.step === 1 ? "forward" : undefined },
    { id: "unrotate", from: "ty", to: "ry", tone: "good", arc: 1.1, dashed: true, active: input.step === 3, flow: input.step === 3 ? "forward" : undefined },
  ];
  const captions = [
    "The original computation: input x and its output.",
    "The input is rotated by the secret basis. It stays on the same sphere, so its length still leaks.",
    "The worker computes with the conjugated weight and returns a rotated output.",
    "Rotating back recovers the original output exactly.",
  ];
  return {
    caption: `${captions[input.step] ?? captions[0]!} Only the first three coordinates are drawn.`,
    nodes, links,
    legend: [{ tone: "plain", text: "client's own vector" }, { tone: "accent", text: "what the worker sees" }, { tone: "good", text: "true and recovered output" }],
  };
}

// ---------------------------------------------------------------- diffusion

export function diffusionScene(input: { stages: { title: string; points: number[][] }[]; step: number }): Scene3D {
  const nodes: SceneNode[] = [];
  const links: SceneLink[] = [];
  const last = input.stages.length - 1;
  input.stages.forEach((stage, layer) => {
    const own = layer === 0 || layer === last;
    const current = layer === input.step;
    nodes.push({ id: `level-${layer}`, kind: "slab", position: [1.25, layer * 1.5, 1.25], size: [6.4, 0.05, 6.4], tone: own ? "good" : "accent", opacity: current ? 0.34 : 0.12, label: stage.title });
    stage.points.forEach((point, index) => {
      nodes.push({
        id: `d-${layer}-${index}`, kind: current ? "orb" : "sphere",
        position: [(point[0] ?? 0) * 2.5, layer * 1.5 + 0.24, (point[1] ?? 0) * 2.5],
        size: [0.36, 0.36, 0.36], tone: own ? "good" : "accent", active: current,
        detail: `Latent point ${index} at "${stage.title}": (${short(point[0] ?? 0)}, ${short(point[1] ?? 0)}).`,
      });
      if (layer > 0) links.push({ id: `dl-${layer}-${index}`, from: `d-${layer - 1}-${index}`, to: `d-${layer}-${index}`, tone: current ? "accent" : "plain", active: current, flow: current ? "forward" : undefined });
    });
  });
  return {
    caption: `${input.stages[input.step]?.title ?? "Latent"}: four latent points; each level up is the next stage, and only the first and last levels are in the client's own coordinates.`,
    nodes, links,
    legend: [{ tone: "good", text: "client's coordinates" }, { tone: "accent", text: "rotated coordinates the worker sees" }],
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
    const current = stage === input.step;
    values.forEach((value, index) => {
      nodes.push(bar(`f-${stage}-${index}`, index * spread, stage * 2.4, value, largest, value < 0 ? "warn" : current ? "accent" : "plain", {
        active: current,
        size: [0.6 * spread, 0, 0],
        opacity: stage > input.step ? 0.35 : 1,
        label: index === 0 ? `Stage ${stage}` : undefined,
        detail: `Stage ${stage}, entry ${index}: ${short(value)}`,
      }));
    });
  });
  if (input.step > 0 && input.stages[input.step]) {
    const block = 2 ** (input.step - 1);
    for (let index = 0; index < width; index += 1) {
      const partner = index ^ block;
      links.push({ id: `fs-${index}`, from: `f-${input.step - 1}-${index}`, to: `f-${input.step}-${index}`, tone: "plain", arc: 0.8, flow: "forward" });
      links.push({ id: `fx-${index}`, from: `f-${input.step - 1}-${partner}`, to: `f-${input.step}-${index}`, tone: "accent", arc: 1.6, dashed: true, flow: "forward" });
    }
  }
  return {
    caption: input.stages.length === 0
      ? "Enter a valid vector to draw the transform."
      : input.step === 0
        ? "Stage 0 is your input vector. Orange bars are negative values."
        : `Stage ${input.step}: every entry is the sum or difference of two entries ${2 ** (input.step - 1)} apart in the previous stage.`,
    nodes, links,
    legend: [{ tone: "accent", text: "current stage and its partner inputs" }, { tone: "warn", text: "negative value" }],
  };
}

// ---------------------------------------------------------------- pipeline

interface PipelineStep { request: number; state: string; stageIndex: number }

export function pipelineScene(input: { nodes: number; failNode: number | null; events: PipelineStep[]; index: number }): Scene3D {
  const event = input.events[Math.max(0, Math.min(input.index, input.events.length - 1))];
  if (!event) return { caption: "No events to show.", nodes: [], links: [] };
  const atClient = event.stageIndex < 0;
  const trouble = event.state === "resent" || event.state === "recovered";
  const span = (input.nodes - 1) * 2.2;
  const nodes: SceneNode[] = [{ id: "client", kind: "box", position: [-2.8, 0.65, 0], size: [1.3, 1.3, 1.3], tone: "good", label: "Client", detail: "Encrypts each request before it leaves and decodes the result when it returns.", active: atClient }];
  const links: SceneLink[] = [];
  for (let index = 0; index < input.nodes; index += 1) {
    const failed = input.failNode === index;
    const current = event.stageIndex === index;
    nodes.push({
      id: `w-${index}`, kind: "box", position: [index * 2.2, 0.5, 0], size: [1, 1, 1], tone: failed ? "warn" : "accent",
      label: `Worker ${index}${failed ? " (fails)" : ""}`,
      detail: failed ? "This worker fails once; the request is recovered and sent again." : "Computes one stage on encrypted state, then scrubs its memory.",
      active: current, opacity: failed && !current ? 0.55 : 1, pulse: current && trouble,
    });
    links.push({ id: `hop-${index}`, from: index === 0 ? "client" : `w-${index - 1}`, to: `w-${index}`, tone: current && trouble ? "warn" : current ? "accent" : "plain", active: current, flow: current ? "forward" : undefined });
  }
  links.push({ id: "return", from: `w-${input.nodes - 1}`, to: "client", tone: "good", dashed: true, arc: Math.max(2, span * 0.45), active: event.state === "decoded", flow: event.state === "decoded" ? "forward" : undefined });

  // One packet per request, parked wherever that request last was (wiping a worker does not move it).
  const requests = [...new Set(input.events.map((candidate) => candidate.request))].sort((a, b) => a - b);
  for (const request of requests) {
    let place: PipelineStep | undefined;
    for (let cursor = 0; cursor <= input.index && cursor < input.events.length; cursor += 1) {
      const candidate = input.events[cursor]!;
      if (candidate.request === request && candidate.state !== "scrubbed") place = candidate;
    }
    const waiting = !place;
    const done = place?.state === "decoded";
    const x = !place || place.stageIndex < 0 ? -2.8 : place.stageIndex * 2.2;
    const lane = (request % 4) * 0.42 - 0.63;
    const stuck = place?.state === "resent" || place?.state === "recovered";
    nodes.push({
      id: `req-${request}`, kind: "orb", position: [x, 1.75 + Math.floor(request / 4) * 0.45, lane], size: [0.3, 0.3, 0.3],
      tone: stuck ? "warn" : done ? "good" : "accent", opacity: waiting ? 0.35 : 1,
      label: `req ${request}`, detail: waiting ? "Not sent yet." : `Request ${request} is ${place!.state}.`,
      active: request === event.request, pulse: request === event.request && stuck,
    });
  }
  return {
    caption: `Request ${event.request} is ${event.state} at ${atClient ? "the client" : `worker ${event.stageIndex}`}.`,
    nodes, links,
    legend: [{ tone: "accent", text: "in flight" }, { tone: "warn", text: "lost or failed, sent again" }, { tone: "good", text: "returned to the client" }],
  };
}

// ---------------------------------------------------------------- onion

export function onionScene(input: { hops: string[]; step: number; tampered: boolean; compromised: boolean; colluding: boolean }): Scene3D {
  const count = input.hops.length;
  const gap = 2.6;
  const nodes: SceneNode[] = [{ id: "sender", kind: "box", position: [-gap, 0.5, 0], size: [1, 1, 1], tone: "good", label: "Sender", detail: "Wraps the payload once per hop, innermost layer first." }];
  const links: SceneLink[] = [];
  const at = Math.max(0, Math.min(input.step, count - 1));
  for (let index = 0; index < count; index += 1) {
    const current = index === at;
    const exposed = input.colluding || (input.compromised && current);
    nodes.push({
      id: `hop-${index}`, kind: "box", position: [index * gap, 0.5, 0], size: [1, 1, 1], tone: exposed ? "warn" : "plain",
      label: `Hop ${index} → ${input.hops[index]}${exposed ? " (compromised)" : ""}`,
      detail: "Can open only its own layer, which tells it nothing but the next hop.",
      active: current, pulse: exposed && current,
    });
    links.push({ id: `route-${index}`, from: index === 0 ? "sender" : `hop-${index - 1}`, to: `hop-${index}`, tone: input.colluding && index > 0 ? "warn" : current ? "accent" : "plain", dashed: input.colluding && index > 0, active: current, flow: current ? "forward" : undefined });
    if (index < at) nodes.push({ id: `peel-${index}`, kind: "ring", position: [index * gap, 0.06, 0], size: [1.9, 0, 0], tone: "accent", detail: `Layer ${index} was removed here.` });
  }
  // Layer j is removed by hop j, so the layers still wrapped are j ≥ at. Outer layers are larger.
  const remaining = Math.max(count - at, count === 0 ? 0 : 1);
  for (let layer = at; layer < count; layer += 1) {
    const outer = layer === at;
    nodes.push({
      id: `shell-${layer}`, kind: "shell", position: [at * gap, 2.6, 0], size: [0.75 + (count - layer) * 0.55, 0, 0],
      tone: outer && input.tampered ? "warn" : "accent", active: outer, pulse: outer && input.tampered,
      opacity: outer ? 0.34 : 0.2,
      label: outer ? `${remaining} layer${remaining === 1 ? "" : "s"} left${input.tampered ? ", tampered" : ""}` : undefined,
    });
  }
  if (count > 0) nodes.push({ id: "payload", kind: "orb", position: [at * gap, 2.6, 0], size: [0.42, 0.42, 0.42], tone: "good", detail: "The payload. Only the last hop's layer wraps it directly." });
  const risk = input.colluding ? " Colluding hops can link the two ends of the route." : input.compromised ? " A compromised hop sees only its own layer and the next hop." : "";
  return {
    caption: `At hop ${at}, ${remaining} layer${remaining === 1 ? "" : "s"} still wrap${remaining === 1 ? "s" : ""} the payload; this hop removes one.${input.tampered ? " The ciphertext was altered, so this hop's check fails." : ""}${risk}`,
    nodes, links,
    legend: [{ tone: "accent", text: "encryption layer" }, { tone: "good", text: "payload" }, { tone: "warn", text: "tampered or compromised" }],
  };
}

// ---------------------------------------------------------------- mixture of experts

export function moeScene(input: { tokens: number; plan: { tokenIndices: number[][]; dummy: boolean[][] } | null; loads: number[]; protectedMode: boolean; step: number }): Scene3D {
  if (!input.plan) return { caption: "Nothing can be dispatched: an expert would receive more tokens than it has slots.", nodes: [], links: [] };
  const nodes: SceneNode[] = [];
  const links: SceneLink[] = [];
  const largest = largestOf(input.loads);
  const experts = input.loads.length;
  const offset = ((input.tokens - 1) * 1.1 - (experts - 1) * 2.2) / 2;
  for (let token = 0; token < input.tokens; token += 1) {
    nodes.push({ id: `t-${token}`, kind: "sphere", position: [token * 1.1, 4.2, 0], size: [0.42, 0.42, 0.42], tone: "good", label: token === 0 ? "Tokens" : undefined, detail: `Token ${token}.` });
  }
  let real = 0;
  let dummy = 0;
  input.loads.forEach((load, expert) => {
    const active = expert === input.step;
    nodes.push(bar(`e-${expert}`, expert * 2.2 + offset, 0, load, largest, "accent", { label: `Expert ${expert}: ${load} slots`, detail: `Receives ${load} slot${load === 1 ? "" : "s"} this step. An observer sees only this number.`, active, size: [1.1, 0, 0] }));
    input.plan!.tokenIndices[expert]!.forEach((token, slot) => {
      const isDummy = input.plan!.dummy[expert]![slot]!;
      if (isDummy) dummy += 1;
      else real += 1;
      if (isDummy && !input.protectedMode) return;
      links.push({ id: `r-${expert}-${slot}`, from: `t-${token}`, to: `e-${expert}`, tone: isDummy ? "muted" : "accent", dashed: isDummy, active: active && !isDummy, flow: isDummy ? undefined : "forward" });
    });
  });
  return {
    caption: input.protectedMode
      ? `${real} real routes (solid) and ${dummy} dummy routes (dashed) give every expert the same load.`
      : `${real} real routes only: the bar heights reveal which experts the prompt used.`,
    nodes, links,
    legend: [{ tone: "good", text: "tokens" }, { tone: "accent", text: "experts and real routes" }, { tone: "muted", text: "dummy routes" }],
  };
}

// ---------------------------------------------------------------- memory

export function memoryScene(input: { steps: { operation: string; bytes: number[] }[]; index: number }): Scene3D {
  const current = input.steps[Math.max(0, Math.min(input.index, input.steps.length - 1))];
  if (!current) return { caption: "No step selected.", nodes: [], links: [] };
  const nodes: SceneNode[] = [];
  const links: SceneLink[] = [];
  const afterWipe = current.operation === "zero" || current.operation === "reuse";
  const stale = afterWipe && current.bytes.some((value) => value !== 0);
  const width = Math.max(current.bytes.length, 1);
  const track = (input.steps.length - 1) * 1.7;
  const shift = (track - (width - 1) * 1.3) / 2;
  // One buffer whose cells keep their identity, so a wipe is seen as the bytes draining away.
  nodes.push({ id: "buffer", kind: "slab", position: [track / 2, 0.05, -2.4], size: [width * 1.3 + 0.5, 0.1, 1.6], tone: stale ? "warn" : "muted", opacity: 0.4, label: "Physical buffer" });
  current.bytes.forEach((value, cell) => {
    const residue = afterWipe && value !== 0;
    nodes.push(bar(`cell-${cell}`, shift + cell * 1.3, -2.4, value, 255, value === 0 ? "good" : residue ? "warn" : "accent", {
      position: [0, 0.1, 0], size: [0.95, 0, 0], pulse: residue,
      label: `0x${value.toString(16).padStart(2, "0")}`,
      detail: value === 0 ? "This byte reads as zero." : residue ? "Left over from the previous request and readable by the next one." : "A secret byte of the current request.",
    }));
  });
  input.steps.forEach((step, column) => {
    const here = column === input.index;
    nodes.push({ id: `step-${column}`, kind: "cylinder", position: [column * 1.7, 0.12, 0], size: [1, 0.24, 1], tone: here ? (stale ? "warn" : "accent") : column < input.index ? "plain" : "muted", active: here, label: step.operation });
    if (column > 0) links.push({ id: `next-${column}`, from: `step-${column - 1}`, to: `step-${column}`, tone: here ? "accent" : "plain", active: here, flow: here ? "forward" : undefined });
  });
  return {
    caption: `${current.operation}: ${stale ? "the secret bytes are still there and the next owner can read them." : current.bytes.every((value) => value === 0) ? "all four bytes read as zero." : "the buffer holds the secret bytes."}`,
    nodes, links,
    legend: [{ tone: "accent", text: "secret bytes in use" }, { tone: "good", text: "zeroed" }, { tone: "warn", text: "stale bytes left behind" }],
  };
}
