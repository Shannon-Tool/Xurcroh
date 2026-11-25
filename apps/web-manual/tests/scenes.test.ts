import { describe, expect, it } from "vitest";
import { fwht } from "@/lib/simulation/algebra";
import { executeShareGrid, placeOperations } from "@/lib/simulation/fortress";
import { simulateMemory } from "@/lib/simulation/memory";
import { routeMoe } from "@/lib/simulation/moe";
import { simulatePipeline } from "@/lib/simulation/pipeline";
import {
  conjugationScene, diffusionScene, freivaldsScene, fwhtScene, memoryScene,
  moeScene, onionScene, pipelineScene, placementScene, quantizedScene, shareGridScene,
} from "@/lib/scene/builders";
import type { Scene3D } from "@/lib/scene/types";

const TONES = ["plain", "accent", "good", "warn", "muted", "cipher"];

function wellFormed(scene: Scene3D): void {
  const ids = scene.nodes.map((node) => node.id);
  expect(new Set(ids).size).toBe(ids.length);
  for (const link of scene.links) {
    expect(ids).toContain(link.from);
    expect(ids).toContain(link.to);
  }
  const links = scene.links.map((link) => link.id);
  expect(new Set(links).size).toBe(links.length);
  for (const node of scene.nodes) {
    for (const value of [...node.position, ...node.size, ...(node.tip ?? [])]) expect(Number.isFinite(value)).toBe(true);
    if (node.kind === "arrow") expect(node.tip).toBeDefined();
    expect(TONES).toContain(node.tone);
  }
  for (const entry of scene.legend ?? []) expect(TONES).toContain(entry.tone);
  expect(scene.caption.length).toBeGreaterThan(10);
}
const active = (scene: Scene3D) => scene.nodes.filter((node) => node.active).map((node) => node.id);
const find = (scene: Scene3D, id: string) => scene.nodes.find((node) => node.id === id);
const flowing = (scene: Scene3D) => scene.links.filter((link) => link.flow).map((link) => link.id);

describe("3D scene builders", () => {
  it("share grid: shares go out to the active worker, verified products come back, tampering cuts the stream", () => {
    const roles = executeShareGrid([[1, 2], [3, 4]], [5, 7], 0).roles;
    for (let step = 0; step < 4; step += 1) {
      const scene = shareGridScene({ roles, step, tampered: false });
      wellFormed(scene);
      expect(active(scene)).toEqual([`role-${step}`]);
      expect(scene.links.find((link) => link.id === `link-${step}`)!.flow).toBe("forward");
      expect(scene.links.filter((link) => link.flow === "back")).toHaveLength(step);
      expect(scene.nodes.filter((node) => node.kind === "ring")).toHaveLength(step);
    }
    const done = shareGridScene({ roles, step: 4, tampered: false });
    expect(active(done)).toEqual(["client"]);
    expect(find(done, "output")).toBeDefined();
    const bad = shareGridScene({ roles, step: 2, tampered: true });
    expect(find(bad, "role-0")!.tone).toBe("warn");
    expect(find(bad, "seal-0")).toBeUndefined();
    expect(bad.links.find((link) => link.id === "link-0")!.flow).toBeUndefined();
    const refused = shareGridScene({ roles, step: 4, tampered: true });
    expect(refused.caption).toMatch(/refuses/);
    expect(find(refused, "output")).toBeUndefined();
    expect(find(refused, "client")!.tone).toBe("warn");
  });

  it("Freivalds: the beam is level when the check passes and tips when a product is wrong", () => {
    const good = freivaldsScene({ x: [5, 7], product: [19, 43], challenge: [11, 13], expected: [19, 43], valid: true });
    wellFormed(good);
    expect(find(good, "beam")!.tilt).toBe(0);
    expect(find(good, "left")!.tone).toBe("good");
    expect(find(good, "left")!.position[1]).toBe(find(good, "right")!.position[1]);
    const bad = freivaldsScene({ x: [5, 7], product: [20, 43], challenge: [11, 13], expected: [19, 43], valid: false });
    wellFormed(bad);
    expect(find(bad, "p-0")!.tone).toBe("warn");
    expect(find(bad, "beam")!.tilt).not.toBe(0);
    // The claimed side is heavier (20·11 > 19·11), so it hangs lower.
    expect(find(bad, "right")!.position[1]).toBeLessThan(find(bad, "left")!.position[1]);
    expect(find(bad, "right")!.detail).toMatch(/779/);
  });

  it("quantized formats: one platform of bars per stage, the current one lit and fed by the flow", () => {
    const stages = [{ title: "a", values: [1, -2, 3] }, { title: "b", values: [0, 0, 0] }];
    const scene = quantizedScene({ stages, step: 1 });
    wellFormed(scene);
    expect(scene.nodes.filter((node) => node.id.startsWith("q-"))).toHaveLength(6);
    expect(scene.nodes.filter((node) => node.kind === "slab")).toHaveLength(2);
    expect(active(scene)).toEqual(["q-1-0", "q-1-1", "q-1-2"]);
    expect(flowing(scene)).toEqual(["flow-1"]);
    expect(flowing(quantizedScene({ stages, step: 0 }))).toEqual([]);
  });

  it("placement: sixteen blocks on distinct pads, grouped by jurisdiction, and a reshuffle moves them", () => {
    const first = placementScene({ roles: placeOperations(0) });
    wellFormed(first);
    const blocks = (scene: Scene3D) => scene.nodes.filter((node) => node.id.startsWith("place-"));
    expect(blocks(first)).toHaveLength(16);
    expect(new Set(blocks(first).map((node) => `${node.position[0]}/${node.position[2]}`)).size).toBe(16);
    expect(first.nodes.filter((node) => node.kind === "slab")).toHaveLength(2);
    expect(blocks(first).every((node) => node.tone !== "warn")).toBe(true);
    const second = placementScene({ roles: placeOperations(1) });
    expect(blocks(second).map((node) => node.id)).toEqual(blocks(first).map((node) => node.id));
    expect(JSON.stringify(blocks(second).map((node) => node.position))).not.toBe(JSON.stringify(blocks(first).map((node) => node.position)));
  });

  it("conjugation: the input and the rotated input sit at the same distance from the origin", () => {
    const scene = conjugationScene({ x: [1, 2], rotatedInput: [2, 1], expected: [7, 10], transformedOutput: [10, 7], recovered: [7, 10], step: 1 });
    wellFormed(scene);
    const distance = (id: string) => Math.hypot(...find(scene, id)!.position.map((value, axis) => (axis === 1 ? value - 0.1 : value)));
    expect(distance("x")).toBeCloseTo(distance("rx"));
    expect(active(scene)).toContain("rx");
    expect(find(scene, "arrow-rx")!.tip).toEqual(find(scene, "rx")!.position);
    expect(flowing(scene)).toEqual(["rotate"]);
    expect(flowing(conjugationScene({ x: [1, 2], rotatedInput: [2, 1], expected: [7, 10], transformedOutput: [10, 7], recovered: [7, 10], step: 3 }))).toEqual(["unrotate"]);
  });

  it("diffusion: one level per stage with tracks between them", () => {
    const points = [[0, 0], [1, 0], [0, 1], [1, 1]];
    const scene = diffusionScene({ stages: ["a", "b", "c", "d"].map((title) => ({ title, points })), step: 2 });
    wellFormed(scene);
    expect(scene.nodes.filter((node) => node.id.startsWith("d-"))).toHaveLength(16);
    expect(scene.nodes.filter((node) => node.kind === "slab")).toHaveLength(4);
    expect(scene.links).toHaveLength(12);
    expect(active(scene)).toHaveLength(4);
    expect(flowing(scene)).toHaveLength(4);
    expect(find(scene, "level-0")!.tone).toBe("good");
    expect(find(scene, "level-1")!.tone).toBe("accent");
  });

  it("FWHT: butterflies join each entry to itself and its partner in the previous stage", () => {
    const { stages } = fwht([1, 2, 3, 4, 5, 6, 7, 8]);
    const scene = fwhtScene({ stages, step: 2 });
    wellFormed(scene);
    expect(scene.nodes).toHaveLength(stages.length * 8);
    expect(scene.links).toHaveLength(16);
    expect(scene.links.find((link) => link.id === "fx-0")!.from).toBe("f-1-2");
    expect(flowing(scene)).toHaveLength(16);
    expect(fwhtScene({ stages, step: 0 }).links).toHaveLength(0);
    wellFormed(fwhtScene({ stages: [], step: 0 }));
  });

  it("pipeline: every request has its own packet, which follows that request from stage to stage", () => {
    const events = simulatePipeline({ nodes: 3, latencyMs: 3, packetLoss: 0, requests: 2, failNode: 1 });
    events.forEach((event, index) => {
      const scene = pipelineScene({ nodes: 3, failNode: 1, events, index });
      wellFormed(scene);
      expect(scene.nodes.filter((node) => node.id.startsWith("req-"))).toHaveLength(2);
      expect(active(scene)).toContain(event.stageIndex < 0 ? "client" : `w-${event.stageIndex}`);
      expect(active(scene)).toContain(`req-${event.request}`);
      if (event.stageIndex >= 0 && event.state !== "scrubbed") expect(find(scene, `req-${event.request}`)!.position[0]).toBe(find(scene, `w-${event.stageIndex}`)!.position[0]);
    });
    const start = pipelineScene({ nodes: 3, failNode: 1, events, index: 0 });
    expect(find(start, "w-1")!.tone).toBe("warn");
    expect(find(start, "req-1")!.opacity).toBeLessThan(1);
    const recovering = events.findIndex((event) => event.state === "recovered");
    expect(find(pipelineScene({ nodes: 3, failNode: 1, events, index: recovering }), `req-${events[recovering]!.request}`)!.tone).toBe("warn");
    const end = pipelineScene({ nodes: 3, failNode: 1, events, index: events.length - 1 });
    expect(find(end, "req-0")!.tone).toBe("good");
    expect(find(end, "req-0")!.position[0]).toBe(find(end, "client")!.position[0]);
  });

  it("onion: one layer comes off per hop and stays behind, and tampering or collusion is visible", () => {
    const hops = ["worker-1", "worker-2", "client"];
    const at = (step: number) => onionScene({ hops, step, tampered: false, compromised: false, colluding: false });
    const shells = (step: number) => at(step).nodes.filter((node) => node.kind === "shell").map((node) => node.id);
    expect([shells(0), shells(1), shells(2)]).toEqual([["shell-0", "shell-1", "shell-2"], ["shell-1", "shell-2"], ["shell-2"]]);
    expect(at(2).nodes.filter((node) => node.kind === "ring").map((node) => node.id)).toEqual(["peel-0", "peel-1"]);
    expect(find(at(1), "payload")!.position[0]).toBe(find(at(1), "hop-1")!.position[0]);
    const scene = onionScene({ hops, step: 1, tampered: true, compromised: true, colluding: false });
    wellFormed(scene);
    expect(find(scene, "shell-1")!.tone).toBe("warn");
    expect(find(scene, "hop-1")!.tone).toBe("warn");
    expect(find(scene, "hop-0")!.tone).toBe("plain");
    expect(onionScene({ hops, step: 0, tampered: false, compromised: false, colluding: true }).links.filter((link) => link.dashed)).toHaveLength(2);
    wellFormed(onionScene({ hops: [], step: 0, tampered: false, compromised: false, colluding: false }));
  });

  it("onion: the packet is wrapped at the sender, travels, stops where it is refused, and carries its real bytes as a label", () => {
    const hops = ["worker-1", "worker-2", "client"];
    const base = { hops, tampered: false, compromised: false, colluding: false };
    const shells = (scene: Scene3D) => scene.nodes.filter((node) => node.kind === "shell").length;
    // Wrapping: the packet is at the sender and gains one layer at a time.
    const wrapping = [0, 1, 2, 3].map((wrapped) => onionScene({ ...base, step: 0, phase: "wrap", wrapped, note: `sealed ×${wrapped}` }));
    expect(wrapping.map(shells)).toEqual([0, 1, 2, 3]);
    for (const scene of wrapping) {
      wellFormed(scene);
      expect(find(scene, "payload")!.position[0]).toBe(find(scene, "sender")!.position[0]);
      expect(scene.nodes.some((node) => node.kind === "ring")).toBe(false);
    }
    expect(find(wrapping[0]!, "payload")!.label).toBe("sealed ×0");
    expect(wrapping[2]!.nodes.find((node) => node.kind === "shell")!.label).toBe("sealed ×2");
    // Arrived: no layer left, every relay has removed its own.
    const arrived = onionScene({ ...base, step: 2, phase: "arrived", note: "readable again" });
    expect(shells(arrived)).toBe(0);
    expect(arrived.nodes.filter((node) => node.kind === "ring")).toHaveLength(3);
    expect(find(arrived, "payload")!.label).toBe("readable again");
    // Refused at the middle relay: the packet stays there, with its layers on, however far the playback goes.
    for (const phase of ["relay", "arrived"] as const) {
      const stopped = onionScene({ ...base, tampered: true, step: 2, phase, stoppedAt: 1 });
      wellFormed(stopped);
      expect(find(stopped, "payload")!.position[0]).toBe(find(stopped, "hop-1")!.position[0]);
      expect(find(stopped, "hop-1")!.tone).toBe("warn");
      expect(find(stopped, "hop-1")!.label).toContain("refuses");
      expect(shells(stopped)).toBe(2);
      expect(stopped.caption).toContain("goes no further");
    }
  });

  it("share grid, check and placement follow the step they are given", () => {
    const roles = [0, 1, 2, 3].map((role) => ({ activationShare: role >> 1, weightShare: role & 1 }));
    const before = shareGridScene({ roles, step: -1, tampered: false, values: { client: "Your device: 1 0 1" } });
    wellFormed(before);
    expect(find(before, "client")!.label).toBe("Your device: 1 0 1");
    expect(before.nodes.some((node) => node.active)).toBe(false);
    expect(before.caption).toContain("nothing has been sent yet");
    const working = shareGridScene({ roles, step: 1, tampered: false, values: { roles: ["sent 5 …", "piece 7 …"] } });
    expect(find(working, "role-1")!.label).toBe("Worker (0,1): piece 7 …");
    expect(find(working, "role-1")!.active).toBe(true);

    const check = (step: number) => freivaldsScene({ x: [1, 1], product: [600, 300], challenge: [11, 13], expected: [600, 300], valid: true, step, sides: { left: "10 500", right: "10 500" } });
    expect(check(0).nodes.some((node) => node.id.startsWith("r-"))).toBe(false);
    expect(find(check(0), "beam")!.label).toBe("not compared yet");
    expect(find(check(2), "right")!.label).toBe("from the result: 10 500");
    expect(check(2).nodes.some((node) => node.id === "left")).toBe(false);
    expect(find(check(3), "left")!.active).toBe(true);
    expect(find(check(4), "beam")!.label).toBe("balanced: accept");
    for (const step of [0, 1, 2, 3, 4]) wellFormed(check(step));

    const placed = placeOperations(3);
    const count = (scene: Scene3D) => scene.nodes.filter((node) => node.id.startsWith("place-")).length;
    expect([0, 1, 2, 3, 4].map((done) => count(placementScene({ roles: placed, placed: done })))).toEqual([0, 4, 8, 12, 16]);
    expect(placementScene({ roles: placed, placed: 2 }).nodes.filter((node) => node.active)).toHaveLength(4);
    expect(placementScene({ roles: placed }).nodes.some((node) => node.active)).toBe(false);
  });

  it("mixture of experts: dummy routes are dashed and still, and disappear when protection is off", () => {
    const hidden = [[1, 0], [0, 1], [1, 1], [2, 0], [0, 2], [2, 2]];
    const plan = routeMoe(hidden, [[1, 0], [0, 1], [1, 1], [2, 0.5]], 1, 4);
    const guarded = moeScene({ tokens: 6, plan, loads: plan.loads, protectedMode: true, step: 0 });
    wellFormed(guarded);
    expect(guarded.links).toHaveLength(16);
    expect(guarded.links.filter((link) => !link.dashed)).toHaveLength(6);
    expect(flowing(guarded)).toHaveLength(6);
    const open = moeScene({ tokens: 6, plan, loads: [1, 1, 2, 2], protectedMode: false, step: 0 });
    expect(open.links).toHaveLength(6);
    expect(moeScene({ tokens: 6, plan: null, loads: [], protectedMode: true, step: 0 }).nodes).toHaveLength(0);
  });

  it("memory: the same four cells drain to zero on a wipe, or stay behind in the warning tone", () => {
    const cells = (scene: Scene3D) => scene.nodes.filter((node) => node.id.startsWith("cell-"));
    const before = memoryScene({ steps: simulateMemory(true), index: 2 });
    const safe = memoryScene({ steps: simulateMemory(true), index: 5 });
    wellFormed(safe);
    expect(cells(safe).map((node) => node.id)).toEqual(cells(before).map((node) => node.id));
    expect(cells(safe)[0]!.size[1]).toBeLessThan(cells(before)[0]!.size[1]);
    expect(safe.nodes.filter((node) => node.tone === "warn")).toHaveLength(0);
    expect(cells(safe).every((node) => node.tone === "good")).toBe(true);
    const unsafe = memoryScene({ steps: simulateMemory(false), index: 5 });
    expect(cells(unsafe).every((node) => node.tone === "warn" && node.pulse)).toBe(true);
    expect(unsafe.caption).toMatch(/next owner can read/);
    expect(active(unsafe)).toContain("step-5");
    expect(flowing(unsafe)).toEqual(["next-5"]);
  });
});
