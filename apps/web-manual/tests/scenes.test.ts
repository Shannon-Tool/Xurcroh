import { describe, expect, it } from "vitest";
import { fwht } from "@/lib/simulation/algebra";
import { executeShareGrid, placeOperations } from "@/lib/simulation/fortress";
import { simulateMemory } from "@/lib/simulation/memory";
import { routeMoe } from "@/lib/simulation/moe";
import { simulatePipeline } from "@/lib/simulation/pipeline";
import {
  conjugationScene, diffusionScene, freivaldsScene, fwhtScene, memoryScene, moeScene, onionScene,
  pipelineScene, placementScene, quantizedScene, shareGridScene,
} from "@/lib/scene/builders";
import type { Scene3D } from "@/lib/scene/types";

function wellFormed(scene: Scene3D): void {
  const ids = scene.nodes.map((node) => node.id);
  expect(new Set(ids).size).toBe(ids.length);
  for (const link of scene.links) {
    expect(ids).toContain(link.from);
    expect(ids).toContain(link.to);
  }
  const links = scene.links.map((link) => link.id);
  expect(new Set(links).size).toBe(links.length);
  for (const packet of scene.packets) expect(links).toContain(packet.link);
  for (const node of scene.nodes) {
    for (const value of [...node.position, ...node.size]) expect(Number.isFinite(value)).toBe(true);
  }
  expect(scene.caption.length).toBeGreaterThan(10);
}
const active = (scene: Scene3D) => scene.nodes.filter((node) => node.active).map((node) => node.id);

describe("3D scene builders", () => {
  it("share grid: the active worker follows the step and a tampered worker is marked", () => {
    const roles = executeShareGrid([[1, 2], [3, 4]], [5, 7], 0).roles;
    for (let step = 0; step < 4; step += 1) {
      const scene = shareGridScene({ roles, step, tampered: false });
      wellFormed(scene);
      expect(active(scene)).toEqual([`role-${step}`]);
      expect(scene.packets).toHaveLength(1);
    }
    const done = shareGridScene({ roles, step: 4, tampered: false });
    expect(active(done)).toEqual(["client"]);
    expect(done.packets).toHaveLength(0);
    const bad = shareGridScene({ roles, step: 0, tampered: true });
    expect(bad.nodes.find((node) => node.id === "role-0")!.tone).toBe("warn");
    expect(shareGridScene({ roles, step: 4, tampered: true }).caption).toMatch(/refuses/);
  });

  it("Freivalds: a wrong product is highlighted and the check turns to the warning tone", () => {
    const good = freivaldsScene({ x: [5, 7], product: [19, 43], challenge: [11, 13], expected: [19, 43], valid: true });
    wellFormed(good);
    expect(good.nodes.find((node) => node.id === "left")!.tone).toBe("good");
    const bad = freivaldsScene({ x: [5, 7], product: [20, 43], challenge: [11, 13], expected: [19, 43], valid: false });
    expect(bad.nodes.find((node) => node.id === "p-0")!.tone).toBe("warn");
    expect(bad.links[0]!.dashed).toBe(true);
  });

  it("quantized formats: one row of bars per stage and the current row is active", () => {
    const stages = [{ title: "a", values: [1, -2, 3] }, { title: "b", values: [0, 0, 0] }];
    const scene = quantizedScene({ stages, step: 1 });
    wellFormed(scene);
    expect(scene.nodes).toHaveLength(6);
    expect(active(scene)).toEqual(["q-1-0", "q-1-1", "q-1-2"]);
  });

  it("placement: sixteen blocks, none overlapping, and a reshuffle changes the picture", () => {
    const first = placementScene({ roles: placeOperations(0) });
    wellFormed(first);
    expect(first.nodes).toHaveLength(16);
    expect(new Set(first.nodes.map((node) => `${node.position[0]}/${node.position[2]}`)).size).toBe(16);
    expect(JSON.stringify(placementScene({ roles: placeOperations(1) }))).not.toBe(JSON.stringify(first));
  });

  it("conjugation: the input and the rotated input sit at the same distance from the origin", () => {
    const scene = conjugationScene({ x: [1, 2], rotatedInput: [2, 1], expected: [7, 10], transformedOutput: [10, 7], recovered: [7, 10], step: 1 });
    wellFormed(scene);
    const distance = (id: string) => Math.hypot(...scene.nodes.find((node) => node.id === id)!.position.map((value, axis) => (axis === 1 ? value - 0.1 : value)));
    expect(distance("x")).toBeCloseTo(distance("rx"));
    expect(active(scene)).toContain("rx");
  });

  it("diffusion: one level per stage with tracks between them", () => {
    const points = [[0, 0], [1, 0], [0, 1], [1, 1]];
    const scene = diffusionScene({ stages: ["a", "b", "c", "d"].map((title) => ({ title, points })), step: 2 });
    wellFormed(scene);
    expect(scene.nodes).toHaveLength(16);
    expect(scene.links).toHaveLength(12);
    expect(active(scene)).toHaveLength(4);
  });

  it("FWHT: butterflies join each entry to itself and its partner in the previous stage", () => {
    const { stages } = fwht([1, 2, 3, 4, 5, 6, 7, 8]);
    const scene = fwhtScene({ stages, step: 2 });
    wellFormed(scene);
    expect(scene.nodes).toHaveLength(stages.length * 8);
    expect(scene.links).toHaveLength(16);
    expect(scene.links.find((link) => link.id === "fx-0")!.from).toBe("f-1-2");
    expect(fwhtScene({ stages, step: 0 }).links).toHaveLength(0);
    wellFormed(fwhtScene({ stages: [], step: 0 }));
  });

  it("pipeline: the packet sits on the hop into the active worker, and returns to the client at the end", () => {
    const events = simulatePipeline({ nodes: 3, latencyMs: 3, packetLoss: 0, requests: 1, failNode: 1 });
    for (const event of events) {
      const scene = pipelineScene({ nodes: 3, failNode: 1, event });
      wellFormed(scene);
      expect(scene.packets[0]!.link).toBe(event.stageIndex < 0 ? "return" : `hop-${event.stageIndex}`);
      expect(active(scene)).toEqual([event.stageIndex < 0 ? "client" : `w-${event.stageIndex}`]);
    }
    expect(pipelineScene({ nodes: 3, failNode: 1, event: events[0]! }).nodes.find((node) => node.id === "w-1")!.tone).toBe("warn");
  });

  it("onion: one layer comes off per hop, and tampering or collusion is visible", () => {
    const hops = ["worker-1", "worker-2", "client"];
    const shells = (step: number) => onionScene({ hops, step, tampered: false, compromised: false, colluding: false }).nodes.filter((node) => node.kind === "shell").length;
    expect([shells(0), shells(1), shells(2)]).toEqual([3, 2, 1]);
    const scene = onionScene({ hops, step: 1, tampered: true, compromised: true, colluding: false });
    wellFormed(scene);
    expect(scene.nodes.find((node) => node.id === "shell-0")!.tone).toBe("warn");
    expect(scene.nodes.find((node) => node.id === "hop-1")!.tone).toBe("warn");
    expect(scene.nodes.find((node) => node.id === "hop-0")!.tone).toBe("plain");
    expect(onionScene({ hops, step: 0, tampered: false, compromised: false, colluding: true }).links.filter((link) => link.dashed)).toHaveLength(2);
  });

  it("mixture of experts: dummy routes are dashed and disappear when protection is off", () => {
    const hidden = [[1, 0], [0, 1], [1, 1], [2, 0], [0, 2], [2, 2]];
    const plan = routeMoe(hidden, [[1, 0], [0, 1], [1, 1], [2, 0.5]], 1, 4);
    const guarded = moeScene({ tokens: 6, plan, loads: plan.loads, protectedMode: true, step: 0 });
    wellFormed(guarded);
    expect(guarded.links).toHaveLength(16);
    expect(guarded.links.filter((link) => !link.dashed)).toHaveLength(6);
    const open = moeScene({ tokens: 6, plan, loads: [1, 1, 2, 2], protectedMode: false, step: 0 });
    expect(open.links).toHaveLength(6);
    expect(moeScene({ tokens: 6, plan: null, loads: [], protectedMode: true, step: 0 }).nodes).toHaveLength(0);
  });

  it("memory: stale bytes after a skipped wipe are in the warning tone", () => {
    const safe = memoryScene({ steps: simulateMemory(true), index: 5 });
    wellFormed(safe);
    expect(safe.nodes.filter((node) => node.tone === "warn")).toHaveLength(0);
    const unsafe = memoryScene({ steps: simulateMemory(false), index: 5 });
    expect(unsafe.nodes.filter((node) => node.tone === "warn")).toHaveLength(8);
    expect(unsafe.caption).toMatch(/next owner can read/);
    expect(active(unsafe)).toHaveLength(4);
  });
});
