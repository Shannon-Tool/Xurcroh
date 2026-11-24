import { describe, expect, it } from "vitest";
import { declutter } from "@/lib/scene/labels";
import { curvePoint, flowOffsets } from "@/lib/scene/motion";
import { DEFAULT_ORBIT, orbitMoving, restingMotion, stepOrbit } from "@/lib/scene/orbit";
import { isSlow, pickQuality } from "@/lib/scene/quality";
import { EMPTY_LIVE, advanceScene, type LiveScene } from "@/lib/scene/tween";
import type { Scene3D } from "@/lib/scene/types";

const first: Scene3D = {
  caption: "first",
  nodes: [
    { id: "a", kind: "box", position: [0, 0, 0], size: [1, 1, 1], tone: "plain", label: "A" },
    { id: "b", kind: "sphere", position: [4, 0, 0], size: [1, 1, 1], tone: "accent" },
  ],
  links: [{ id: "ab", from: "a", to: "b", tone: "plain", flow: "forward" }],
};
const second: Scene3D = {
  caption: "second",
  nodes: [
    { id: "a", kind: "box", position: [2, 0, 0], size: [1, 3, 1], tone: "warn", label: "A", active: true },
    { id: "c", kind: "orb", position: [0, 2, 0], size: [0.4, 0.4, 0.4], tone: "good" },
  ],
  links: [],
};

function run(live: LiveScene, target: Scene3D, seconds: number): LiveScene {
  let state = live;
  for (let elapsed = 0; elapsed < seconds; elapsed += 1 / 60) state = advanceScene(state, target, 1 / 60);
  return state;
}

describe("scene tweening", () => {
  it("shows a scene at once when asked to snap", () => {
    const live = advanceScene(EMPTY_LIVE, first, Infinity);
    expect(live.settled).toBe(true);
    expect(live.nodes.map((node) => [node.id, node.presence])).toEqual([["a", 1], ["b", 1]]);
    expect(live.links).toHaveLength(1);
    expect(live.links[0]!.a).toEqual([0, 0, 0]);
    expect(live.links[0]!.b).toEqual([4, 0, 0]);
  });

  it("grows new nodes in over several frames instead of popping", () => {
    const early = advanceScene(EMPTY_LIVE, first, 1 / 60);
    expect(early.settled).toBe(false);
    expect(early.nodes[0]!.presence).toBeGreaterThan(0);
    expect(early.nodes[0]!.presence).toBeLessThan(0.5);
    expect(run(EMPTY_LIVE, first, 2).settled).toBe(true);
  });

  it("glides a kept node to its new place, size and emphasis", () => {
    const start = advanceScene(EMPTY_LIVE, first, Infinity);
    const mid = run(start, second, 0.1);
    const a = mid.nodes.find((node) => node.id === "a")!;
    expect(a.position[0]).toBeGreaterThan(0.2);
    expect(a.position[0]).toBeLessThan(1.9);
    expect(a.size[1]).toBeGreaterThan(1);
    expect(a.size[1]).toBeLessThan(3);
    expect(a.emphasis).toBeGreaterThan(0);
    expect(a.tone).toBe("warn");
    const done = run(start, second, 2);
    expect(done.settled).toBe(true);
    expect(done.nodes.find((node) => node.id === "a")!.position).toEqual([2, 0, 0]);
    expect(done.nodes.find((node) => node.id === "a")!.emphasis).toBe(1);
  });

  it("fades removed nodes and their links out, then drops them", () => {
    const start = advanceScene(EMPTY_LIVE, first, Infinity);
    const mid = run(start, second, 0.1);
    const b = mid.nodes.find((node) => node.id === "b")!;
    expect(b.leaving).toBe(true);
    expect(b.presence).toBeLessThan(1);
    expect(mid.links[0]!.leaving).toBe(true);
    const done = run(start, second, 2);
    expect(done.nodes.map((node) => node.id).sort()).toEqual(["a", "c"]);
    expect(done.links).toHaveLength(0);
  });

  it("keeps links attached to nodes while they move", () => {
    const moved: Scene3D = { ...first, nodes: [first.nodes[0]!, { ...first.nodes[1]!, position: [4, 2, 0] }] };
    const mid = run(advanceScene(EMPTY_LIVE, first, Infinity), moved, 0.1);
    expect(mid.links[0]!.b).toEqual(mid.nodes.find((node) => node.id === "b")!.position);
  });
});

describe("flow along links", () => {
  it("spaces packets evenly and moves them in the link's direction", () => {
    const now = flowOffsets(0, 4, 0.5, 1);
    expect(now).toEqual([0, 0.25, 0.5, 0.75]);
    const later = flowOffsets(0.2, 4, 0.5, 1);
    expect(later[0]).toBeCloseTo(0.1);
    const back = flowOffsets(0.2, 4, 0.5, -1);
    expect(back[0]).toBeCloseTo(0.9);
    for (const value of flowOffsets(123.4, 5, 0.7, -1)) {
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(1);
    }
  });

  it("puts the middle of an arced link above the straight line", () => {
    expect(curvePoint([0, 0, 0], [2, 0, 0], 0, 0.5)).toEqual([1, 0, 0]);
    const middle = curvePoint([0, 0, 0], [2, 0, 0], 1, 0.5);
    expect(middle[0]).toBeCloseTo(1);
    expect(middle[1]).toBeCloseTo(0.5);
    expect(curvePoint([0, 0, 0], [2, 0, 0], 1, 0)).toEqual([0, 0, 0]);
    expect(curvePoint([0, 0, 0], [2, 0, 0], 1, 1)).toEqual([2, 0, 0]);
  });
});

describe("orbit motion", () => {
  it("keeps spinning after a drag and comes to rest", () => {
    const thrown = { ...restingMotion(DEFAULT_ORBIT), spinTheta: 2 };
    const soon = stepOrbit(thrown, 0.1, false);
    expect(soon.orbit.theta).toBeGreaterThan(DEFAULT_ORBIT.theta);
    expect(orbitMoving(soon)).toBe(true);
    let state = thrown;
    for (let index = 0; index < 600; index += 1) state = stepOrbit(state, 1 / 60, false);
    expect(orbitMoving(state)).toBe(false);
  });

  it("turns slowly on its own only while auto-rotation is on", () => {
    const still = restingMotion(DEFAULT_ORBIT);
    expect(stepOrbit(still, 1, false).orbit.theta).toBe(DEFAULT_ORBIT.theta);
    const turned = stepOrbit(still, 1, true).orbit.theta - DEFAULT_ORBIT.theta;
    expect(turned).toBeGreaterThan(0.02);
    expect(turned).toBeLessThan(0.3);
  });
});

describe("label declutter", () => {
  const point = (id: string, x: number, y: number, extra: Partial<{ active: boolean; depth: number; visible: boolean }> = {}) =>
    ({ id, text: "Worker label", x, y, visible: true, active: false, depth: 0.5, ...extra });

  it("hides a label that would sit on top of a more important one", () => {
    const shown = declutter([point("far", 100, 100, { depth: 0.9 }), point("near", 104, 102, { depth: 0.2 })]);
    expect(shown.map((label) => label.id)).toEqual(["near"]);
  });

  it("always prefers the active label and keeps labels that do not collide", () => {
    const shown = declutter([point("plain", 100, 100, { depth: 0.1 }), point("active", 102, 101, { active: true, depth: 0.9 }), point("apart", 400, 300)]);
    expect(shown.map((label) => label.id).sort()).toEqual(["active", "apart"]);
  });

  it("drops labels that are off screen", () => {
    expect(declutter([point("gone", 10, 10, { visible: false })])).toHaveLength(0);
  });
});

describe("quality choice", () => {
  it("follows the reader's choice first, then the hardware", () => {
    expect(pickQuality({ preference: "off", rendererName: "NVIDIA RTX", slow: false })).toBe("lite");
    expect(pickQuality({ preference: "on", rendererName: "SwiftShader", slow: true })).toBe("full");
    expect(pickQuality({ preference: "auto", rendererName: "ANGLE (Google, Vulkan (SwiftShader Device))", slow: false })).toBe("lite");
    expect(pickQuality({ preference: "auto", rendererName: "llvmpipe (LLVM 15)", slow: false })).toBe("lite");
    expect(pickQuality({ preference: "auto", rendererName: "Apple M2", slow: false })).toBe("full");
    expect(pickQuality({ preference: "auto", rendererName: "Apple M2", slow: true })).toBe("lite");
  });

  it("calls a run of long frames slow, but not a single hitch", () => {
    expect(isSlow(Array.from({ length: 40 }, () => 16))).toBe(false);
    expect(isSlow([...Array.from({ length: 39 }, () => 16), 400])).toBe(false);
    expect(isSlow(Array.from({ length: 40 }, () => 55))).toBe(true);
    expect(isSlow([55, 55, 55])).toBe(false);
  });
});
