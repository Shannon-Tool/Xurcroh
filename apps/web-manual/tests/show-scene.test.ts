import { describe, expect, it } from "vitest";
import { DEFAULT_ORBIT, followDirection } from "@/lib/scene/orbit";
import { showCamera, showScene, showWhy, type ShowSceneInput } from "@/lib/scene/show-scene";
import { EMPTY_LIVE, advanceScene } from "@/lib/scene/tween";
import type { Beat } from "@/lib/show/timeline";

const BEATS: Beat[] = ["split", "send", "compute", "return", "verify", "combine", "stream", "rejected", "done"];
const base: ShowSceneInput = { kind: "text", beat: "split", revealed: 0, units: 40, rejections: 0, tamperRole: null };
const flowing = (input: Partial<ShowSceneInput>) => showScene({ ...base, ...input }).links.filter((link) => link.flow).map((link) => link.id).sort();

describe("show scene", () => {
  it("is well formed on every beat of every kind and always carries the five screens", () => {
    for (const kind of ["text", "picture", "clip"] as const) {
      for (const beat of BEATS) {
        const scene = showScene({ ...base, kind, beat, tamperRole: beat === "rejected" ? 2 : null, rejections: beat === "rejected" ? 1 : 0 });
        const ids = scene.nodes.map((node) => node.id);
        expect(new Set(ids).size).toBe(ids.length);
        for (const link of scene.links) {
          expect(ids).toContain(link.from);
          expect(ids).toContain(link.to);
        }
        expect(scene.nodes.filter((node) => node.kind === "screen").map((node) => node.screen).sort()).toEqual(["device", "worker-0", "worker-1", "worker-2", "worker-3"]);
        expect(scene.caption.length).toBeGreaterThan(20);
      }
    }
  });

  it("sends shares out, brings products back, and does both at once while streaming", () => {
    expect(flowing({ beat: "send" })).toEqual(["out-0", "out-1", "out-2", "out-3"]);
    expect(flowing({ beat: "return" })).toEqual(["back-0", "back-1", "back-2", "back-3"]);
    expect(flowing({ beat: "stream" })).toHaveLength(8);
    expect(flowing({ beat: "compute" })).toEqual([]);
    expect(flowing({ beat: "done" })).toEqual([]);
  });

  it("lights the device when it works and the workers when they compute", () => {
    const active = (beat: Beat) => showScene({ ...base, beat }).nodes.filter((node) => node.active).map((node) => node.id);
    expect(active("split")).toEqual(["device"]);
    expect(active("compute")).toEqual(["worker-0", "worker-1", "worker-2", "worker-3"]);
    expect(active("verify")).toContain("device");
    expect(showScene({ ...base, beat: "verify" }).nodes.filter((node) => node.kind === "ring")).toHaveLength(4);
    expect(showScene({ ...base, beat: "send" }).nodes.filter((node) => node.kind === "ring")).toHaveLength(0);
  });

  it("flashes the device once per revealed unit", () => {
    const device = (revealed: number) => showScene({ ...base, beat: "stream", revealed }).nodes.find((node) => node.id === "device")!;
    expect(device(3).flash).toBe(3);
    expect(device(4).flash).toBe(4);
  });

  it("marks the tampered worker, cuts its return link and says which worker failed", () => {
    const scene = showScene({ ...base, beat: "rejected", revealed: 5, rejections: 1, tamperRole: 2 });
    const worker = scene.nodes.find((node) => node.id === "worker-2")!;
    expect(worker.tone).toBe("warn");
    expect(worker.flash).toBe(1);
    expect(scene.nodes.find((node) => node.id === "worker-1")!.tone).toBe("accent");
    const link = scene.links.find((entry) => entry.id === "back-2")!;
    expect(link.tone).toBe("warn");
    expect(link.dashed).toBe(true);
    expect(scene.caption).toContain("Worker (1,0)");
    expect(scene.caption).toMatch(/rejected/);
    expect(showCamera("rejected", 2).focusId).toBe("worker-2");
  });

  it("names what is being produced in the caption", () => {
    expect(showScene({ ...base, kind: "text", beat: "stream", revealed: 7 }).caption).toMatch(/Token 7 of 40/);
    expect(showScene({ ...base, kind: "picture", beat: "stream", revealed: 3, units: 14 }).caption).toMatch(/Step 3 of 14/);
    expect(showScene({ ...base, kind: "clip", beat: "stream", revealed: 5, units: 24 }).caption).toMatch(/Frame 3 of 12/);
  });

  it("offers everyday wording and a reason for every beat", () => {
    for (const beat of BEATS) {
      const plain = showScene({ ...base, beat, words: "plain", tamperRole: beat === "rejected" ? 1 : null }).caption;
      expect(plain).not.toMatch(/share|product|Freivalds/i);
      expect(plain.length).toBeGreaterThan(20);
      expect(showWhy(beat).length).toBeGreaterThan(20);
    }
    expect(showScene({ ...base, beat: "split", words: "plain" }).caption).toMatch(/^Step 1 of 6 · Split\./);
    expect(showScene({ ...base, beat: "verify", words: "plain" }).caption).toMatch(/^Step 5 of 6 · Check\./);
    expect(showScene({ ...base, beat: "split" }).caption).toMatch(/shares/);
  });

  it("carries screens and flashes through the tween", () => {
    const live = advanceScene(EMPTY_LIVE, showScene({ ...base, beat: "stream", revealed: 2 }), Infinity);
    expect(live.nodes.find((node) => node.id === "screen-device")!.screen).toBe("device");
    expect(live.nodes.find((node) => node.id === "device")!.flash).toBe(2);
    expect(live.nodes.find((node) => node.id === "controller")!.flash).toBe(0);
  });
});

describe("directed camera", () => {
  it("eases the view towards the director's shot and sways around it", () => {
    const shot = showCamera("split", null);
    const part = followDirection(DEFAULT_ORBIT, { ...shot, sway: 0 }, 0, 0.5);
    expect(part.theta).toBeCloseTo((DEFAULT_ORBIT.theta + shot.theta) / 2);
    expect(part.zoom).toBeCloseTo((DEFAULT_ORBIT.zoom + shot.zoom) / 2);
    expect(followDirection(DEFAULT_ORBIT, { ...shot, sway: 0 }, 0, 1)).toEqual({ theta: shot.theta, phi: shot.phi, zoom: shot.zoom });
    const swayed = followDirection(DEFAULT_ORBIT, { ...shot, sway: 0.3 }, 4, 1);
    expect(swayed.theta).not.toBe(shot.theta);
    expect(Math.abs(swayed.theta - shot.theta)).toBeLessThanOrEqual(0.3);
  });

  it("uses a wide swaying shot for the stream and close shots for the slow steps", () => {
    expect(showCamera("stream", null).focusId).toBeNull();
    expect(showCamera("stream", null).sway).toBeGreaterThan(0);
    expect(showCamera("split", null).focusId).toBe("device");
    expect(showCamera("combine", null).focusId).toBe("device");
    expect(showCamera("compute", null).theta).toBeGreaterThan(showCamera("send", null).theta);
  });
});
