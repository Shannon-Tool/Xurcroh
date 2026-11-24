import { describe, expect, it } from "vitest";
import { DEFAULT_ORBIT, followDirection } from "@/lib/scene/orbit";
import { relayOf, showCamera, showScene, showWhy, type ShowSceneInput } from "@/lib/scene/show-scene";
import { EMPTY_LIVE, advanceScene } from "@/lib/scene/tween";
import type { Beat } from "@/lib/show/timeline";

const BEATS: Beat[] = ["split", "attest", "seal", "route", "compute", "return", "verify", "combine", "stream", "rejected", "done"];
const base: ShowSceneInput = { kind: "text", beat: "split", revealed: 0, units: 40, rejections: 0, tamperRole: null };
const scene = (input: Partial<ShowSceneInput>) => showScene({ ...base, ...input });
const flowing = (input: Partial<ShowSceneInput>) => scene(input).links.filter((link) => link.flow).map((link) => link.id).sort();
const node = (input: Partial<ShowSceneInput>, id: string) => scene(input).nodes.find((entry) => entry.id === id)!;

describe("show scene", () => {
  it("is well formed on every beat of every kind, with a screen on the device, every relay and every worker", () => {
    for (const kind of ["text", "picture", "clip"] as const) {
      for (const beat of BEATS) {
        for (const rejectKind of beat === "rejected" ? (["result", "software", "packet"] as const) : [null]) {
          const built = scene({ kind, beat, rejectKind, tamperRole: beat === "rejected" ? 2 : null, rejections: beat === "rejected" ? 1 : 0 });
          const ids = built.nodes.map((entry) => entry.id);
          expect(new Set(ids).size).toBe(ids.length);
          const links = built.links.map((link) => link.id);
          expect(new Set(links).size).toBe(links.length);
          for (const link of built.links) {
            expect(ids).toContain(link.from);
            expect(ids).toContain(link.to);
          }
          expect(built.nodes.filter((entry) => entry.kind === "screen").map((entry) => entry.screen).sort())
            .toEqual(["device", "relay-0", "relay-1", "relay-2", "worker-0", "worker-1", "worker-2", "worker-3"]);
          expect(built.caption.length).toBeGreaterThan(20);
        }
      }
    }
  });

  it("puts a three-relay circuit between the device and four workers that each sit in an enclave", () => {
    const built = scene({ beat: "route" });
    const relays = built.nodes.filter((entry) => /^relay-\d$/.test(entry.id));
    expect(relays).toHaveLength(3);
    expect(relays.every((entry) => entry.tone === "cipher")).toBe(true);
    const device = node({ beat: "route" }, "device").position[0];
    const worker = node({ beat: "route" }, "worker-0").position[0];
    for (const relay of relays) expect(relay.position[0] > device && relay.position[0] < worker).toBe(true);
    for (let index = 0; index < 4; index += 1) {
      const shell = node({ beat: "route" }, `enclave-${index}`);
      const inside = node({ beat: "route" }, `worker-${index}`);
      expect(shell.kind).toBe("shell");
      expect(shell.size[0]).toBeGreaterThan(inside.size[0] * 1.5);
      expect([shell.position[0], shell.position[2]]).toEqual([inside.position[0], inside.position[2]]);
    }
    // The way out runs device → relay → relay → relay → workers, and every link on it carries sealed data.
    const out = built.links.filter((link) => /^(hop|fan)-/.test(link.id));
    expect(out.map((link) => `${link.from}>${link.to}`).slice(0, 5)).toEqual(expect.arrayContaining(["device>relay-0", "relay-0>relay-1", "relay-1>relay-2"]));
    expect(out.every((link) => link.tone === "cipher" && link.flow === "forward")).toBe(true);
    expect(built.legend!.some((entry) => entry.tone === "cipher")).toBe(true);
  });

  it("routes sealed data out, brings sealed results back, and does both while streaming", () => {
    expect(flowing({ beat: "route" })).toEqual(["fan-0", "fan-1", "fan-2", "fan-3", "hop-0", "hop-1", "hop-2"]);
    expect(flowing({ beat: "return" })).toEqual(["back-0", "back-1", "back-2", "ret-0", "ret-1", "ret-2", "ret-3"]);
    expect(flowing({ beat: "stream" })).toHaveLength(14);
    expect(flowing({ beat: "compute" })).toEqual([]);
    expect(flowing({ beat: "done" })).toEqual([]);
  });

  it("shows the reports during attestation and the sealed packet with three layers during sealing", () => {
    expect(scene({ beat: "attest" }).links.filter((link) => link.id.startsWith("report-")).map((link) => `${link.from}>${link.to}`))
      .toEqual(["worker-0>device", "worker-1>device", "worker-2>device", "worker-3>device"]);
    expect(scene({ beat: "route" }).links.some((link) => link.id.startsWith("report-"))).toBe(false);
    expect(node({ beat: "attest" }, "enclave-1").pulse).toBe(true);
    const sealing = scene({ beat: "seal" });
    expect(sealing.nodes.find((entry) => entry.id === "packet")!.tone).toBe("cipher");
    expect(sealing.nodes.filter((entry) => entry.id.startsWith("layer-") && entry.kind === "ring")).toHaveLength(3);
    expect(scene({ beat: "route" }).nodes.some((entry) => entry.id === "packet")).toBe(false);
  });

  it("lights the device when it works and the workers when they compute", () => {
    const active = (beat: Beat) => scene({ beat }).nodes.filter((entry) => entry.active).map((entry) => entry.id);
    expect(active("split")).toEqual(["device"]);
    expect(active("compute")).toEqual(["worker-0", "worker-1", "worker-2", "worker-3"]);
    expect(active("verify")).toContain("device");
    expect(scene({ beat: "verify" }).nodes.filter((entry) => entry.id.startsWith("seal-"))).toHaveLength(4);
    expect(scene({ beat: "route" }).nodes.filter((entry) => entry.id.startsWith("seal-"))).toHaveLength(0);
  });

  it("flashes the device once per revealed unit", () => {
    expect(node({ beat: "stream", revealed: 3 }, "device").flash).toBe(3);
    expect(node({ beat: "stream", revealed: 4 }, "device").flash).toBe(4);
  });

  it("marks a worker whose result failed, cuts its return link and names it", () => {
    const input = { beat: "rejected" as const, rejectKind: "result" as const, revealed: 5, rejections: 1, tamperRole: 2 };
    expect(node(input, "worker-2").tone).toBe("warn");
    expect(node(input, "worker-2").flash).toBe(1);
    expect(node(input, "worker-1").tone).toBe("accent");
    expect(scene(input).links.find((link) => link.id === "ret-2")).toMatchObject({ tone: "warn", dashed: true });
    expect(scene(input).caption).toContain("Worker (1,0)");
    expect(scene(input).caption).toMatch(/rejected/);
    expect(showCamera("rejected", 2, "result").focusId).toBe("worker-2");
  });

  it("marks an enclave whose software was swapped and sends nothing to it", () => {
    const input = { beat: "rejected" as const, rejectKind: "software" as const, rejections: 1, tamperRole: 1, words: "plain" as const };
    expect(node(input, "enclave-1").tone).toBe("warn");
    expect(node(input, "enclave-0").tone).toBe("good");
    expect(node(input, "worker-1").tone).toBe("accent");
    expect(scene(input).links.find((link) => link.id === "report-1")).toMatchObject({ tone: "warn", dashed: true });
    expect(scene(input).links.find((link) => link.id === "fan-1")!.flow).toBeUndefined();
    expect(scene(input).caption).toMatch(/Worker \(0,1\) is not running the expected software.*sends it nothing/);
  });

  it("stops an altered packet at a relay and lets nothing pass beyond it", () => {
    const input = { beat: "rejected" as const, rejectKind: "packet" as const, rejections: 1, tamperRole: 1, words: "plain" as const };
    const stop = relayOf(1);
    expect(node(input, `relay-${stop}`).tone).toBe("warn");
    expect(scene(input).links.find((link) => link.id === `hop-${stop}`)).toMatchObject({ tone: "warn", dashed: true });
    expect(scene(input).links.filter((link) => link.id.startsWith("fan-")).every((link) => !link.flow && !link.active)).toBe(true);
    expect(scene(input).caption).toContain(`Relay ${stop + 1} noticed`);
    expect(showCamera("rejected", 1, "packet").focusId).toBe(`relay-${stop}`);
  });

  it("names what is being produced in the caption", () => {
    expect(scene({ kind: "text", beat: "stream", revealed: 7 }).caption).toMatch(/Token 7 of 40/);
    expect(scene({ kind: "picture", beat: "stream", revealed: 3, units: 14 }).caption).toMatch(/Step 3 of 14/);
    expect(scene({ kind: "clip", beat: "stream", revealed: 5, units: 24 }).caption).toMatch(/Frame 3 of 12/);
  });

  it("offers everyday wording and a reason for every beat, and is honest about what is simulated", () => {
    for (const beat of BEATS) {
      const plain = scene({ beat, words: "plain", tamperRole: beat === "rejected" ? 1 : null }).caption;
      expect(plain).not.toMatch(/share|product|Freivalds|AEAD|attestation/i);
      expect(plain.length).toBeGreaterThan(20);
      expect(showWhy(beat).length).toBeGreaterThan(20);
    }
    expect(scene({ beat: "split", words: "plain" }).caption).toMatch(/^Step 1 of 8 · Split\./);
    expect(scene({ beat: "attest", words: "plain" }).caption).toMatch(/^Step 2 of 8 · Attest\./);
    expect(scene({ beat: "verify", words: "plain" }).caption).toMatch(/^Step 7 of 8 · Check\./);
    expect(scene({ beat: "split" }).caption).toMatch(/shares/);
    expect(showWhy("attest")).toMatch(/one more layer, not proof/);
    expect(showWhy("split")).toMatch(/disguise/);
    expect(showWhy("done")).toMatch(/stand-ins inside the page/);
    expect(node({ beat: "route" }, "circuit").label).toMatch(/stand-ins/);
    expect(node({ beat: "route" }, "zone").label).toMatch(/simulated/);
  });

  it("carries screens and flashes through the tween", () => {
    const live = advanceScene(EMPTY_LIVE, scene({ beat: "stream", revealed: 2 }), Infinity);
    expect(live.nodes.find((entry) => entry.id === "screen-device")!.screen).toBe("device");
    expect(live.nodes.find((entry) => entry.id === "screen-relay-1")!.screen).toBe("relay-1");
    expect(live.nodes.find((entry) => entry.id === "device")!.flash).toBe(2);
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

  it("uses a wide swaying shot for the stream and closer shots for the slow steps", () => {
    expect(showCamera("stream", null).focusId).toBeNull();
    expect(showCamera("stream", null).sway).toBeGreaterThan(0);
    expect(showCamera("split", null).focusId).toBe("device");
    expect(showCamera("route", null).focusId).toBeNull();
    expect(showCamera("compute", null).focusId).toBe("worker-1");
  });
});
