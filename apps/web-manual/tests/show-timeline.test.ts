import { describe, expect, it } from "vitest";
import { ATTRACT_ORDER, SLOW_BEATS, beatMs, currentBeat, initialShow, showReducer, tamperRoleOf, type ShowState } from "@/lib/show/timeline";

const UNITS = 5;

function tick(state: ShowState, times = 1): ShowState {
  let next = state;
  for (let index = 0; index < times; index += 1) next = showReducer(next, { type: "tick", units: UNITS });
  return next;
}

describe("show timeline", () => {
  it("plays the first unit slowly through all six steps and reveals it on the last", () => {
    let state = initialShow("text-shares", { playing: true, attract: false });
    const seen: string[] = [];
    for (let index = 0; index < SLOW_BEATS.length; index += 1) {
      seen.push(currentBeat(state, UNITS));
      if (index < SLOW_BEATS.length - 1) expect(state.revealed).toBe(0);
      state = tick(state);
    }
    expect(seen).toEqual(["split", "send", "compute", "return", "verify", "combine"]);
    // Entering "combine" revealed the first unit; the tick after it started the stream.
    expect(state.revealed).toBe(2);
    expect(currentBeat(state, UNITS)).toBe("stream");
  });

  it("then reveals one unit per beat until it is done, and stops", () => {
    let state = tick(initialShow("text-shares", { playing: true, attract: false }), SLOW_BEATS.length - 1);
    expect(state.revealed).toBe(1);
    state = tick(state, UNITS - 1);
    expect(state.revealed).toBe(UNITS);
    expect(currentBeat(state, UNITS)).toBe("done");
    expect(state.playing).toBe(true);
    state = tick(state);
    expect(state.playing).toBe(false);
    expect(state.revealed).toBe(UNITS);
  });

  it("moves on to the next model by itself only while nobody has touched it", () => {
    let state = initialShow(ATTRACT_ORDER[0]!, { playing: true, attract: true });
    state = tick(state, SLOW_BEATS.length - 1 + UNITS - 1);
    expect(currentBeat(state, UNITS)).toBe("done");
    state = tick(state);
    expect(state.modelId).toBe(ATTRACT_ORDER[1]);
    expect(state.revealed).toBe(0);
    expect(state.playing).toBe(true);
    expect(state.attract).toBe(true);
    const last = initialShow(ATTRACT_ORDER[ATTRACT_ORDER.length - 1]!, { playing: true, attract: true });
    expect(tick(last, SLOW_BEATS.length - 1 + UNITS).modelId).toBe(ATTRACT_ORDER[0]);
  });

  it("stops the automatic tour as soon as the reader uses a control", () => {
    const start = initialShow("text-shares", { playing: true, attract: true });
    for (const action of [{ type: "pause" }, { type: "play" }, { type: "next", units: UNITS }, { type: "restart" }, { type: "tamper" }, { type: "speed", speed: 2 }, { type: "select", modelId: "picture-city" }] as const) {
      expect(showReducer(start, action).attract).toBe(false);
    }
  });

  it("starts the tour without ending it", () => {
    const waiting = initialShow("text-shares", { playing: false, attract: true });
    expect(showReducer(waiting, { type: "autostart" })).toMatchObject({ playing: true, attract: true });
  });

  it("steps one unit at a time when asked, without playing", () => {
    let state = initialShow("text-shares", { playing: false, attract: false });
    state = showReducer(state, { type: "next", units: UNITS });
    expect(state.revealed).toBe(1);
    expect(state.playing).toBe(false);
    expect(currentBeat(state, UNITS)).toBe("combine");
    state = showReducer(state, { type: "next", units: UNITS });
    expect(state.revealed).toBe(2);
    expect(currentBeat(state, UNITS)).toBe("stream");
    for (let index = 0; index < 10; index += 1) state = showReducer(state, { type: "next", units: UNITS });
    expect(state.revealed).toBe(UNITS);
    expect(currentBeat(state, UNITS)).toBe("done");
  });

  it("turns the next reveal into a rejection when a worker is tampered with, then recovers", () => {
    let state = tick(initialShow("text-shares", { playing: true, attract: false }), SLOW_BEATS.length - 1);
    state = showReducer(state, { type: "tamper" });
    expect(state.tamperArmed).toBe(true);
    state = tick(state);
    expect(state.rejecting).toBe(true);
    expect(state.rejections).toBe(1);
    expect(state.revealed).toBe(1);
    expect(currentBeat(state, UNITS)).toBe("rejected");
    expect(tamperRoleOf(state)).toBe(2);
    expect(state.tamperArmed).toBe(false);
    state = tick(state);
    expect(state.rejecting).toBe(false);
    expect(tamperRoleOf(state)).toBeNull();
    expect(state.revealed).toBe(2);
    expect(state.rejections).toBe(1);
  });

  it("restarts and switches models from a clean state", () => {
    const played = tick(initialShow("text-shares", { playing: true, attract: false }), 8);
    const again = showReducer(played, { type: "restart" });
    expect(again).toMatchObject({ modelId: "text-shares", unit: 0, beat: 0, revealed: 0, rejections: 0, playing: true });
    expect(again.run).toBe(played.run + 1);
    const other = showReducer(played, { type: "select", modelId: "clip-sunrise" });
    expect(other).toMatchObject({ modelId: "clip-sunrise", revealed: 0, playing: true });
    expect(other.run).toBe(played.run + 1);
    expect(showReducer(showReducer(played, { type: "pause" }), { type: "select", modelId: "clip-sunrise", playing: false }).playing).toBe(false);
  });

  it("paces slow steps, streams and the closing hold, and halves them at double speed", () => {
    const start = initialShow("text-shares", { playing: true, attract: false });
    expect(beatMs(start, "text", UNITS)).toBeGreaterThanOrEqual(900);
    const stream = tick(start, SLOW_BEATS.length);
    expect(beatMs(stream, "text", UNITS)).toBeLessThan(beatMs(stream, "picture", UNITS));
    expect(beatMs(showReducer(stream, { type: "speed", speed: 2 }), "picture", UNITS)).toBe(beatMs(stream, "picture", UNITS) / 2);
    const done = tick(start, SLOW_BEATS.length - 1 + UNITS - 1);
    expect(beatMs(done, "clip", UNITS)).toBeGreaterThan(beatMs(done, "text", UNITS));
    expect(beatMs(showReducer(done, { type: "speed", speed: 2 }), "text", UNITS)).toBe(beatMs(done, "text", UNITS));
  });
});
