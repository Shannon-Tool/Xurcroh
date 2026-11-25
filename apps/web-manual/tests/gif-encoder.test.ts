// @vitest-environment node
import { describe, expect, it } from "vitest";
import { decodeGif, encodeGif } from "@/scripts/lib/gif.mjs";

const COLOURS = { black: [0, 0, 0], red: [255, 0, 0], green: [0, 255, 0], blue: [0, 0, 255], white: [255, 255, 255] } as const;

function picture(width: number, height: number, paint: (x: number, y: number) => readonly number[]): Uint8Array {
  const rgba = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const [r, g, b] = paint(x, y);
      rgba.set([r!, g!, b!, 255], (y * width + x) * 4);
    }
  }
  return rgba;
}

const W = 12;
const H = 8;
const stripes = picture(W, H, (x) => (x < 4 ? COLOURS.red : x < 8 ? COLOURS.green : COLOURS.blue));
const moved = picture(W, H, (x, y) => (x === 5 && y === 3 ? COLOURS.white : x < 4 ? COLOURS.red : x < 8 ? COLOURS.green : COLOURS.blue));
const other = picture(W, H, (x, y) => ((x + y) % 2 === 0 ? COLOURS.black : COLOURS.white));

describe("GIF encoder for the recordings", () => {
  it("writes a looping GIF89a that decodes back to the frames it was given", () => {
    const bytes = encodeGif({ width: W, height: H, frames: [{ rgba: stripes, delayMs: 100 }, { rgba: other, delayMs: 200 }, { rgba: moved, delayMs: 100 }] });
    expect(new TextDecoder().decode(bytes.subarray(0, 6))).toBe("GIF89a");
    expect(bytes[bytes.length - 1]).toBe(0x3b);
    const gif = decodeGif(bytes);
    expect([gif.width, gif.height, gif.loops]).toEqual([W, H, 0]);
    expect(gif.frames.map((frame) => frame.delayMs)).toEqual([100, 200, 100]);
    expect(Array.from(gif.frames[0]!.rgba)).toEqual(Array.from(stripes));
    expect(Array.from(gif.frames[1]!.rgba)).toEqual(Array.from(other));
    expect(Array.from(gif.frames[2]!.rgba)).toEqual(Array.from(moved));
  });

  it("stores only the pixels that changed since the frame before", () => {
    const gif = decodeGif(encodeGif({ width: W, height: H, frames: [{ rgba: stripes, delayMs: 100 }, { rgba: moved, delayMs: 100 }] }));
    expect(gif.frames[1]!.changed).toBe(1);
    expect(Array.from(gif.frames[1]!.rgba)).toEqual(Array.from(moved));
  });

  it("merges frames that show nothing new into a longer pause, keeping the total time", () => {
    const gif = decodeGif(encodeGif({ width: W, height: H, frames: [stripes, stripes, stripes, moved, moved].map((rgba) => ({ rgba, delayMs: 100 })) }));
    expect(gif.frames.map((frame) => frame.delayMs)).toEqual([300, 200]);
  });

  it("can ignore changes too small to see, without letting them pile up", () => {
    const faint = (step: number) => picture(W, H, () => [100 + step, 100, 100]);
    const frames = [0, 1, 2, 3, 4, 5, 6, 7, 8].map((step) => ({ rgba: faint(step * 2), delayMs: 100 }));
    const gif = decodeGif(encodeGif({ width: W, height: H, frames, fuzz: 6 }));
    // Each step is within the tolerance of the one before, but the drift from what is shown is not ignored for ever.
    expect(gif.frames.length).toBeGreaterThan(1);
    expect(gif.frames.length).toBeLessThan(frames.length);
    const last = gif.frames[gif.frames.length - 1]!.rgba;
    expect(Math.abs(last[0]! - 116)).toBeLessThanOrEqual(8);
  });

  it("refuses frames of the wrong size", () => {
    expect(() => encodeGif({ width: W, height: H, frames: [{ rgba: new Uint8Array(12), delayMs: 100 }] })).toThrow(/size/);
    expect(() => encodeGif({ width: W, height: H, frames: [] })).toThrow(/frame/);
  });
});
