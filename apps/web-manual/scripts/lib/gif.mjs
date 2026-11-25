// GIF writing and reading for the manual's recordings. Pure JavaScript, no system tools.
//
// One palette is used for a whole recording, so colours do not flicker between frames, and each
// frame stores only the pixels that differ from what is already on screen.
import * as gifenc from "gifenc";

// Node loads the package's CommonJS build (everything on the default export); bundlers load its ES build.
const { GIFEncoder, applyPalette, quantize } = "quantize" in gifenc ? gifenc : gifenc.default;

const FORMAT = "rgb565";

/** A palette for a recording, taken from some of its frames. One entry is kept free for "unchanged". */
export function paletteFrom(samples, colors = 256) {
  if (samples.length === 0) throw new Error("A recording needs at least one frame");
  const pixels = samples.reduce((sum, sample) => sum + sample.length / 4, 0);
  // Two million sampled pixels are plenty to choose 255 colours from.
  const stride = Math.max(1, Math.ceil(pixels / 2_000_000));
  const picked = new Uint8Array(samples.reduce((sum, sample) => sum + Math.ceil(sample.length / 4 / stride), 0) * 4);
  let offset = 0;
  for (const sample of samples) {
    for (let pixel = 0; pixel < sample.length / 4; pixel += stride) {
      picked[offset] = sample[pixel * 4];
      picked[offset + 1] = sample[pixel * 4 + 1];
      picked[offset + 2] = sample[pixel * 4 + 2];
      picked[offset + 3] = 255;
      offset += 4;
    }
  }
  return quantize(picked.subarray(0, offset), Math.max(2, Math.min(256, colors) - 1), { format: FORMAT });
}

/**
 * Writes frames one at a time. A pixel is stored only if it differs from what the viewer already sees;
 * with `fuzz` above zero, a difference of at most that much per colour channel from what is shown is left alone.
 * With `settleEvery` above zero, every that-many-th frame is compared with a quarter of the tolerance, which
 * clears the faint leftovers the tolerance would otherwise keep on screen.
 */
export function createGifWriter({ width, height, palette, fuzz = 0, settleEvery = 0 }) {
  const count = width * height;
  const transparentIndex = palette.length;
  if (transparentIndex > 255) throw new Error("The palette must leave one entry free");
  const table = [...palette.map((colour) => colour.slice(0, 3)), [0, 0, 0]];
  const gif = GIFEncoder();
  const shown = new Uint8Array(count);
  let pending = null;
  let written = 0;
  let added = 0;

  const flush = () => {
    if (!pending) return;
    const first = written === 0;
    gif.writeFrame(pending.index, width, height, {
      palette: first ? table : undefined,
      delay: pending.delayMs,
      transparent: !first,
      transparentIndex,
      dispose: 1,
      repeat: 0,
    });
    written += 1;
    pending = null;
  };

  return {
    get written() { return written + (pending ? 1 : 0); },
    add(rgba, delayMs) {
      if (rgba.length !== count * 4) throw new Error(`Frame size ${rgba.length / 4} pixels does not match ${width}×${height}`);
      const index = applyPalette(rgba, palette, FORMAT);
      if (written === 0 && !pending) {
        shown.set(index);
        pending = { index, delayMs };
        return;
      }
      added += 1;
      const allowed = settleEvery > 0 && added % settleEvery === 0 ? Math.floor(fuzz / 4) : fuzz;
      let changed = 0;
      for (let pixel = 0; pixel < count; pixel += 1) {
        const before = shown[pixel];
        if (index[pixel] === before) { index[pixel] = transparentIndex; continue; }
        if (allowed > 0) {
          const colour = palette[before];
          const at = pixel * 4;
          if (Math.abs(rgba[at] - colour[0]) <= allowed && Math.abs(rgba[at + 1] - colour[1]) <= allowed && Math.abs(rgba[at + 2] - colour[2]) <= allowed) {
            index[pixel] = transparentIndex;
            continue;
          }
        }
        shown[pixel] = index[pixel];
        changed += 1;
      }
      if (changed === 0) { pending.delayMs += delayMs; return; }
      flush();
      pending = { index, delayMs };
    },
    finish() {
      if (!pending && written === 0) throw new Error("A recording needs at least one frame");
      flush();
      gif.finish();
      return gif.bytes();
    },
  };
}

/** Encodes a whole recording held in memory. */
export function encodeGif({ width, height, frames, colors = 256, fuzz = 0, settleEvery = 0 }) {
  if (frames.length === 0) throw new Error("A recording needs at least one frame");
  for (const frame of frames) {
    if (frame.rgba.length !== width * height * 4) throw new Error(`Frame size ${frame.rgba.length / 4} pixels does not match ${width}×${height}`);
  }
  const every = Math.max(1, Math.floor(frames.length / 12));
  const writer = createGifWriter({ width, height, fuzz, settleEvery, palette: paletteFrom(frames.filter((_, index) => index % every === 0).map((frame) => frame.rgba), colors) });
  for (const frame of frames) writer.add(frame.rgba, frame.delayMs);
  return writer.finish();
}

/** The facts about a GIF that can be read without unpacking its pictures: size, frames, length, looping. */
export function inspectGif(bytes) {
  const signature = String.fromCharCode(...bytes.subarray(0, 6));
  if (signature !== "GIF89a" && signature !== "GIF87a") throw new Error("Not a GIF");
  const u16 = (at) => bytes[at] | (bytes[at + 1] << 8);
  let at = 13 + (bytes[10] & 0x80 ? 3 * (1 << ((bytes[10] & 7) + 1)) : 0);
  const skipBlocks = () => {
    while (at < bytes.length && bytes[at] !== 0) at += 1 + bytes[at];
    if (at >= bytes.length) throw new Error("The GIF is cut short");
    at += 1;
  };
  let frames = 0;
  let durationMs = 0;
  let loops = null;
  let ended = false;
  while (at < bytes.length && !ended) {
    const block = bytes[at];
    at += 1;
    if (block === 0x3b) { ended = true; break; }
    if (block === 0x21) {
      const label = bytes[at];
      at += 1;
      if (label === 0xf9) { durationMs += u16(at + 2) * 10; at += 6; continue; }
      if (label === 0xff && String.fromCharCode(...bytes.subarray(at + 1, at + 12)) === "NETSCAPE2.0") loops = u16(at + 14);
      skipBlocks();
      continue;
    }
    if (block !== 0x2c) throw new Error(`Unexpected block ${block} at ${at - 1}`);
    const flags = bytes[at + 8];
    at += 9 + (flags & 0x80 ? 3 * (1 << ((flags & 7) + 1)) : 0) + 1;
    skipBlocks();
    frames += 1;
  }
  if (!ended) throw new Error("The GIF is cut short");
  return { width: u16(6), height: u16(8), frames, durationMs, loops };
}

/** Reads a GIF back into the pictures a viewer sees, frame by frame. Used to check the recordings. */
export function decodeGif(bytes) {
  const signature = String.fromCharCode(...bytes.subarray(0, 6));
  if (signature !== "GIF89a" && signature !== "GIF87a") throw new Error("Not a GIF");
  const u16 = (at) => bytes[at] | (bytes[at + 1] << 8);
  const width = u16(6);
  const height = u16(8);
  let at = 13;
  let globalTable = null;
  if (bytes[10] & 0x80) {
    const size = 3 * (1 << ((bytes[10] & 7) + 1));
    globalTable = bytes.subarray(at, at + size);
    at += size;
  }
  const canvas = new Uint8Array(width * height * 4);
  const frames = [];
  let loops = null;
  let control = { delayMs: 0, transparent: -1, dispose: 0 };

  const subBlocks = () => {
    const parts = [];
    let total = 0;
    while (bytes[at] !== 0) {
      if (at >= bytes.length) throw new Error("The GIF is cut short");
      const size = bytes[at];
      parts.push(bytes.subarray(at + 1, at + 1 + size));
      total += size;
      at += 1 + size;
    }
    at += 1;
    const joined = new Uint8Array(total);
    let offset = 0;
    for (const part of parts) { joined.set(part, offset); offset += part.length; }
    return joined;
  };

  while (at < bytes.length) {
    const block = bytes[at];
    at += 1;
    if (block === 0x3b) break;
    if (block === 0x21) {
      const label = bytes[at];
      at += 1;
      if (label === 0xf9) {
        const flags = bytes[at + 1];
        control = { delayMs: u16(at + 2) * 10, transparent: flags & 1 ? bytes[at + 4] : -1, dispose: (flags >> 2) & 7 };
        at += 6;
      } else if (label === 0xff) {
        const name = String.fromCharCode(...bytes.subarray(at + 1, at + 12));
        at += 12;
        const data = subBlocks();
        if (name === "NETSCAPE2.0" && data[0] === 1) loops = data[1] | (data[2] << 8);
      } else {
        subBlocks();
      }
      continue;
    }
    if (block !== 0x2c) throw new Error(`Unexpected block ${block} at ${at - 1}`);
    const left = u16(at);
    const top = u16(at + 2);
    const frameWidth = u16(at + 4);
    const frameHeight = u16(at + 6);
    const flags = bytes[at + 8];
    at += 9;
    if (flags & 0x40) throw new Error("Interlaced GIFs are not supported");
    let table = globalTable;
    if (flags & 0x80) {
      const size = 3 * (1 << ((flags & 7) + 1));
      table = bytes.subarray(at, at + size);
      at += size;
    }
    if (!table) throw new Error("Frame without a colour table");
    const minimumCodeSize = bytes[at];
    at += 1;
    const indices = unpackLzw(subBlocks(), minimumCodeSize, frameWidth * frameHeight);
    let changed = 0;
    for (let y = 0; y < frameHeight; y += 1) {
      for (let x = 0; x < frameWidth; x += 1) {
        const index = indices[y * frameWidth + x];
        if (index === control.transparent) continue;
        const target = ((top + y) * width + left + x) * 4;
        canvas[target] = table[index * 3];
        canvas[target + 1] = table[index * 3 + 1];
        canvas[target + 2] = table[index * 3 + 2];
        canvas[target + 3] = 255;
        changed += 1;
      }
    }
    frames.push({ delayMs: control.delayMs, changed, rgba: canvas.slice() });
    if (control.dispose === 2) canvas.fill(0);
    control = { delayMs: 0, transparent: -1, dispose: 0 };
  }
  return { width, height, loops, frames };
}

function unpackLzw(data, minimumCodeSize, pixelCount) {
  const clear = 1 << minimumCodeSize;
  const end = clear + 1;
  const prefix = new Int32Array(4096);
  const suffix = new Uint8Array(4096);
  const stack = new Uint8Array(4097);
  const output = new Uint8Array(pixelCount);
  let codeSize = minimumCodeSize + 1;
  let available = clear + 2;
  let previous = -1;
  let first = 0;
  let bits = 0;
  let held = 0;
  let written = 0;
  let cursor = 0;
  for (let code = 0; code < clear; code += 1) suffix[code] = code;
  while (written < pixelCount) {
    while (held < codeSize && cursor < data.length) {
      bits |= data[cursor] << held;
      held += 8;
      cursor += 1;
    }
    if (held < codeSize) break;
    let code = bits & ((1 << codeSize) - 1);
    bits >>>= codeSize;
    held -= codeSize;
    if (code === clear) { codeSize = minimumCodeSize + 1; available = clear + 2; previous = -1; continue; }
    if (code === end) break;
    if (previous === -1) { output[written] = suffix[code]; written += 1; previous = code; first = code; continue; }
    const incoming = code;
    let top = 0;
    if (code >= available) { stack[top] = first; top += 1; code = previous; }
    while (code >= clear) { stack[top] = suffix[code]; top += 1; code = prefix[code]; }
    first = suffix[code];
    stack[top] = first;
    top += 1;
    if (available < 4096) {
      prefix[available] = previous;
      suffix[available] = first;
      available += 1;
      if ((available & (available - 1)) === 0 && available < 4096) codeSize += 1;
    }
    previous = incoming;
    while (top > 0 && written < pixelCount) { top -= 1; output[written] = stack[top]; written += 1; }
  }
  return output;
}
