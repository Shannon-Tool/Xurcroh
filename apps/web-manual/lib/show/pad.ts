import { PAD } from "@/lib/simulation/show";

/** Ready-made drawings for the pad, so the demonstration has something to work on before the reader draws. */
export type PadPreset = "smile" | "arrow" | "blank";

export function presetInk(preset: PadPreset): Uint8ClampedArray {
  const { width, height } = PAD;
  const ink = new Uint8ClampedArray(width * height);
  if (preset === "blank") return ink;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let on = false;
      if (preset === "smile") {
        const dx = x - 15.5;
        const dy = y - 9.5;
        const radius = Math.hypot(dx, dy);
        on = Math.abs(radius - 8) < 0.8 || (Math.abs(dx) > 2.2 && Math.abs(dx) < 4.2 && y >= 6 && y <= 7) || (Math.abs(Math.hypot(dx, dy + 1) - 5) < 0.7 && dy > 2);
      } else {
        on = (y >= 9 && y <= 10 && x >= 5 && x <= 24) || (x >= 18 && x <= 26 && Math.abs(Math.abs(y - 9.5) - (26 - x) * 0.75) < 0.8);
      }
      if (on) ink[y * width + x] = 255;
    }
  }
  return ink;
}

/** Marks a round dab of ink at a pad pixel. Returns true when anything changed. */
export function stampInk(ink: Uint8ClampedArray, x: number, y: number): boolean {
  let changed = false;
  for (let dy = -1; dy <= 1; dy += 1) {
    for (let dx = -1; dx <= 1; dx += 1) {
      const px = x + dx;
      const py = y + dy;
      if (px < 0 || py < 0 || px >= PAD.width || py >= PAD.height || Math.abs(dx) + Math.abs(dy) > 1) continue;
      if (ink[py * PAD.width + px] !== 255) changed = true;
      ink[py * PAD.width + px] = 255;
    }
  }
  return changed;
}

/** The pad as RGB bytes for drawing it: pale ink on a dark ground. */
export function inkToRgb(ink: ArrayLike<number>): Uint8ClampedArray {
  const rgb = new Uint8ClampedArray(ink.length * 3);
  for (let index = 0; index < ink.length; index += 1) {
    const value = ink[index]!;
    rgb[index * 3] = 14 + value * 0.85;
    rgb[index * 3 + 1] = 22 + value * 0.88;
    rgb[index * 3 + 2] = 34 + value * 0.86;
  }
  return rgb;
}
