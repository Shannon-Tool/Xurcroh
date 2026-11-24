/** Glyphs used while text is turning into something else. Hexadecimal, because that is what bytes look like. */
const GLYPHS = "0123456789abcdef";

/**
 * One frame of text turning from `from` into `to`. Characters settle from the left; a short band of
 * changing glyphs runs ahead of them. `progress` runs from 0 (all `from`) to 1 (all `to`).
 */
export function scrambleFrame(from: string, to: string, progress: number, tick: number): string {
  const amount = Math.max(0, Math.min(1, progress));
  if (amount >= 1) return to;
  if (amount <= 0) return from;
  const length = Math.max(from.length, to.length);
  const settled = Math.floor(amount * length);
  const band = Math.max(2, Math.ceil(length * 0.18));
  let out = "";
  for (let index = 0; index < length; index += 1) {
    if (index < settled) out += to[index] ?? "";
    else if (index < settled + band) out += (to[index] ?? from[index]) === " " ? " " : GLYPHS[(index * 7 + tick * 5 + settled) % GLYPHS.length];
    else out += from[index] ?? "";
  }
  return out;
}

/** Bytes or numbers as a short run of hexadecimal, the way sealed or split data is shown on the stage. */
export function hexRun(values: ArrayLike<number>, digits: number, count: number): string {
  const shown = Math.min(count, values.length);
  let out = "";
  for (let index = 0; index < shown; index += 1) out += `${index > 0 ? " " : ""}${(values[index]! >>> 0).toString(16).padStart(digits, "0").slice(-digits)}`;
  return out;
}
