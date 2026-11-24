/** `full` adds glow, soft shadows and a sharper picture; `lite` is the plain renderer for weak hardware. */
export type Quality = "full" | "lite";
export type EffectsPreference = "auto" | "on" | "off";

const SOFTWARE_RENDERERS = /swiftshader|llvmpipe|softpipe|software|basic render/i;
const SLOW_FRAME_MS = 34;
const SAMPLES_NEEDED = 30;

export function pickQuality(input: { preference: EffectsPreference; rendererName: string; slow: boolean }): Quality {
  if (input.preference === "on") return "full";
  if (input.preference === "off") return "lite";
  if (input.slow || SOFTWARE_RENDERERS.test(input.rendererName)) return "lite";
  return "full";
}

/** True when the typical recent frame took too long. One long frame (a tab switch, a compile) does not count. */
export function isSlow(frameMs: number[]): boolean {
  if (frameMs.length < SAMPLES_NEEDED) return false;
  const sorted = [...frameMs].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)]! > SLOW_FRAME_MS;
}
