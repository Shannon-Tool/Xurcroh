import type { Vec3 } from "@/lib/scene/types";

/** Differences smaller than this are treated as arrived, so scenes come to rest. */
export const SETTLE = 1e-3;

/** Fraction of the remaining distance covered in `dt` seconds; `Infinity` means jump straight there. */
export function blendFactor(dt: number, rate: number): number {
  if (!Number.isFinite(dt)) return 1;
  return 1 - Math.exp(-rate * Math.max(0, dt));
}

export function approach(current: number, target: number, factor: number): number {
  const next = current + (target - current) * factor;
  return Math.abs(target - next) < SETTLE ? target : next;
}

export function approachVec(current: Vec3, target: Vec3, factor: number): Vec3 {
  return [approach(current[0], target[0], factor), approach(current[1], target[1], factor), approach(current[2], target[2], factor)];
}

export function sameVec(a: Vec3, b: Vec3): boolean {
  return a[0] === b[0] && a[1] === b[1] && a[2] === b[2];
}

function fraction(value: number): number {
  const result = value - Math.floor(value);
  return result >= 1 ? 0 : result;
}

/** Positions (0–1 along a link) of `count` evenly spaced packets at `time` seconds. */
export function flowOffsets(time: number, count: number, speed: number, direction: 1 | -1): number[] {
  return Array.from({ length: count }, (_, index) => fraction(index / count + direction * time * speed));
}

/** Point on a link: a straight line when `arc` is 0, otherwise a curve whose middle rises by `arc / 2`. */
export function curvePoint(a: Vec3, b: Vec3, arc: number, t: number): Vec3 {
  const u = 1 - t;
  const control: Vec3 = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + arc, (a[2] + b[2]) / 2];
  const mix = (index: 0 | 1 | 2) => u * u * a[index] + 2 * u * t * control[index] + t * t * b[index];
  return [mix(0), mix(1), mix(2)];
}
