export interface Orbit {
  /** Angle around the vertical axis, radians. */
  theta: number;
  /** Angle down from straight overhead, radians. */
  phi: number;
  /** Distance multiplier; smaller is closer. */
  zoom: number;
}

export const DEFAULT_ORBIT: Orbit = { theta: 0.6, phi: 1.05, zoom: 1 };
const MIN_PHI = 0.15;
const MAX_PHI = 1.5;
const MIN_ZOOM = 0.45;
const MAX_ZOOM = 2.5;

function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value));
}

export function rotateOrbit(orbit: Orbit, deltaTheta: number, deltaPhi: number): Orbit {
  return { ...orbit, theta: orbit.theta + deltaTheta, phi: clamp(orbit.phi + deltaPhi, MIN_PHI, MAX_PHI) };
}

export function zoomOrbit(orbit: Orbit, factor: number): Orbit {
  return { ...orbit, zoom: clamp(orbit.zoom * factor, MIN_ZOOM, MAX_ZOOM) };
}

/** Camera offset from the point it looks at, for a scene of the given radius. */
export function orbitOffset(orbit: Orbit, radius: number): [number, number, number] {
  const distance = radius * orbit.zoom;
  return [
    distance * Math.sin(orbit.phi) * Math.sin(orbit.theta),
    distance * Math.cos(orbit.phi),
    distance * Math.sin(orbit.phi) * Math.cos(orbit.theta),
  ];
}
