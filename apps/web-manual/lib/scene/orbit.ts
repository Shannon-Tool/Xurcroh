/** Camera position on a sphere around the scene: angle around, angle down from the top, and distance factor. */
export interface Orbit {
  theta: number;
  phi: number;
  zoom: number;
}

export const DEFAULT_ORBIT: Orbit = { theta: 0.6, phi: 1.05, zoom: 1 };

const MIN_PHI = 0.15;
const MAX_PHI = 1.5;
const MIN_ZOOM = 0.45;
const MAX_ZOOM = 2.5;

export function rotateOrbit(orbit: Orbit, deltaTheta: number, deltaPhi: number): Orbit {
  return { ...orbit, theta: orbit.theta + deltaTheta, phi: Math.max(MIN_PHI, Math.min(MAX_PHI, orbit.phi + deltaPhi)) };
}

export function zoomOrbit(orbit: Orbit, factor: number): Orbit {
  return { ...orbit, zoom: Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, orbit.zoom * factor)) };
}

export function orbitOffset(orbit: Orbit, radius: number): [number, number, number] {
  const distance = radius * orbit.zoom;
  return [
    distance * Math.sin(orbit.phi) * Math.sin(orbit.theta),
    distance * Math.cos(orbit.phi),
    distance * Math.sin(orbit.phi) * Math.cos(orbit.theta),
  ];
}

/** An orbit plus the spin a drag left behind, in radians per second. */
export interface OrbitMotion {
  orbit: Orbit;
  spinTheta: number;
  spinPhi: number;
}

const DAMPING = 4;
const REST = 0.01;
const AUTO_SPEED = 0.09;

export function restingMotion(orbit: Orbit): OrbitMotion {
  return { orbit, spinTheta: 0, spinPhi: 0 };
}

export function orbitMoving(motion: OrbitMotion): boolean {
  return motion.spinTheta !== 0 || motion.spinPhi !== 0;
}

/** Carries a thrown view on, slowing it to rest; turns it gently when auto-rotation is on. */
export function stepOrbit(motion: OrbitMotion, dt: number, autoRotate: boolean): OrbitMotion {
  const decay = Math.exp(-DAMPING * dt);
  const settle = (spin: number) => (Math.abs(spin * decay) < REST ? 0 : spin * decay);
  const turn = motion.spinTheta * dt + (autoRotate ? AUTO_SPEED * dt : 0);
  const orbit = turn === 0 && motion.spinPhi === 0 ? motion.orbit : rotateOrbit(motion.orbit, turn, motion.spinPhi * dt);
  return { orbit, spinTheta: settle(motion.spinTheta), spinPhi: settle(motion.spinPhi) };
}

/** A shot chosen by a scene's director: where the view should settle while the reader is not steering. */
export interface Direction {
  theta: number;
  phi: number;
  zoom: number;
  /** Node to centre on, or null for the whole scene. */
  focusId: string | null;
  /** How far, in radians, the view drifts either side of `theta`. */
  sway: number;
}

/** Moves `orbit` a fraction of the way to the director's shot; `factor` 1 lands on it. */
export function followDirection(orbit: Orbit, direction: Direction, time: number, factor: number): Orbit {
  const theta = direction.theta + direction.sway * Math.sin(time * 0.35);
  if (factor >= 1) return { theta, phi: direction.phi, zoom: direction.zoom };
  return {
    theta: orbit.theta + (theta - orbit.theta) * factor,
    phi: orbit.phi + (direction.phi - orbit.phi) * factor,
    zoom: orbit.zoom + (direction.zoom - orbit.zoom) * factor,
  };
}
