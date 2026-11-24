export type SceneTone = "plain" | "accent" | "good" | "warn" | "muted";
export type Vec3 = readonly [number, number, number];

/**
 * box / slab: size is width, height, depth (a slab is a see-through platform).
 * sphere / shell / orb: size[0] is the diameter (a shell is glass, an orb glows).
 * cylinder: size[0] is the diameter, size[1] the height.
 * ring: size[0] is the diameter; it lies flat.
 * arrow: runs from `position` to `tip`; size[0] is the shaft thickness.
 * screen: a flat panel that turns to face the viewer; size is width, height. It shows the picture named by `screen`.
 */
export type SceneNodeKind = "box" | "slab" | "sphere" | "shell" | "orb" | "cylinder" | "ring" | "arrow" | "screen";

export interface SceneNode {
  id: string;
  kind: SceneNodeKind;
  position: Vec3;
  size: Vec3;
  tone: SceneTone;
  label?: string;
  /** Longer explanation shown when the reader points at the node. */
  detail?: string;
  active?: boolean;
  /** 0–1; defaults depend on the kind. */
  opacity?: number;
  /** Arrow head position. */
  tip?: Vec3;
  /** Rotation about the depth axis, in radians. */
  tilt?: number;
  /** Slowly throbs to draw the eye (still under reduced motion). */
  pulse?: boolean;
  /** Name of the picture a screen shows; the viewer is handed the picture separately. */
  screen?: string;
  /** A counter: each time it changes, the node flashes once (not under reduced motion). */
  flash?: number;
}

export interface SceneLink {
  id: string;
  from: string;
  to: string;
  tone: SceneTone;
  dashed?: boolean;
  active?: boolean;
  /** How far the middle of the link rises above a straight line. */
  arc?: number;
  /** Direction in which light travels along the link. */
  flow?: "forward" | "back";
}

export interface SceneLegendEntry {
  tone: SceneTone;
  text: string;
}

/** Plain description of one 3D drawing. Builders return it; the stage draws it. */
export interface Scene3D {
  caption: string;
  nodes: SceneNode[];
  links: SceneLink[];
  legend?: SceneLegendEntry[];
}
