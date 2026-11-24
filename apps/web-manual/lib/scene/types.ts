/** Plain description of a 3D scene. Labs build one from their simulation state; the viewer draws it. */
export type SceneTone = "plain" | "accent" | "good" | "warn" | "muted";
export type Vec3 = readonly [number, number, number];

export interface SceneNode {
  id: string;
  /** box: sized block; sphere: ball of diameter size[0]; shell: translucent ball of diameter size[0]. */
  kind: "box" | "sphere" | "shell";
  position: Vec3;
  size: Vec3;
  tone: SceneTone;
  label?: string;
  active?: boolean;
}

export interface SceneLink {
  id: string;
  from: string;
  to: string;
  tone: SceneTone;
  dashed?: boolean;
  active?: boolean;
}

export interface ScenePacket {
  id: string;
  link: string;
  /** 0 at the link's start, 1 at its end. */
  progress: number;
  tone: SceneTone;
}

export interface Scene3D {
  /** One sentence saying what the picture shows in its current state; also the text alternative. */
  caption: string;
  nodes: SceneNode[];
  links: SceneLink[];
  packets: ScenePacket[];
}
