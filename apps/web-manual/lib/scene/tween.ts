import { approach, approachVec, blendFactor, sameVec } from "@/lib/scene/motion";
import type { Scene3D, SceneLink, SceneNode, SceneNodeKind, SceneTone, Vec3 } from "@/lib/scene/types";

/** A node as it is drawn right now, part-way between the scene it came from and its target. */
export interface LiveNode {
  id: string;
  kind: SceneNodeKind;
  position: Vec3;
  size: Vec3;
  tip: Vec3;
  tone: SceneTone;
  label?: string;
  detail?: string;
  opacity: number;
  tilt: number;
  pulse: boolean;
  screen?: string;
  flash: number;
  /** 0 = not there yet / gone, 1 = fully present. May start below 0 so that nodes arrive one after another. */
  presence: number;
  /** 0 = ordinary, 1 = the active element of the current step. */
  emphasis: number;
  leaving: boolean;
}

export interface LiveLink {
  id: string;
  a: Vec3;
  b: Vec3;
  tone: SceneTone;
  dashed: boolean;
  arc: number;
  flow: 0 | 1 | -1;
  presence: number;
  emphasis: number;
  leaving: boolean;
}

export interface LiveScene {
  nodes: LiveNode[];
  links: LiveLink[];
  /** True when nothing is still moving towards its target. */
  settled: boolean;
}

export const EMPTY_LIVE: LiveScene = { nodes: [], links: [], settled: true };

const RATE = 7;
const GONE = 0.02;
const STAGGER = 0.05;

function defaultOpacity(kind: SceneNodeKind): number {
  if (kind === "shell") return 0.22;
  if (kind === "slab") return 0.3;
  return 1;
}

function nodeTarget(node: SceneNode) {
  return {
    opacity: node.opacity ?? defaultOpacity(node.kind),
    tilt: node.tilt ?? 0,
    tip: node.tip ?? node.position,
    emphasis: node.active ? 1 : 0,
  };
}

function moveNode(current: LiveNode, node: SceneNode, factor: number): LiveNode {
  const target = nodeTarget(node);
  return {
    id: node.id,
    kind: node.kind,
    position: approachVec(current.position, node.position, factor),
    size: approachVec(current.size, node.size, factor),
    tip: approachVec(current.tip, target.tip, factor),
    tone: node.tone,
    label: node.label,
    detail: node.detail,
    opacity: approach(current.opacity, target.opacity, factor),
    tilt: approach(current.tilt, target.tilt, factor),
    pulse: node.pulse === true,
    screen: node.screen,
    flash: node.flash ?? 0,
    presence: approach(current.presence, 1, factor),
    emphasis: approach(current.emphasis, target.emphasis, factor),
    leaving: false,
  };
}

function enterNode(node: SceneNode, order: number, factor: number): LiveNode {
  const target = nodeTarget(node);
  const start: LiveNode = {
    id: node.id,
    kind: node.kind,
    position: node.position,
    size: node.size,
    tip: target.tip,
    tone: node.tone,
    label: node.label,
    detail: node.detail,
    opacity: target.opacity,
    tilt: target.tilt,
    pulse: node.pulse === true,
    screen: node.screen,
    flash: node.flash ?? 0,
    presence: -Math.min(order * STAGGER, 0.6),
    emphasis: target.emphasis,
    leaving: false,
  };
  return moveNode(start, node, factor);
}

function nodeSettled(live: LiveNode, node: SceneNode): boolean {
  const target = nodeTarget(node);
  return live.presence === 1 && live.emphasis === target.emphasis && live.opacity === target.opacity && live.tilt === target.tilt
    && sameVec(live.position, node.position) && sameVec(live.size, node.size) && sameVec(live.tip, target.tip);
}

function flowOf(link: SceneLink): 0 | 1 | -1 {
  if (link.flow === "forward") return 1;
  if (link.flow === "back") return -1;
  return 0;
}

/**
 * Moves the drawn scene a little closer to `target`. Kept nodes glide, new ones grow in, removed ones
 * fade out and are then dropped. Pass `dt = Infinity` to show the target at once (reduced motion).
 */
export function advanceScene(live: LiveScene, target: Scene3D, dt: number): LiveScene {
  const factor = blendFactor(dt, RATE);
  const snap = factor === 1;
  let settled = true;

  const previousNodes = new Map(live.nodes.map((node) => [node.id, node]));
  const wanted = new Set(target.nodes.map((node) => node.id));
  const nodes: LiveNode[] = [];
  let entering = 0;
  for (const node of target.nodes) {
    const previous = previousNodes.get(node.id);
    const next = previous && previous.kind === node.kind ? moveNode(previous, node, factor) : enterNode(node, snap ? 0 : entering++, factor);
    if (!nodeSettled(next, node)) settled = false;
    nodes.push(next);
  }
  for (const previous of live.nodes) {
    if (wanted.has(previous.id)) continue;
    const presence = approach(Math.min(previous.presence, 1), 0, factor);
    if (snap || presence < GONE) continue;
    settled = false;
    nodes.push({ ...previous, presence, leaving: true });
  }

  const places = new Map(nodes.map((node) => [node.id, node.position]));
  const previousLinks = new Map(live.links.map((link) => [link.id, link]));
  const wantedLinks = new Set(target.links.map((link) => link.id));
  const links: LiveLink[] = [];
  for (const link of target.links) {
    const a = places.get(link.from);
    const b = places.get(link.to);
    if (!a || !b) continue;
    const previous = previousLinks.get(link.id);
    const emphasisTarget = link.active ? 1 : 0;
    const arcTarget = link.arc ?? 0;
    const next: LiveLink = {
      id: link.id,
      a,
      b,
      tone: link.tone,
      dashed: link.dashed === true,
      arc: approach(previous?.arc ?? arcTarget, arcTarget, factor),
      flow: flowOf(link),
      presence: approach(previous?.presence ?? 0, 1, factor),
      emphasis: approach(previous?.emphasis ?? emphasisTarget, emphasisTarget, factor),
      leaving: false,
    };
    if (next.presence !== 1 || next.emphasis !== emphasisTarget || next.arc !== arcTarget) settled = false;
    links.push(next);
  }
  for (const previous of live.links) {
    if (wantedLinks.has(previous.id)) continue;
    const presence = approach(previous.presence, 0, factor);
    if (snap || presence < GONE) continue;
    settled = false;
    links.push({ ...previous, presence, leaving: true });
  }

  return { nodes, links, settled };
}
