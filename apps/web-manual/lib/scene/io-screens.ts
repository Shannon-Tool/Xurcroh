import type { Scene3D, SceneNode, SceneTone } from "@/lib/scene/types";

export const IO_SCREENS = ["io-in", "io-seen", "io-out"] as const;

/**
 * Adds three screens behind a lab's scene: what goes in, what the other party sees, what comes out,
 * joined by links that carry light from one to the next. The viewer is handed their pictures by name.
 */
export function withIOScreens(scene: Scene3D, outputTone: SceneTone = "good"): Scene3D {
  let minX = Infinity;
  let maxX = -Infinity;
  let minZ = Infinity;
  let maxY = 0;
  for (const node of scene.nodes) {
    maxY = Math.max(maxY, node.position[1] + (node.kind === "box" || node.kind === "slab" || node.kind === "cylinder" ? node.size[1] : node.size[0]) / 2, node.tip ? node.tip[1] : 0);
    const halfX = node.size[0] / 2;
    const halfZ = (node.kind === "box" || node.kind === "slab" ? node.size[2] : node.size[0]) / 2;
    minX = Math.min(minX, node.position[0] - halfX, node.tip ? node.tip[0] : Infinity);
    maxX = Math.max(maxX, node.position[0] + halfX, node.tip ? node.tip[0] : -Infinity);
    minZ = Math.min(minZ, node.position[2] - halfZ, node.tip ? node.tip[2] : Infinity);
  }
  if (!Number.isFinite(minX)) {
    minX = -3;
    maxX = 3;
    minZ = 0;
  }
  const centre = (minX + maxX) / 2;
  const gap = Math.max(4.3, (maxX - minX) / 2.2);
  const z = minZ - 1.8;
  // High enough to clear the scene in front of them.
  const y = Math.max(2, Math.min(maxY, 4) * 0.6 + 1.5);
  const screen = (id: string, index: number, tone: SceneTone, label: string, detail: string): SceneNode => ({
    id, kind: "screen", screen: id, position: [centre + (index - 1) * gap, y, z], size: [3.7, 2.3, 0], tone, label, detail,
  });
  return {
    ...scene,
    nodes: [
      ...scene.nodes,
      screen("io-in", 0, "good", "Goes in", "What you put in. It stays on your device."),
      screen("io-seen", 1, "accent", "Others see", "What the other party holds while it works."),
      screen("io-out", 2, outputTone, "Comes out", "What you get back."),
    ],
    links: [
      ...scene.links,
      { id: "io-flow-a", from: "io-in", to: "io-seen", tone: "accent", arc: 0.9, flow: "forward" },
      { id: "io-flow-b", from: "io-seen", to: "io-out", tone: outputTone, arc: 0.9, flow: "forward" },
    ],
  };
}
