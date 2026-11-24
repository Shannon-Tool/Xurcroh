import type { Scene3D, SceneNode, Vec3 } from "@/lib/scene/types";
import { PROTECTION_LAYERS } from "@/lib/show/protection";

const RING = 3.2;
const OUTSIDE = 6.3;

function around(index: number, radius: number, height: number): Vec3 {
  const angle = (index / PROTECTION_LAYERS.length) * Math.PI * 2 + Math.PI / 6;
  return [Math.cos(angle) * radius, height, Math.sin(angle) * radius];
}

/** Your data in the middle, one guard post per layer around it, and an intruder facing the layer on stage. */
export function protectionScene(input: { layer: number }): Scene3D {
  const layer = Math.max(0, Math.min(PROTECTION_LAYERS.length - 1, input.layer));
  const current = PROTECTION_LAYERS[layer]!;
  const nodes: SceneNode[] = [
    { id: "ground", kind: "slab", position: [0, 0.04, 0], size: [3, 0.08, 3], tone: "good", opacity: 0.2 },
    { id: "data", kind: "orb", position: [0, 1.1, 0], size: [1, 1, 1], tone: "good", label: "Your data", detail: "What every layer is there to protect.", pulse: true },
  ];
  PROTECTION_LAYERS.forEach((entry, index) => {
    const on = index === layer;
    nodes.push({
      id: `layer-${entry.id}`, kind: "cylinder", position: around(index, RING, on ? 1.3 : 0.9), size: [1.1, on ? 2.6 : 1.8, 1.1],
      tone: on ? "accent" : index < layer ? "good" : "plain", opacity: on ? 1 : 0.55, active: on, pulse: on,
      label: `${index + 1}. ${entry.name}`, detail: `${entry.claim}. ${entry.mechanism}`,
    });
  });
  nodes.push({ id: "intruder", kind: "box", position: around(layer, OUTSIDE, 0.5), size: [1, 1, 1], tone: "warn", label: "Someone trying to get in", detail: current.threat });
  return {
    caption: `Layer ${layer + 1} of ${PROTECTION_LAYERS.length} · ${current.name}: ${current.claim.toLowerCase()}. ${current.mechanism}`,
    nodes,
    links: [
      { id: "attempt", from: "intruder", to: `layer-${current.id}`, tone: "warn", arc: 0.5, dashed: true, active: true, flow: "forward" },
      ...PROTECTION_LAYERS.map((entry, index) => ({ id: `guard-${entry.id}`, from: `layer-${entry.id}`, to: "data", tone: index === layer ? "accent" as const : "muted" as const, active: index === layer })),
    ],
    legend: [
      { tone: "good", text: "your data and the layers already shown" },
      { tone: "accent", text: "the layer on stage" },
      { tone: "warn", text: "the threat it answers" },
    ],
  };
}
