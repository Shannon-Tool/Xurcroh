import * as THREE from "three";
import { orbitOffset, type Orbit } from "@/lib/scene/orbit";
import type { Scene3D, SceneNode, SceneTone } from "@/lib/scene/types";

export type Palette = Record<SceneTone, string> & { text: string; grid: string };

export interface LabelPoint {
  id: string;
  text: string;
  x: number;
  y: number;
  visible: boolean;
  active: boolean;
}

export interface Stage {
  update(scene: Scene3D, palette: Palette): void;
  view(orbit: Orbit, width: number, height: number): LabelPoint[];
  dispose(): void;
}

function disposeTree(root: THREE.Object3D): void {
  root.traverse((object) => {
    const mesh = object as THREE.Mesh;
    mesh.geometry?.dispose();
    const material = mesh.material;
    if (Array.isArray(material)) material.forEach((item) => item.dispose());
    else material?.dispose();
  });
}

function nodeMesh(node: SceneNode, palette: Palette): THREE.Object3D {
  const group = new THREE.Group();
  const color = new THREE.Color(palette[node.tone]);
  const geometry = node.kind === "box"
    ? new THREE.BoxGeometry(node.size[0], node.size[1], node.size[2])
    : new THREE.SphereGeometry(node.size[0] / 2, 32, 20);
  const shell = node.kind === "shell";
  const material = new THREE.MeshStandardMaterial({
    color,
    roughness: 0.55,
    metalness: 0.05,
    transparent: shell,
    opacity: shell ? (node.active ? 0.42 : 0.2) : 1,
    depthWrite: !shell,
    emissive: color,
    emissiveIntensity: node.active ? 0.55 : 0.08,
  });
  group.add(new THREE.Mesh(geometry, material));
  if (node.kind === "box") {
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(geometry),
      new THREE.LineBasicMaterial({ color: new THREE.Color(palette.text), transparent: true, opacity: node.active ? 0.95 : 0.4 }),
    );
    group.add(edges);
  }
  group.position.set(node.position[0], node.position[1], node.position[2]);
  if (node.active && !shell) group.scale.setScalar(1.08);
  return group;
}

function linkMesh(from: THREE.Vector3, to: THREE.Vector3, color: THREE.Color, dashed: boolean, active: boolean): THREE.Object3D {
  if (dashed) {
    const line = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([from, to]),
      new THREE.LineDashedMaterial({ color, dashSize: 0.18, gapSize: 0.14 }),
    );
    line.computeLineDistances();
    return line;
  }
  const direction = new THREE.Vector3().subVectors(to, from);
  const length = Math.max(direction.length(), 1e-6);
  const radius = active ? 0.055 : 0.028;
  const mesh = new THREE.Mesh(
    new THREE.CylinderGeometry(radius, radius, length, 10),
    new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: active ? 0.6 : 0.1, roughness: 0.6 }),
  );
  mesh.position.copy(from).addScaledVector(direction, 0.5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.clone().divideScalar(length));
  return mesh;
}

/** Owns the WebGL renderer for one canvas. Draws only when asked, so an idle page costs nothing. */
export function createStage(canvas: HTMLCanvasElement): Stage {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  const world = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 400);
  world.add(new THREE.AmbientLight(0xffffff, 1.25));
  const sun = new THREE.DirectionalLight(0xffffff, 1.7);
  sun.position.set(5, 10, 7);
  world.add(sun);

  let content = new THREE.Group();
  world.add(content);
  const center = new THREE.Vector3();
  let radius = 6;
  let anchors: { id: string; text: string; active: boolean; point: THREE.Vector3 }[] = [];

  return {
    update(scene, palette) {
      world.remove(content);
      disposeTree(content);
      content = new THREE.Group();
      const positions = new Map<string, THREE.Vector3>();
      const bounds = new THREE.Box3();
      anchors = [];
      for (const node of scene.nodes) {
        const position = new THREE.Vector3(node.position[0], node.position[1], node.position[2]);
        positions.set(node.id, position);
        const half = node.kind === "box"
          ? new THREE.Vector3(node.size[0] / 2, node.size[1] / 2, node.size[2] / 2)
          : new THREE.Vector3(node.size[0] / 2, node.size[0] / 2, node.size[0] / 2);
        bounds.expandByPoint(position.clone().sub(half)).expandByPoint(position.clone().add(half));
        content.add(nodeMesh(node, palette));
        if (node.label) anchors.push({ id: node.id, text: node.label, active: node.active === true, point: position.clone().add(new THREE.Vector3(0, half.y + 0.22, 0)) });
      }
      const ends = new Map<string, [THREE.Vector3, THREE.Vector3]>();
      for (const link of scene.links) {
        const from = positions.get(link.from);
        const to = positions.get(link.to);
        if (!from || !to) continue;
        ends.set(link.id, [from, to]);
        content.add(linkMesh(from, to, new THREE.Color(palette[link.tone]), link.dashed === true, link.active === true));
      }
      for (const packet of scene.packets) {
        const pair = ends.get(packet.link);
        if (!pair) continue;
        const color = new THREE.Color(palette[packet.tone]);
        const ball = new THREE.Mesh(
          new THREE.SphereGeometry(0.13, 20, 14),
          new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.9 }),
        );
        ball.position.lerpVectors(pair[0], pair[1], Math.max(0, Math.min(1, packet.progress)));
        content.add(ball);
      }
      if (bounds.isEmpty()) bounds.set(new THREE.Vector3(-1, 0, -1), new THREE.Vector3(1, 1, 1));
      const size = bounds.getSize(new THREE.Vector3());
      bounds.getCenter(center);
      radius = Math.max(size.length() * 1.15, 4);
      const span = Math.max(4, Math.ceil(Math.max(size.x, size.z) + 2));
      const grid = new THREE.GridHelper(span, span, new THREE.Color(palette.grid), new THREE.Color(palette.grid));
      grid.position.set(center.x, bounds.min.y - 0.02, center.z);
      const gridMaterial = grid.material as THREE.Material;
      gridMaterial.transparent = true;
      gridMaterial.opacity = 0.45;
      content.add(grid);
      world.add(content);
    },
    view(orbit, width, height) {
      const safeWidth = Math.max(1, width);
      const safeHeight = Math.max(1, height);
      renderer.setSize(safeWidth, safeHeight, false);
      camera.aspect = safeWidth / safeHeight;
      camera.updateProjectionMatrix();
      const offset = orbitOffset(orbit, radius);
      camera.position.set(center.x + offset[0], center.y + offset[1], center.z + offset[2]);
      camera.lookAt(center);
      camera.updateMatrixWorld();
      renderer.render(world, camera);
      return anchors.map((anchor) => {
        const projected = anchor.point.clone().project(camera);
        return {
          id: anchor.id,
          text: anchor.text,
          active: anchor.active,
          x: (projected.x * 0.5 + 0.5) * safeWidth,
          y: (-projected.y * 0.5 + 0.5) * safeHeight,
          visible: projected.z < 1 && Math.abs(projected.x) <= 1.05 && Math.abs(projected.y) <= 1.05,
        };
      });
    },
    dispose() {
      disposeTree(content);
      renderer.dispose();
    },
  };
}
