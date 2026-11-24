import * as THREE from "three";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import type { LabelPoint } from "@/lib/scene/labels";
import { blendFactor, curvePoint, flowOffsets } from "@/lib/scene/motion";
import { orbitOffset, type Orbit } from "@/lib/scene/orbit";
import type { Quality } from "@/lib/scene/quality";
import type { LiveLink, LiveNode, LiveScene } from "@/lib/scene/tween";
import type { SceneTone, Vec3 } from "@/lib/scene/types";

export type Palette = Record<SceneTone, string> & { text: string; grid: string; backdropTop: string; backdropBottom: string; dark: boolean };

export interface StageView {
  orbit: Orbit;
  width: number;
  height: number;
  /** Seconds of animation time; stands still when motion is paused or reduced. */
  time: number;
  /** Seconds since the previous frame; `Infinity` jumps colours and framing straight to their targets. */
  dt: number;
  /** Node the camera should centre on, if the reader picked one. */
  focusId: string | null;
  /** Pixels at the side that panels cover: positive on the right, negative on the left. The scene is centred in what is left. */
  inset?: number;
  /** Pixels along the bottom that panels cover. */
  insetBottom?: number;
  /** Total width covered by side panels, left and right together. */
  insetCover?: number;
}

export interface StageResult {
  labels: LabelPoint[];
  /** True while something is still moving, so the caller knows to draw another frame. */
  animating: boolean;
}

export interface Stage {
  readonly rendererName: string;
  setPalette(palette: Palette): void;
  setQuality(quality: Quality): void;
  draw(live: LiveScene, view: StageView): StageResult;
  /** Id of the node under a point given in canvas pixels, from the last drawn frame. */
  pick(x: number, y: number): string | null;
  /** Gives a named screen its picture, or tells the stage that the picture has been redrawn. */
  setScreen(id: string, source: HTMLCanvasElement): void;
  dispose(): void;
}

const UP = new THREE.Vector3(0, 1, 0);
const FLASH_SECONDS = 0.75;
const MAX_FLOW_ORBS = 256;

const SHELL_VERTEX = /* glsl */ `
varying vec3 vNormal;
varying vec3 vView;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vNormal = normalize(normalMatrix * normal);
  vView = normalize(-mv.xyz);
  gl_Position = projectionMatrix * mv;
}`;
const SHELL_FRAGMENT = /* glsl */ `
uniform vec3 uColor;
uniform float uOpacity;
uniform float uGlow;
varying vec3 vNormal;
varying vec3 vView;
void main() {
  float rim = pow(1.0 - abs(dot(normalize(vNormal), normalize(vView))), 2.2);
  float alpha = clamp(uOpacity * (0.45 + 2.4 * rim), 0.0, 1.0);
  gl_FragColor = vec4(uColor * (1.0 + uGlow * (0.4 + rim * 1.6)), alpha);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`;

// The tube is built with radius 1 and pulled in here, so its thickness can change without rebuilding it.
const LINK_VERTEX = /* glsl */ `
uniform float uRadius;
varying float vAlong;
void main() {
  vAlong = uv.x;
  vec3 p = position - normal * (1.0 - uRadius);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
}`;
const LINK_FRAGMENT = /* glsl */ `
uniform vec3 uColor;
uniform float uOpacity;
uniform float uDashes;
uniform float uTime;
uniform float uFlow;
uniform float uGlow;
uniform float uReveal;
varying float vAlong;
void main() {
  if (vAlong > uReveal) discard;
  if (uDashes > 0.0 && fract(vAlong * uDashes) > 0.55) discard;
  float wave = uFlow == 0.0 ? 0.0 : smoothstep(0.55, 1.0, fract(vAlong * 2.0 - uTime * uFlow * 0.5));
  gl_FragColor = vec4(uColor * (uGlow + wave * 0.9), uOpacity);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`;

const GRID_VERTEX = /* glsl */ `
varying vec3 vWorld;
void main() {
  vec4 world = modelMatrix * vec4(position, 1.0);
  vWorld = world.xyz;
  gl_Position = projectionMatrix * viewMatrix * world;
}`;
const GRID_FRAGMENT = /* glsl */ `
uniform vec3 uColor;
uniform vec3 uCenter;
uniform float uRadius;
uniform float uOpacity;
varying vec3 vWorld;
float gridLine(vec2 coord) {
  vec2 cell = abs(fract(coord - 0.5) - 0.5) / fwidth(coord);
  return 1.0 - min(min(cell.x, cell.y), 1.0);
}
void main() {
  float fine = gridLine(vWorld.xz);
  float coarse = gridLine(vWorld.xz / 4.0);
  float fade = 1.0 - smoothstep(uRadius * 0.3, uRadius * 1.15, distance(vWorld.xz, uCenter.xz));
  float alpha = (fine * 0.55 + coarse * 0.45) * fade * uOpacity;
  if (alpha < 0.003) discard;
  gl_FragColor = vec4(uColor, alpha);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`;

interface NodeVisual {
  kind: LiveNode["kind"];
  group: THREE.Group;
  mesh: THREE.Mesh;
  head: THREE.Mesh | null;
  halo: THREE.Mesh;
  color: THREE.Color;
  built: Vec3 | null;
  lastSize: Vec3;
  burst: THREE.Mesh | null;
  flashSeen: number;
  flashAt: number;
}

interface LinkVisual {
  mesh: THREE.Mesh;
  material: THREE.ShaderMaterial;
  color: THREE.Color;
  built: string;
  length: number;
}

function backdropTexture(top: string, bottom: string): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = 2;
  canvas.height = 128;
  const context = canvas.getContext("2d");
  if (context) {
    const gradient = context.createLinearGradient(0, 0, 0, canvas.height);
    gradient.addColorStop(0, top);
    gradient.addColorStop(1, bottom);
    context.fillStyle = gradient;
    context.fillRect(0, 0, canvas.width, canvas.height);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function ease(value: number): number {
  const t = Math.max(0, Math.min(1, value));
  return 1 - (1 - t) ** 3;
}

function halfExtent(node: LiveNode): THREE.Vector3 {
  if (node.kind === "box" || node.kind === "slab") return new THREE.Vector3(node.size[0] / 2, node.size[1] / 2, node.size[2] / 2);
  if (node.kind === "cylinder") return new THREE.Vector3(node.size[0] / 2, node.size[1] / 2, node.size[0] / 2);
  if (node.kind === "ring") return new THREE.Vector3(node.size[0] / 2, 0.05, node.size[0] / 2);
  if (node.kind === "arrow") return new THREE.Vector3(0.05, 0.05, 0.05);
  if (node.kind === "screen") return new THREE.Vector3(node.size[0] / 2, node.size[1] / 2, node.size[0] / 2);
  return new THREE.Vector3(node.size[0] / 2, node.size[0] / 2, node.size[0] / 2);
}

/**
 * Owns the WebGL renderer for one canvas and draws whatever live scene it is handed. Meshes are kept
 * between frames and only moved, so a scene can animate. Nothing here decides what a scene means.
 */
export function createStage(canvas: HTMLCanvasElement): Stage {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, powerPreference: "high-performance" });
  renderer.toneMapping = THREE.NeutralToneMapping;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  const gl = renderer.getContext();
  const debugInfo = gl.getExtension("WEBGL_debug_renderer_info");
  const rendererName = String(debugInfo ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER));

  const world = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(36, 1, 0.1, 400);
  const hemisphere = new THREE.HemisphereLight(0xffffff, 0x8899aa, 0.6);
  const sun = new THREE.DirectionalLight(0xffffff, 1.6);
  sun.shadow.mapSize.set(1024, 1024);
  sun.shadow.bias = -0.0004;
  sun.shadow.normalBias = 0.03;
  world.add(hemisphere, sun, sun.target);

  // Studio reflections are costly to prepare, so they are built the first time full quality is asked for.
  let environment: THREE.Texture | null = null;
  function studioLight(): THREE.Texture {
    if (environment) return environment;
    const pmrem = new THREE.PMREMGenerator(renderer);
    const room = new RoomEnvironment();
    environment = pmrem.fromScene(room, 0.04).texture;
    room.traverse((object) => {
      const mesh = object as THREE.Mesh;
      mesh.geometry?.dispose();
      (mesh.material as THREE.Material | undefined)?.dispose?.();
    });
    pmrem.dispose();
    return environment;
  }

  // Shared shapes: every sphere, cylinder, ring and arrow reuses these and is only scaled.
  const sphereGeometry = new THREE.SphereGeometry(0.5, 40, 28);
  const cylinderGeometry = new THREE.CylinderGeometry(0.5, 0.5, 1, 40);
  const ringGeometry = new THREE.TorusGeometry(0.5, 0.03, 12, 72).rotateX(Math.PI / 2);
  const coneGeometry = new THREE.ConeGeometry(0.5, 1, 24);
  const haloGeometry = new THREE.RingGeometry(0.42, 0.5, 64).rotateX(-Math.PI / 2);
  const panelGeometry = new THREE.PlaneGeometry(1, 1);
  const shared = [sphereGeometry, cylinderGeometry, ringGeometry, coneGeometry, haloGeometry, panelGeometry];
  const screens = new Map<string, THREE.CanvasTexture>();

  const floorGeometry = new THREE.PlaneGeometry(240, 240).rotateX(-Math.PI / 2);
  const gridMaterial = new THREE.ShaderMaterial({
    vertexShader: GRID_VERTEX,
    fragmentShader: GRID_FRAGMENT,
    transparent: true,
    depthWrite: false,
    uniforms: { uColor: { value: new THREE.Color() }, uCenter: { value: new THREE.Vector3() }, uRadius: { value: 8 }, uOpacity: { value: 0.6 } },
  });
  const grid = new THREE.Mesh(floorGeometry, gridMaterial);
  const shadowMaterial = new THREE.ShadowMaterial({ opacity: 0.22 });
  const shadowFloor = new THREE.Mesh(floorGeometry, shadowMaterial);
  shadowFloor.receiveShadow = true;
  grid.renderOrder = -3;
  shadowFloor.renderOrder = -2;
  world.add(grid, shadowFloor);

  const flowMaterial = new THREE.MeshBasicMaterial({ toneMapped: false });
  const flowOrbs = new THREE.InstancedMesh(sphereGeometry, flowMaterial, MAX_FLOW_ORBS);
  flowOrbs.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  flowOrbs.frustumCulled = false;
  flowOrbs.count = 0;
  world.add(flowOrbs);

  const content = new THREE.Group();
  world.add(content);

  let composer: EffectComposer | null = null;
  let bloom: UnrealBloomPass | null = null;
  let quality: Quality = "lite";
  let palette: Palette | null = null;
  let backdrop: THREE.Texture | null = null;
  let drawnWidth = 0;
  let drawnHeight = 0;
  let drawnInset = 0;
  let drawnInsetBottom = 0;
  let framed = false;
  let floorY = 0;
  const center = new THREE.Vector3();
  let radius = 6;
  const lookAt = new THREE.Vector3();
  const nodes = new Map<string, NodeVisual>();
  const links = new Map<string, LinkVisual>();
  const pickable: THREE.Object3D[] = [];
  const raycaster = new THREE.Raycaster();
  const matrix = new THREE.Matrix4();
  const quaternion = new THREE.Quaternion();
  const scratch = new THREE.Vector3();
  const scale = new THREE.Vector3();
  const tint = new THREE.Color();

  function toneColor(tone: SceneTone): THREE.Color {
    return tint.set(palette ? palette[tone] : "#888888");
  }

  function makeMaterial(kind: LiveNode["kind"]): THREE.Material {
    if (kind === "shell") {
      return new THREE.ShaderMaterial({
        vertexShader: SHELL_VERTEX,
        fragmentShader: SHELL_FRAGMENT,
        transparent: true,
        depthWrite: false,
        uniforms: { uColor: { value: new THREE.Color() }, uOpacity: { value: 0.22 }, uGlow: { value: 0 } },
      });
    }
    if (kind === "orb") return new THREE.MeshBasicMaterial({ toneMapped: false });
    if (kind === "screen") return new THREE.MeshBasicMaterial({ toneMapped: false, transparent: true, side: THREE.DoubleSide });
    return new THREE.MeshPhysicalMaterial({ roughness: 0.38, metalness: 0.08, clearcoat: 0.7, clearcoatRoughness: 0.28, transparent: true });
  }

  function createNode(node: LiveNode): NodeVisual {
    const group = new THREE.Group();
    const material = makeMaterial(node.kind);
    const geometry = node.kind === "box" || node.kind === "slab"
      ? new THREE.BufferGeometry()
      : node.kind === "cylinder" || node.kind === "arrow" ? cylinderGeometry : node.kind === "ring" ? ringGeometry : node.kind === "screen" ? panelGeometry : sphereGeometry;
    const mesh = new THREE.Mesh(geometry, material);
    const solid = node.kind !== "shell" && node.kind !== "slab" && node.kind !== "orb" && node.kind !== "screen";
    mesh.castShadow = solid;
    mesh.receiveShadow = solid || node.kind === "slab";
    mesh.userData.nodeId = node.id;
    group.add(mesh);
    let head: THREE.Mesh | null = null;
    if (node.kind === "arrow") {
      head = new THREE.Mesh(coneGeometry, material);
      head.castShadow = true;
      head.userData.nodeId = node.id;
      group.add(head);
    }
    if (node.kind === "screen") {
      // A coloured border behind the picture, so the panel reads as a lit screen.
      head = new THREE.Mesh(panelGeometry, new THREE.MeshBasicMaterial({ toneMapped: false, transparent: true, side: THREE.DoubleSide }));
      head.position.z = -0.012;
      head.userData.nodeId = node.id;
      group.add(head);
    }
    const halo = new THREE.Mesh(haloGeometry, new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false, toneMapped: false }));
    halo.visible = false;
    halo.renderOrder = -1;
    content.add(group, halo);
    const color = new THREE.Color().copy(toneColor(node.tone));
    return { kind: node.kind, group, mesh, head, halo, color, built: null, lastSize: node.size, burst: null, flashSeen: node.flash, flashAt: -Infinity };
  }

  function removeNode(visual: NodeVisual): void {
    content.remove(visual.group, visual.halo);
    if (visual.burst) {
      content.remove(visual.burst);
      (visual.burst.material as THREE.Material).dispose();
    }
    if (visual.kind === "screen" && visual.head) (visual.head.material as THREE.Material).dispose();
    if (visual.built) visual.mesh.geometry.dispose();
    (visual.mesh.material as THREE.Material).dispose();
    (visual.halo.material as THREE.Material).dispose();
  }

  function syncNode(node: LiveNode, visual: NodeVisual, time: number, colorFactor: number): boolean {
    let moving = false;
    const target = toneColor(node.tone);
    if (!visual.color.equals(target)) {
      visual.color.lerp(target, colorFactor);
      if (Math.abs(visual.color.r - target.r) + Math.abs(visual.color.g - target.g) + Math.abs(visual.color.b - target.b) < 0.004) visual.color.copy(target);
      else moving = true;
    }
    const throb = node.pulse ? 0.5 + 0.5 * Math.sin(time * 4.2) : 0;
    const presence = ease(node.presence);

    // A change of the flash counter plays one short burst; never while motion is off (colorFactor 1).
    if (node.flash !== visual.flashSeen) {
      visual.flashSeen = node.flash;
      visual.flashAt = colorFactor === 1 ? -Infinity : time;
    }
    const flashAge = time - visual.flashAt;
    const flash = flashAge >= 0 && flashAge < FLASH_SECONDS ? 1 - flashAge / FLASH_SECONDS : 0;
    if (flash > 0) moving = true;
    const glow = node.emphasis * 0.3 + throb * 0.35 + flash * 0.9;

    const { group, mesh } = visual;
    group.visible = presence > 0.001;
    group.position.set(node.position[0], node.position[1], node.position[2]);
    group.rotation.set(0, 0, node.tilt);
    group.scale.setScalar(Math.max(presence, 0.001) * (1 + 0.05 * node.emphasis + 0.04 * throb + 0.07 * flash));

    if (node.kind === "box" || node.kind === "slab") {
      // Rebuild the rounded box only once its size has come to rest; stretch the old one meanwhile.
      const resting = visual.lastSize[0] === node.size[0] && visual.lastSize[1] === node.size[1] && visual.lastSize[2] === node.size[2];
      const stale = !visual.built || visual.built[0] !== node.size[0] || visual.built[1] !== node.size[1] || visual.built[2] !== node.size[2];
      if (stale && (resting || !visual.built)) {
        if (visual.built) mesh.geometry.dispose();
        const smallest = Math.max(0.01, Math.min(node.size[0], node.size[1], node.size[2]));
        mesh.geometry = new RoundedBoxGeometry(Math.max(node.size[0], 0.01), Math.max(node.size[1], 0.01), Math.max(node.size[2], 0.01), 3, Math.min(0.09, smallest * 0.3));
        visual.built = node.size;
      }
      const built = visual.built!;
      mesh.scale.set(node.size[0] / Math.max(built[0], 0.01), node.size[1] / Math.max(built[1], 0.01), node.size[2] / Math.max(built[2], 0.01));
      visual.lastSize = node.size;
    } else if (node.kind === "cylinder") {
      mesh.scale.set(node.size[0], node.size[1], node.size[0]);
    } else if (node.kind === "arrow") {
      scratch.set(node.tip[0] - node.position[0], node.tip[1] - node.position[1], node.tip[2] - node.position[2]);
      const length = Math.max(scratch.length(), 1e-4);
      const thickness = Math.max(node.size[0], 0.02);
      const headLength = Math.min(length * 0.4, thickness * 3.2);
      quaternion.setFromUnitVectors(UP, scratch.divideScalar(length));
      mesh.quaternion.copy(quaternion);
      mesh.scale.set(thickness, Math.max(length - headLength, 1e-4), thickness);
      mesh.position.copy(scratch).multiplyScalar((length - headLength) / 2);
      if (visual.head) {
        visual.head.quaternion.copy(quaternion);
        visual.head.scale.set(thickness * 2.6, headLength, thickness * 2.6);
        visual.head.position.copy(scratch).multiplyScalar(length - headLength / 2);
      }
    } else if (node.kind === "screen") {
      // Turn about the vertical axis to face the camera, so the picture is always readable.
      group.rotation.set(0, Math.atan2(camera.position.x - node.position[0], camera.position.z - node.position[2]), 0);
      mesh.scale.set(node.size[0], node.size[1], 1);
      visual.head?.scale.set(node.size[0] + 0.14, node.size[1] + 0.14, 1);
    } else {
      mesh.scale.setScalar(Math.max(node.size[0], 0.001));
    }

    const material = mesh.material;
    if (material instanceof THREE.ShaderMaterial) {
      (material.uniforms.uColor!.value as THREE.Color).copy(visual.color);
      material.uniforms.uOpacity!.value = node.opacity * presence;
      material.uniforms.uGlow!.value = glow;
      mesh.renderOrder = 1 + node.size[0];
    } else if (material instanceof THREE.MeshPhysicalMaterial) {
      material.color.copy(visual.color);
      material.emissive.copy(visual.color);
      material.emissiveIntensity = 0.05 + glow;
      material.opacity = node.opacity;
      material.depthWrite = node.opacity > 0.95;
      mesh.renderOrder = node.opacity > 0.95 ? 0 : 1;
    } else if (material instanceof THREE.MeshBasicMaterial && node.kind === "screen") {
      const picture = (node.screen ? screens.get(node.screen) : undefined) ?? null;
      if (material.map !== picture) {
        material.map = picture;
        material.needsUpdate = true;
      }
      material.color.setScalar(picture ? 1 + 0.5 * flash : 0.08);
      material.opacity = presence;
      const border = visual.head!.material as THREE.MeshBasicMaterial;
      border.color.copy(visual.color).multiplyScalar((palette?.dark ? 1.25 : 1) + 0.9 * flash + 0.3 * throb);
      border.opacity = presence;
      mesh.renderOrder = 2;
      visual.head!.renderOrder = 2;
    } else if (material instanceof THREE.MeshBasicMaterial) {
      material.color.copy(visual.color).multiplyScalar(palette?.dark ? 1.9 + throb : 1.15);
    }

    // The burst: a ring on the floor that runs outwards once.
    if (flash > 0 && node.kind !== "screen") {
      if (!visual.burst) {
        visual.burst = new THREE.Mesh(haloGeometry, new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false, toneMapped: false }));
        visual.burst.renderOrder = -1;
        content.add(visual.burst);
      }
      const footprint = Math.max(node.size[0], node.size[2] || node.size[0]);
      visual.burst.visible = true;
      visual.burst.position.set(node.position[0], floorY + 0.016, node.position[2]);
      visual.burst.scale.setScalar(footprint * (1.4 + 4.2 * (1 - flash)));
      const burstMaterial = visual.burst.material as THREE.MeshBasicMaterial;
      burstMaterial.color.copy(visual.color).multiplyScalar(palette?.dark ? 1.8 : 1);
      burstMaterial.opacity = flash;
    } else if (visual.burst) {
      visual.burst.visible = false;
    }

    // A ring on the floor marks the active node; it ripples outwards while motion is on.
    const halo = visual.halo;
    const marked = node.emphasis > 0.02 && node.kind !== "ring" && node.kind !== "arrow" && node.kind !== "orb" && node.kind !== "screen" && !node.leaving;
    halo.visible = marked;
    if (marked) {
      const ripple = time * 0.7 - Math.floor(time * 0.7);
      const footprint = node.kind === "box" || node.kind === "slab" || node.kind === "cylinder" ? Math.max(node.size[0], node.size[2] || node.size[0]) : node.size[0];
      halo.position.set(node.position[0], floorY + 0.012, node.position[2]);
      halo.scale.setScalar(footprint * (1.75 + 0.9 * ripple));
      const haloMaterial = halo.material as THREE.MeshBasicMaterial;
      haloMaterial.color.copy(visual.color).multiplyScalar(palette?.dark ? 1.6 : 1);
      haloMaterial.opacity = node.emphasis * (time === 0 ? 0.8 : 0.85 * (1 - ripple));
    }
    return moving;
  }

  function createLink(link: LiveLink): LinkVisual {
    const material = new THREE.ShaderMaterial({
      vertexShader: LINK_VERTEX,
      fragmentShader: LINK_FRAGMENT,
      transparent: true,
      uniforms: {
        uColor: { value: new THREE.Color() }, uOpacity: { value: 1 }, uDashes: { value: 0 }, uTime: { value: 0 },
        uFlow: { value: 0 }, uGlow: { value: 1 }, uReveal: { value: 1 }, uRadius: { value: 0.03 },
      },
    });
    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), material);
    mesh.frustumCulled = false;
    content.add(mesh);
    return { mesh, material, color: new THREE.Color().copy(toneColor(link.tone)), built: "", length: 1 };
  }

  function removeLink(visual: LinkVisual): void {
    content.remove(visual.mesh);
    visual.mesh.geometry.dispose();
    visual.material.dispose();
  }

  function syncLink(link: LiveLink, visual: LinkVisual, time: number, colorFactor: number): boolean {
    const key = `${link.a.join(",")}|${link.b.join(",")}|${link.arc}`;
    if (key !== visual.built) {
      visual.mesh.geometry.dispose();
      const a = new THREE.Vector3(link.a[0], link.a[1], link.a[2]);
      const b = new THREE.Vector3(link.b[0], link.b[1], link.b[2]);
      const control = new THREE.Vector3().addVectors(a, b).multiplyScalar(0.5);
      control.y += link.arc;
      const curve = new THREE.QuadraticBezierCurve3(a, control, b);
      visual.length = Math.max(curve.getLength(), 1e-3);
      visual.mesh.geometry = new THREE.TubeGeometry(curve, link.arc === 0 ? 2 : 28, 1, 8, false);
      visual.built = key;
    }
    let moving = false;
    const target = toneColor(link.tone);
    if (!visual.color.equals(target)) {
      visual.color.lerp(target, colorFactor);
      if (Math.abs(visual.color.r - target.r) + Math.abs(visual.color.g - target.g) + Math.abs(visual.color.b - target.b) < 0.004) visual.color.copy(target);
      else moving = true;
    }
    const uniforms = visual.material.uniforms;
    (uniforms.uColor!.value as THREE.Color).copy(visual.color);
    uniforms.uRadius!.value = (palette?.dark ? 0.024 : 0.034) + 0.03 * link.emphasis;
    uniforms.uGlow!.value = (palette?.dark ? 0.85 : 1) + link.emphasis * (palette?.dark ? 0.9 : 0.25);
    uniforms.uOpacity!.value = link.leaving ? Math.max(0, link.presence) : 0.55 + 0.45 * Math.max(link.emphasis, link.flow === 0 ? 0 : 0.7);
    uniforms.uReveal!.value = link.leaving ? 1 : ease(link.presence);
    uniforms.uDashes!.value = link.dashed ? Math.max(2, Math.round(visual.length * 3.2)) : 0;
    uniforms.uFlow!.value = link.flow;
    uniforms.uTime!.value = time;
    return moving;
  }

  function resize(width: number, height: number): void {
    if (width === drawnWidth && height === drawnHeight) return;
    drawnWidth = width;
    drawnHeight = height;
    renderer.setSize(width, height, false);
    composer?.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }

  function applyQuality(): void {
    const full = quality === "full";
    const ratio = full ? Math.min(window.devicePixelRatio || 1, 2) : 1;
    renderer.setPixelRatio(ratio);
    sun.castShadow = full;
    shadowFloor.visible = full;
    world.environment = full ? studioLight() : null;
    if (full && !composer) {
      const target = new THREE.WebGLRenderTarget(Math.max(1, drawnWidth), Math.max(1, drawnHeight), { type: THREE.HalfFloatType, samples: 4 });
      composer = new EffectComposer(renderer, target);
      composer.addPass(new RenderPass(world, camera));
      bloom = new UnrealBloomPass(new THREE.Vector2(Math.max(1, drawnWidth), Math.max(1, drawnHeight)), 0.5, 0.6, 0.8);
      composer.addPass(bloom);
      composer.addPass(new OutputPass());
    }
    if (!full && composer) {
      composer.dispose();
      composer = null;
      bloom = null;
    }
    composer?.setPixelRatio(ratio);
    if (drawnWidth > 0) {
      renderer.setSize(drawnWidth, drawnHeight, false);
      composer?.setSize(drawnWidth, drawnHeight);
    }
    applyPalette();
  }

  function applyPalette(): void {
    if (!palette) return;
    backdrop?.dispose();
    backdrop = backdropTexture(palette.backdropTop, palette.backdropBottom);
    world.background = backdrop;
    (gridMaterial.uniforms.uColor!.value as THREE.Color).set(palette.grid);
    gridMaterial.uniforms.uOpacity!.value = palette.dark ? 0.75 : 0.9;
    shadowMaterial.opacity = palette.dark ? 0.45 : 0.3;
    hemisphere.intensity = palette.dark ? 0.4 : 0.95;
    sun.intensity = palette.dark ? 1.15 : 2.1;
    world.environmentIntensity = palette.dark ? 0.35 : 0.75;
    if (bloom) {
      bloom.strength = palette.dark ? 0.5 : 0.12;
      bloom.radius = palette.dark ? 0.7 : 0.4;
      bloom.threshold = palette.dark ? 0.86 : 0.98;
    }
  }

  return {
    rendererName,
    setPalette(next) {
      palette = next;
      applyPalette();
    },
    setQuality(next) {
      if (next === quality && (next === "lite" || composer)) return;
      quality = next;
      applyQuality();
    },
    draw(live, view) {
      const width = Math.max(1, Math.round(view.width));
      const height = Math.max(1, Math.round(view.height));
      resize(width, height);
      const inset = Math.max(-width * 0.6, Math.min(view.inset ?? 0, width * 0.6));
      const insetBottom = Math.max(0, Math.min(view.insetBottom ?? 0, height * 0.5));
      if (inset !== drawnInset || insetBottom !== drawnInsetBottom || camera.view?.fullWidth !== width || camera.view?.fullHeight !== height) {
        drawnInset = inset;
        drawnInsetBottom = insetBottom;
        // Slide the picture by half of each covered strip, so the scene sits in the middle of the free area.
        if (inset !== 0 || insetBottom !== 0) camera.setViewOffset(width, height, inset / 2, insetBottom / 2, width, height);
        else camera.clearViewOffset();
      }
      const smooth = blendFactor(view.dt, 9);
      let animating = !live.settled;

      const seen = new Set<string>();
      pickable.length = 0;
      const bounds = new THREE.Box3();
      for (const node of live.nodes) {
        seen.add(node.id);
        let visual = nodes.get(node.id);
        if (visual && visual.kind !== node.kind) {
          removeNode(visual);
          visual = undefined;
        }
        if (!visual) {
          visual = createNode(node);
          nodes.set(node.id, visual);
        }
        if (syncNode(node, visual, view.time, smooth)) animating = true;
        if (node.pulse || node.emphasis > 0.02) animating = true;
        if (!node.leaving) {
          const half = halfExtent(node);
          scratch.set(node.position[0], node.position[1], node.position[2]);
          bounds.expandByPoint(scratch.clone().sub(half)).expandByPoint(scratch.clone().add(half));
          if (node.kind === "arrow") bounds.expandByPoint(scratch.set(node.tip[0], node.tip[1], node.tip[2]));
          if (node.kind !== "slab" && node.kind !== "shell") pickable.push(visual.mesh);
          if (visual.head) pickable.push(visual.head);
        }
      }
      for (const [id, visual] of nodes) {
        if (seen.has(id)) continue;
        removeNode(visual);
        nodes.delete(id);
      }

      const seenLinks = new Set<string>();
      let orbs = 0;
      for (const link of live.links) {
        seenLinks.add(link.id);
        let visual = links.get(link.id);
        if (!visual) {
          visual = createLink(link);
          links.set(link.id, visual);
        }
        if (syncLink(link, visual, view.time, smooth)) animating = true;
        if (link.flow !== 0 && !link.leaving && link.presence > 0.6) {
          animating = true;
          const count = Math.max(1, Math.min(4, Math.round(visual.length / 1.6)));
          const boost = palette?.dark ? 2.1 : 1.25;
          for (const offset of flowOffsets(view.time, count, 0.55 / Math.max(visual.length, 1) + 0.12, link.flow)) {
            if (orbs >= MAX_FLOW_ORBS) break;
            const point = curvePoint(link.a, link.b, link.arc, offset);
            const size = (0.11 + 0.05 * link.emphasis) * Math.sin(Math.PI * offset) ** 0.5;
            matrix.compose(scratch.set(point[0], point[1], point[2]), quaternion.identity(), scale.setScalar(Math.max(size, 0.0001)));
            flowOrbs.setMatrixAt(orbs, matrix);
            flowOrbs.setColorAt(orbs, tint.copy(visual.color).multiplyScalar(boost));
            orbs += 1;
          }
        }
      }
      for (const [id, visual] of links) {
        if (seenLinks.has(id)) continue;
        removeLink(visual);
        links.delete(id);
      }
      flowOrbs.count = orbs;
      flowOrbs.instanceMatrix.needsUpdate = true;
      if (flowOrbs.instanceColor) flowOrbs.instanceColor.needsUpdate = true;

      // Frame the scene: follow its bounds smoothly so the camera does not jump between steps.
      if (bounds.isEmpty()) bounds.set(new THREE.Vector3(-1, 0, -1), new THREE.Vector3(1, 1, 1));
      const size = bounds.getSize(new THREE.Vector3());
      const wantCenter = bounds.getCenter(new THREE.Vector3());
      const wantRadius = Math.max(size.length() * 1.5, 6);
      const follow = framed ? blendFactor(view.dt, 4) : 1;
      framed = true;
      center.lerp(wantCenter, follow);
      radius += (wantRadius - radius) * follow;
      floorY += (bounds.min.y - 0.02 - floorY) * follow;
      if (center.distanceTo(wantCenter) > 0.01 || Math.abs(radius - wantRadius) > 0.01) animating = true;
      grid.position.y = floorY;
      shadowFloor.position.y = floorY + 0.004;
      (gridMaterial.uniforms.uCenter!.value as THREE.Vector3).copy(center);
      gridMaterial.uniforms.uRadius!.value = radius * 1.5;

      const focused = view.focusId ? live.nodes.find((node) => node.id === view.focusId) : undefined;
      const wantLook = focused ? scratch.set(focused.position[0], focused.position[1], focused.position[2]).clone() : center.clone();
      if (lookAt.distanceTo(wantLook) > 0.01) animating = true;
      lookAt.lerp(wantLook, blendFactor(view.dt, 6));
      const cover = Math.min(width * 0.7, Math.max(Math.abs(inset), view.insetCover ?? 0));
      const narrow = Math.max(1, 1.35 / ((width - cover) / Math.max(1, height - insetBottom)));
      const offset = orbitOffset(view.orbit, radius * narrow * (focused ? 0.7 : 1));
      camera.position.set(lookAt.x + offset[0], lookAt.y + offset[1], lookAt.z + offset[2]);
      camera.lookAt(lookAt);
      camera.updateMatrixWorld();

      sun.position.set(center.x + radius * 0.5, center.y + radius * 1.2, center.z + radius * 0.75);
      sun.target.position.copy(center);
      const shadowCamera = sun.shadow.camera;
      shadowCamera.left = -radius;
      shadowCamera.right = radius;
      shadowCamera.top = radius;
      shadowCamera.bottom = -radius;
      shadowCamera.near = 0.1;
      shadowCamera.far = radius * 4;
      shadowCamera.updateProjectionMatrix();

      if (composer) composer.render();
      else renderer.render(world, camera);

      const labels: LabelPoint[] = [];
      for (const node of live.nodes) {
        if (!node.label || node.leaving || node.presence < 0.6) continue;
        const half = halfExtent(node);
        const anchor = node.kind === "arrow"
          ? scratch.set(node.tip[0], node.tip[1] + 0.3, node.tip[2])
          : scratch.set(node.position[0], node.position[1] + half.y + 0.24, node.position[2]);
        anchor.project(camera);
        labels.push({
          id: node.id,
          text: node.label,
          active: node.emphasis > 0.5,
          x: (anchor.x * 0.5 + 0.5) * width,
          y: (-anchor.y * 0.5 + 0.5) * height,
          depth: (anchor.z + 1) / 2,
          visible: anchor.z < 1 && Math.abs(anchor.x) <= 1.02 && Math.abs(anchor.y) <= 1.02,
        });
      }
      return { labels, animating };
    },
    pick(x, y) {
      if (drawnWidth === 0) return null;
      raycaster.setFromCamera(new THREE.Vector2((x / drawnWidth) * 2 - 1, -(y / drawnHeight) * 2 + 1), camera);
      const hit = raycaster.intersectObjects(pickable, false)[0];
      return hit ? String(hit.object.userData.nodeId) : null;
    },
    setScreen(id, source) {
      const current = screens.get(id);
      // A texture keeps the size it was created with, so a resized canvas needs a new one.
      if (current && current.image === source && current.userData.width === source.width && current.userData.height === source.height) {
        current.needsUpdate = true;
        return;
      }
      current?.dispose();
      const texture = new THREE.CanvasTexture(source);
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.magFilter = THREE.NearestFilter;
      texture.minFilter = THREE.LinearFilter;
      texture.generateMipmaps = false;
      texture.userData = { width: source.width, height: source.height };
      screens.set(id, texture);
    },
    dispose() {
      for (const texture of screens.values()) texture.dispose();
      screens.clear();
      for (const visual of nodes.values()) removeNode(visual);
      for (const visual of links.values()) removeLink(visual);
      nodes.clear();
      links.clear();
      shared.forEach((geometry) => geometry.dispose());
      floorGeometry.dispose();
      gridMaterial.dispose();
      shadowMaterial.dispose();
      flowMaterial.dispose();
      flowOrbs.dispose();
      backdrop?.dispose();
      environment?.dispose();
      composer?.dispose();
      renderer.dispose();
      // Browsers allow only a handful of WebGL contexts per page; hand this one back straight away.
      renderer.forceContextLoss();
    },
  };
}
