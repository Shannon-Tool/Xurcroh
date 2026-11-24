import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { axe } from "jest-axe";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DEFAULT_ORBIT, orbitOffset, rotateOrbit, zoomOrbit } from "@/lib/scene/orbit";
import { effectsSetting, motionSetting } from "@/lib/scene/settings";
import type { StageView } from "@/lib/scene/stage";
import type { LiveScene } from "@/lib/scene/tween";
import type { Scene3D } from "@/lib/scene/types";

const stage = {
  rendererName: "Test GPU",
  setPalette: vi.fn((palette: unknown) => { void palette; }),
  setQuality: vi.fn((quality: string) => { void quality; }),
  draw: vi.fn((live: LiveScene, view: StageView) => { void live; void view; return { labels: [{ id: "a", text: "Worker A", x: 40, y: 30, depth: 0.5, visible: true, active: true }], animating: false }; }),
  pick: vi.fn((x: number, y: number): string | null => { void x; void y; return null; }),
  dispose: vi.fn(),
};
const createStage = vi.fn((canvas: HTMLCanvasElement) => { void canvas; return stage; });
vi.mock("@/lib/scene/stage", () => ({ createStage: (canvas: HTMLCanvasElement) => createStage(canvas) }));

import { Scene3DView } from "@/components/Scene3DView";
import { LabShell } from "@/components/LabShell";

const scene: Scene3D = {
  caption: "Two workers and one link.",
  nodes: [
    { id: "a", kind: "box", position: [0, 0.5, 0], size: [1, 1, 1], tone: "accent", label: "Worker A", detail: "Holds share 0.", active: true },
    { id: "b", kind: "sphere", position: [2, 0.5, 0], size: [1, 1, 1], tone: "plain" },
  ],
  links: [{ id: "ab", from: "a", to: "b", tone: "plain", flow: "forward" }],
  legend: [{ tone: "accent", text: "worker" }],
};

function setReducedMotion(reduced: boolean) {
  vi.stubGlobal("matchMedia", (query: string) => ({ matches: reduced && query.includes("reduce"), media: query, addEventListener: () => undefined, removeEventListener: () => undefined }));
}
const lastView = () => stage.draw.mock.calls.at(-1)![1];
const lastLive = () => stage.draw.mock.calls.at(-1)![0];
const pause = (ms: number) => act(() => new Promise<void>((resolve) => { setTimeout(resolve, ms); }));

beforeEach(() => {
  vi.stubGlobal("WebGLRenderingContext", class {});
  setReducedMotion(true);
  window.localStorage.clear();
  effectsSetting.set("auto");
  motionSetting.set("on");
  createStage.mockClear();
  createStage.mockImplementation(() => stage);
  Object.values(stage).forEach((member) => { if (typeof member === "function") member.mockClear(); });
  stage.pick.mockImplementation(() => null);
});
afterEach(() => vi.unstubAllGlobals());

describe("orbit maths", () => {
  it("keeps the camera above the floor and inside the zoom range", () => {
    expect(rotateOrbit(DEFAULT_ORBIT, 0, 10).phi).toBeLessThan(Math.PI / 2);
    expect(rotateOrbit(DEFAULT_ORBIT, 0, -10).phi).toBeGreaterThan(0);
    expect(zoomOrbit(DEFAULT_ORBIT, 1000).zoom).toBe(2.5);
    expect(zoomOrbit(DEFAULT_ORBIT, 0.0001).zoom).toBe(0.45);
    const [x, y, z] = orbitOffset({ theta: 0, phi: Math.PI / 2, zoom: 1 }, 5);
    expect(x).toBeCloseTo(0);
    expect(y).toBeCloseTo(0);
    expect(z).toBeCloseTo(5);
  });
});

describe("Scene3DView", () => {
  it("draws the scene, shows its labels, caption and legend, and releases the renderer on unmount", async () => {
    const { container, unmount } = render(<Scene3DView scene={scene} />);
    await waitFor(() => expect(stage.draw).toHaveBeenCalled());
    expect(lastLive().nodes.map((node) => node.id)).toEqual(["a", "b"]);
    expect(lastLive().links).toHaveLength(1);
    await waitFor(() => expect(container.querySelector(".scene3d-label")).toHaveTextContent("Worker A"));
    expect(screen.getByRole("img", { name: /Two workers and one link/ })).toBeInTheDocument();
    expect(screen.getByRole("list", { name: "Colours in the 3D view" })).toHaveTextContent("worker");
    expect(screen.getByRole("list", { name: "Objects in the 3D view" })).toHaveTextContent("Worker A: Holds share 0.");
    expect(stage.setPalette).toHaveBeenCalled();
    expect((await axe(container)).violations).toHaveLength(0);
    unmount();
    expect(stage.dispose).toHaveBeenCalledTimes(1);
  });

  it("with reduced motion, snaps to each state and draws only when something changes", async () => {
    const { rerender } = render(<Scene3DView scene={scene} />);
    await waitFor(() => expect(stage.draw).toHaveBeenCalled());
    await pause(80);
    expect(lastLive().settled).toBe(true);
    expect(lastView().dt).toBe(Infinity);
    expect(lastView().time).toBe(0);
    const drawn = stage.draw.mock.calls.length;
    rerender(<Scene3DView scene={{ ...scene, nodes: [...scene.nodes] }} />);
    await pause(80);
    expect(stage.draw.mock.calls.length).toBe(drawn);
    rerender(<Scene3DView scene={{ ...scene, caption: "Changed." }} />);
    await waitFor(() => expect(stage.draw.mock.calls.length).toBeGreaterThan(drawn));
  });

  it("with motion allowed, glides into a new state and then stops drawing", async () => {
    setReducedMotion(false);
    render(<Scene3DView scene={scene} autoRotate={false} />);
    await waitFor(() => expect(stage.draw.mock.calls.length).toBeGreaterThan(2));
    expect(stage.draw.mock.calls[1]![0].settled).toBe(false);
    expect(stage.draw.mock.calls[1]![0].nodes[0]!.presence).toBeLessThan(1);
    await waitFor(() => expect(lastLive().settled).toBe(true), { timeout: 4000 });
    await pause(60);
    const drawn = stage.draw.mock.calls.length;
    await pause(120);
    expect(stage.draw.mock.calls.length).toBe(drawn);
  });

  it("turns slowly on its own until the reader takes over, and can be paused", async () => {
    setReducedMotion(false);
    render(<Scene3DView scene={scene} />);
    await waitFor(() => expect(lastView().orbit.theta).toBeGreaterThan(DEFAULT_ORBIT.theta));
    fireEvent.click(screen.getByRole("button", { name: "Pause motion" }));
    await waitFor(() => expect(lastView().dt).toBe(Infinity));
    expect(screen.getByRole("button", { name: "Play motion" })).toBeInTheDocument();
    expect(motionSetting.get()).toBe("off");
  });

  it("rotates and zooms from the keyboard and buttons, and resets", async () => {
    render(<Scene3DView scene={scene} />);
    await waitFor(() => expect(stage.draw).toHaveBeenCalled());
    const canvas = screen.getByRole("img");
    fireEvent.keyDown(canvas, { key: "ArrowRight" });
    await waitFor(() => expect(lastView().orbit.theta).toBeGreaterThan(DEFAULT_ORBIT.theta));
    fireEvent.click(screen.getByRole("button", { name: "Zoom in" }));
    await waitFor(() => expect(lastView().orbit.zoom).toBeLessThan(1));
    fireEvent.click(screen.getByRole("button", { name: "Reset view" }));
    await waitFor(() => expect(lastView().orbit).toEqual(DEFAULT_ORBIT));
  });

  it("lets the reader choose the effects level and remembers it", async () => {
    stage.rendererName = "ANGLE (SwiftShader)";
    render(<Scene3DView scene={scene} />);
    await waitFor(() => expect(stage.setQuality).toHaveBeenLastCalledWith("lite"));
    fireEvent.click(screen.getByRole("button", { name: "Effects: auto" }));
    await waitFor(() => expect(stage.setQuality).toHaveBeenLastCalledWith("full"));
    expect(screen.getByRole("button", { name: "Effects: on" })).toBeInTheDocument();
    expect(window.localStorage.getItem("xurcroh-3d-effects")).toBe("on");
    stage.rendererName = "Test GPU";
  });

  it("explains what the reader is pointing at", async () => {
    stage.pick.mockImplementation(() => "a");
    const { container } = render(<Scene3DView scene={scene} />);
    await waitFor(() => expect(stage.draw).toHaveBeenCalled());
    fireEvent.pointerMove(container.querySelector(".scene3d-frame")!, { pointerType: "mouse", clientX: 30, clientY: 30 });
    await waitFor(() => expect(container.querySelector(".scene3d-tip")).toHaveTextContent("Holds share 0."));
  });

  it("holds a renderer only while it is near the viewport", async () => {
    let report: (visible: boolean) => void = () => undefined;
    vi.stubGlobal("IntersectionObserver", class {
      constructor(callback: (entries: { isIntersecting: boolean }[]) => void) { report = (visible) => callback([{ isIntersecting: visible }]); }
      observe() { return undefined; }
      disconnect() { return undefined; }
    });
    render(<Scene3DView scene={scene} />);
    await pause(30);
    expect(createStage).not.toHaveBeenCalled();
    act(() => report(true));
    await waitFor(() => expect(createStage).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(stage.draw).toHaveBeenCalled());
    act(() => report(false));
    await waitFor(() => expect(stage.dispose).toHaveBeenCalledTimes(1));
    act(() => report(true));
    await waitFor(() => expect(createStage).toHaveBeenCalledTimes(2));
  });

  it("offers to reload when the browser takes the graphics context away", async () => {
    const { container } = render(<Scene3DView scene={scene} />);
    await waitFor(() => expect(stage.draw).toHaveBeenCalled());
    act(() => { container.querySelector("canvas")!.dispatchEvent(new Event("webglcontextlost", { cancelable: true })); });
    expect(await screen.findByText(/stopped the 3D view/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Reload 3D view" }));
    await waitFor(() => expect(createStage).toHaveBeenCalledTimes(2));
  });

  it("explains itself when the browser has no WebGL", async () => {
    vi.unstubAllGlobals();
    render(<Scene3DView scene={scene} />);
    expect(await screen.findByText(/3D view needs WebGL/)).toBeInTheDocument();
    expect(createStage).not.toHaveBeenCalled();
    expect(screen.getByText("Two workers and one link.")).toBeInTheDocument();
  });

  it("falls back when the renderer cannot start", async () => {
    createStage.mockImplementation(() => { throw new Error("no context"); });
    render(<Scene3DView scene={scene} />);
    expect(await screen.findByText(/3D view needs WebGL/)).toBeInTheDocument();
  });
});

describe("LabShell", () => {
  it("shows the 3D view by default and lets the reader hide it", async () => {
    render(<LabShell id="demo" title="Demo" caption="c" scene={scene}><p>data</p></LabShell>);
    expect(await screen.findByRole("img", { name: /Two workers/ })).toBeInTheDocument();
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Hide 3D view" })); });
    expect(screen.queryByRole("img")).toBeNull();
    expect(screen.getByText("data")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Show 3D view" })).toBeInTheDocument();
  });
});
