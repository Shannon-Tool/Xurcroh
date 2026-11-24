import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { axe } from "jest-axe";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DEFAULT_ORBIT, orbitOffset, rotateOrbit, zoomOrbit, type Orbit } from "@/lib/scene/orbit";
import type { Scene3D } from "@/lib/scene/types";

const stage = {
  update: vi.fn((scene: Scene3D, palette: unknown) => { void scene; void palette; }),
  view: vi.fn((orbit: Orbit, width: number, height: number) => { void orbit; void width; void height; return [{ id: "a", text: "Worker A", x: 40, y: 30, visible: true, active: true }]; }),
  dispose: vi.fn(),
};
const createStage = vi.fn((canvas: HTMLCanvasElement) => { void canvas; return stage; });
vi.mock("@/lib/scene/stage", () => ({ createStage: (canvas: HTMLCanvasElement) => createStage(canvas) }));

import { Scene3DView } from "@/components/Scene3DView";
import { LabShell } from "@/components/LabShell";

const scene: Scene3D = {
  caption: "Two workers and one link.",
  nodes: [
    { id: "a", kind: "box", position: [0, 0.5, 0], size: [1, 1, 1], tone: "accent", label: "Worker A", active: true },
    { id: "b", kind: "sphere", position: [2, 0.5, 0], size: [1, 1, 1], tone: "plain" },
  ],
  links: [{ id: "ab", from: "a", to: "b", tone: "plain" }],
  packets: [{ id: "p", link: "ab", progress: 0.5, tone: "warn" }],
};

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
  beforeEach(() => {
    vi.stubGlobal("WebGLRenderingContext", class {});
    createStage.mockClear();
    createStage.mockImplementation(() => stage);
    stage.update.mockClear();
    stage.view.mockClear();
    stage.dispose.mockClear();
  });
  afterEach(() => vi.unstubAllGlobals());

  it("draws the scene, shows its labels and caption, and releases the renderer on unmount", async () => {
    const { container, unmount } = render(<Scene3DView scene={scene} />);
    await waitFor(() => expect(stage.update).toHaveBeenCalled());
    expect(stage.update.mock.calls[0]![0]).toEqual(scene);
    await waitFor(() => expect(screen.getByText("Worker A")).toBeInTheDocument());
    expect(screen.getByRole("img", { name: /Two workers and one link/ })).toBeInTheDocument();
    expect((await axe(container)).violations).toHaveLength(0);
    unmount();
    expect(stage.dispose).toHaveBeenCalledTimes(1);
  });

  it("does not redraw for an equal scene but does for a changed one", async () => {
    const { rerender } = render(<Scene3DView scene={scene} />);
    await waitFor(() => expect(stage.update).toHaveBeenCalledTimes(1));
    rerender(<Scene3DView scene={{ ...scene, nodes: [...scene.nodes] }} />);
    expect(stage.update).toHaveBeenCalledTimes(1);
    rerender(<Scene3DView scene={{ ...scene, caption: "Changed." }} />);
    await waitFor(() => expect(stage.update).toHaveBeenCalledTimes(2));
  });

  it("rotates and zooms from the keyboard and buttons, and resets", async () => {
    render(<Scene3DView scene={scene} />);
    await waitFor(() => expect(stage.view).toHaveBeenCalled());
    const canvas = screen.getByRole("img");
    const last = () => stage.view.mock.calls.at(-1)![0];
    fireEvent.keyDown(canvas, { key: "ArrowRight" });
    await waitFor(() => expect(last().theta).toBeGreaterThan(DEFAULT_ORBIT.theta));
    fireEvent.click(screen.getByRole("button", { name: "Zoom in" }));
    await waitFor(() => expect(last().zoom).toBeLessThan(1));
    fireEvent.click(screen.getByRole("button", { name: "Reset view" }));
    await waitFor(() => expect(last()).toEqual(DEFAULT_ORBIT));
  });

  it("explains itself when the browser has no WebGL", async () => {
    vi.unstubAllGlobals();
    render(<Scene3DView scene={scene} />);
    expect(await screen.findByText(/3D view needs WebGL/)).toBeInTheDocument();
    expect(createStage).not.toHaveBeenCalled();
    expect(screen.getByText("Two workers and one link.")).toBeInTheDocument();
  });

  it("falls back when the renderer cannot start", async () => {
    createStage.mockImplementation(() => { throw new Error("context lost"); });
    render(<Scene3DView scene={scene} />);
    expect(await screen.findByText(/3D view needs WebGL/)).toBeInTheDocument();
  });
});

describe("LabShell", () => {
  beforeEach(() => vi.stubGlobal("WebGLRenderingContext", class {}));
  afterEach(() => vi.unstubAllGlobals());

  it("shows the 3D view by default and lets the reader hide it", async () => {
    render(<LabShell id="demo" title="Demo" caption="c" scene={scene}><p>data</p></LabShell>);
    expect(await screen.findByRole("img", { name: /Two workers/ })).toBeInTheDocument();
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Hide 3D view" })); });
    expect(screen.queryByRole("img")).toBeNull();
    expect(screen.getByText("data")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Show 3D view" })).toBeInTheDocument();
  });
});
