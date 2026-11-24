import { fireEvent, render, screen } from "@testing-library/react";
import { axe } from "jest-axe";
import { describe, expect, it, vi } from "vitest";

vi.mock("@/components/Scene3DView", () => ({
  Scene3DView: ({ scene, screens }: { scene: { nodes: { id: string; kind: string }[] }; screens?: Record<string, unknown> }) => (
    <div data-testid="view" data-screens={Object.keys(screens ?? {}).join(",")}>{scene.nodes.filter((node) => node.kind === "screen").map((node) => node.id).join(",")}</div>
  ),
}));

import { LabShell } from "@/components/LabShell";
import type { LiveStory } from "@/components/LiveIO";
import { withIOScreens } from "@/lib/scene/io-screens";
import type { Scene3D } from "@/lib/scene/types";

const scene: Scene3D = {
  caption: "Two boxes.",
  nodes: [
    { id: "a", kind: "box", position: [-2, 0.5, 1], size: [1, 1, 1], tone: "accent" },
    { id: "b", kind: "box", position: [4, 0.5, -1], size: [1, 1, 1], tone: "plain" },
  ],
  links: [{ id: "ab", from: "a", to: "b", tone: "plain" }],
};

function story(extra: Partial<LiveStory> = {}): LiveStory {
  return {
    plain: "Your numbers are cut in two.",
    input: <input aria-label="Your numbers" defaultValue="5, 7" />,
    seenTitle: "What one worker sees",
    seen: <code>81234, 99120</code>,
    output: <strong>19, 43</strong>,
    why: "One piece alone is random.",
    relies: "Workers do not pool pieces.",
    limit: "Two workers together can rebuild it.",
    screens: { input: "5, 7", seen: "81234", output: "19, 43" },
    ...extra,
  };
}

describe("io screens", () => {
  it("adds three joined screens behind the scene without touching what was there", () => {
    const staged = withIOScreens(scene);
    expect(staged.nodes.slice(0, 2)).toEqual(scene.nodes);
    const added = staged.nodes.slice(2);
    expect(added.map((node) => [node.id, node.kind, node.screen])).toEqual([["io-in", "screen", "io-in"], ["io-seen", "screen", "io-seen"], ["io-out", "screen", "io-out"]]);
    for (const node of added) expect(node.position[2]).toBeLessThan(-1.5);
    expect(added[0]!.position[0]).toBeLessThan(added[1]!.position[0]);
    expect(added[1]!.position[0]).toBeCloseTo(1);
    expect(staged.links.filter((link) => link.flow === "forward").map((link) => [link.from, link.to])).toEqual([["io-in", "io-seen"], ["io-seen", "io-out"]]);
    expect(withIOScreens(scene, "warn").nodes.at(-1)!.tone).toBe("warn");
    expect(withIOScreens({ caption: "Empty scene here.", nodes: [], links: [] }).nodes).toHaveLength(3);
  });
});

describe("lab shell with a story", () => {
  it("tells the lab in plain words, shows in, seen and out, and keeps the exact numbers", async () => {
    const { container } = render(<LabShell id="demo" title="Demo lab" caption="Technical caption." scene={scene} story={story()}><p>exact table</p></LabShell>);
    expect(screen.getByText("Your numbers are cut in two.")).toBeInTheDocument();
    expect(screen.getByText("Technical caption.")).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 3, name: "What goes in" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 3, name: "What one worker sees" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 3, name: "What comes out" })).toBeInTheDocument();
    expect(screen.getByLabelText("Your numbers")).toHaveValue("5, 7");
    expect(screen.getByText("Why this protects you").nextElementSibling).toHaveTextContent("One piece alone is random.");
    expect(screen.getByText("Where it stops").nextElementSibling).toHaveTextContent("Two workers together can rebuild it.");
    expect(screen.getByText("exact table").closest("details")).toHaveAttribute("open");
    expect(screen.getByTestId("view")).toHaveTextContent("io-in,io-seen,io-out");
    expect(screen.getByTestId("view")).toHaveAttribute("data-screens", "io-in,io-seen,io-out");
    expect((await axe(container)).violations).toHaveLength(0);
  });

  it("offers the attack and reports what happened to it", () => {
    const onToggle = vi.fn();
    const { rerender } = render(<LabShell id="demo" title="Demo lab" caption="c" story={story({ attack: { label: "tamper with a worker", on: false, onToggle, outcome: "All four products passed." } })}><p>x</p></LabShell>);
    const button = screen.getByRole("button", { name: "Try to break it: tamper with a worker" });
    expect(button).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByRole("status")).toHaveTextContent("All four products passed.");
    fireEvent.click(button);
    expect(onToggle).toHaveBeenCalledTimes(1);
    rerender(<LabShell id="demo" title="Demo lab" caption="c" story={story({ outputTone: "warn", attack: { label: "tamper with a worker", on: true, onToggle, outcome: "Caught." } })}><p>x</p></LabShell>);
    expect(screen.getByRole("button", { name: "Try to break it: tamper with a worker" })).toHaveAttribute("aria-pressed", "true");
    expect(document.querySelector(".liveio-out")).toHaveClass("warn");
  });

  it("plays the lab from the header and leaves labs without a story as they were", () => {
    const onToggle = vi.fn();
    const { rerender } = render(<LabShell id="demo" title="Demo lab" caption="c" watch={{ playing: false, onToggle }}><p>plain child</p></LabShell>);
    fireEvent.click(screen.getByRole("button", { name: "Watch" }));
    expect(onToggle).toHaveBeenCalledTimes(1);
    expect(screen.getByText("plain child").closest("details")).toBeNull();
    rerender(<LabShell id="demo" title="Demo lab" caption="c" watch={{ playing: true, onToggle }}><p>plain child</p></LabShell>);
    expect(screen.getByRole("button", { name: "Stop" })).toBeInTheDocument();
  });
});
