import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { axe } from "jest-axe";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/components/Scene3DView", () => ({ Scene3DView: ({ scene }: { scene: { caption: string } }) => <div data-testid="view">{scene.caption}</div> }));

import ProtectionPage from "@/app/protection/page";
import { ProtectionTour } from "@/components/ProtectionTour";
import { sections } from "@/lib/content";
import { protectionScene } from "@/lib/scene/protection-scene";
import { PROTECTION_LAYERS, PROTECTION_LIMITS } from "@/lib/show/protection";

function setReducedMotion(reduced: boolean) {
  vi.stubGlobal("matchMedia", (query: string) => ({ matches: reduced && query.includes("reduce"), media: query, addEventListener: () => undefined, removeEventListener: () => undefined }));
}
beforeEach(() => setReducedMotion(true));
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("protection layers", () => {
  it("names a threat, a mechanism, an assumption and a limit for every layer, and points at a real chapter", () => {
    expect(PROTECTION_LAYERS.map((layer) => layer.name)).toEqual(["Split", "Separate", "Check", "Seal the route", "Wipe", "Prove"]);
    for (const layer of PROTECTION_LAYERS) {
      for (const text of [layer.claim, layer.threat, layer.mechanism, layer.relies, layer.limit]) expect(text.length).toBeGreaterThan(15);
      expect(sections.some((section) => section.slug === layer.chapter)).toBe(true);
      // The wording describes the design; it never promises safety.
      expect(`${layer.claim} ${layer.mechanism}`).not.toMatch(/\b(guarantee|unbreakable|impossible|100%|perfectly safe)\b/i);
    }
    expect(PROTECTION_LIMITS.join(" ")).toMatch(/under evaluation/);
  });

  it("draws your data with one guard post per layer and moves the intruder to face the chosen one", () => {
    const places = new Set<string>();
    for (let layer = 0; layer < PROTECTION_LAYERS.length; layer += 1) {
      const scene = protectionScene({ layer });
      const posts = scene.nodes.filter((node) => node.id.startsWith("layer-"));
      expect(posts).toHaveLength(6);
      expect(posts.filter((node) => node.active).map((node) => node.id)).toEqual([`layer-${PROTECTION_LAYERS[layer]!.id}`]);
      expect(posts.filter((node) => node.tone === "good")).toHaveLength(layer);
      expect(posts.map((node) => node.label)).toEqual(PROTECTION_LAYERS.map((entry, index) => `${index + 1}. ${entry.name}`));
      const intruder = scene.nodes.find((node) => node.id === "intruder")!;
      const post = posts[layer]!;
      // The intruder stands outside the ring, on the same side as the post that stops it.
      expect(Math.hypot(intruder.position[0], intruder.position[2])).toBeGreaterThan(Math.hypot(post.position[0], post.position[2]) + 2);
      expect(Math.atan2(intruder.position[2], intruder.position[0])).toBeCloseTo(Math.atan2(post.position[2], post.position[0]));
      places.add(intruder.position.join(","));
      expect(scene.links[0]).toMatchObject({ from: "intruder", to: post.id, tone: "warn" });
      const ids = scene.nodes.map((node) => node.id);
      for (const link of scene.links) expect(ids).toEqual(expect.arrayContaining([link.from, link.to]));
      expect(scene.caption).toContain(`Layer ${layer + 1} of 6`);
    }
    expect(places.size).toBe(6);
  });
});

describe("protection tour", () => {
  it("opens on the first layer with its facts, evidence tag and a live strip the reader can attack", () => {
    render(<ProtectionTour />);
    expect(screen.getByRole("heading", { level: 3, name: "1. Split: no worker gets your data" })).toBeInTheDocument();
    expect(screen.getByText("The threat").nextElementSibling).toHaveTextContent("could read your prompt");
    expect(screen.getByText("Where it stops", { selector: ".tour-facts dt" }).nextElementSibling).toHaveTextContent("matching pieces");
    expect(document.querySelector(".tour-evidence")).toHaveTextContent("Evidence: derived derived from the design, not measured, as tagged in chapter 05, System Architecture.");
    const strip = document.querySelector(".lab-embed") as HTMLElement;
    expect(strip).toHaveAccessibleName("Additive share-grid laboratory");
    fireEvent.change(within(strip).getByLabelText("Your two numbers"), { target: { value: "2, 3" } });
    expect(strip.querySelector(".liveio-out")).toHaveTextContent("Result: 8, 18.");
    fireEvent.click(within(strip).getByRole("button", { name: /^Try to break it/ }));
    expect(strip.querySelector(".liveio-out")).toHaveClass("warn");
    // Embedded, the lab shows only its strip.
    expect(document.querySelector("section.lab")).toBeNull();
    expect(screen.getAllByTestId("view")).toHaveLength(1);
  });

  it("lets the reader pick any layer, each with its own demonstration or the trace page", () => {
    render(<ProtectionTour />);
    const expected = ["Additive share-grid laboratory", "Operation placement and epoch reshuffle", "Exact Freivalds verification", "Interactive onion-routing simulation", "GPU memory lifecycle"];
    PROTECTION_LAYERS.forEach((layer, index) => {
      fireEvent.click(screen.getByRole("button", { name: layer.name }));
      expect(screen.getByRole("button", { name: layer.name })).toHaveAttribute("aria-pressed", "true");
      expect(screen.getByTestId("view")).toHaveTextContent(`Layer ${index + 1} of 6 · ${layer.name}`);
      if (layer.lab) expect(document.querySelector(".lab-embed")).toHaveAccessibleName(expected[index]);
      else expect(screen.getByRole("link", { name: "trace verification page" })).toHaveAttribute("href", "/verify");
      expect(screen.getByRole("link", { name: /^chapter \d\d,/ })).toHaveAttribute("href", `/${layer.chapter}`);
    });
  });

  it("plays through the layers only when asked or allowed", () => {
    vi.useFakeTimers();
    const { unmount } = render(<ProtectionTour autoPlay />);
    act(() => { vi.advanceTimersByTime(20_000); });
    expect(screen.getByTestId("view")).toHaveTextContent("Layer 1 of 6");
    fireEvent.click(screen.getByRole("button", { name: "Play the tour" }));
    act(() => { vi.advanceTimersByTime(9000); });
    expect(screen.getByTestId("view")).toHaveTextContent("Layer 2 of 6");
    fireEvent.click(screen.getByRole("button", { name: "Wipe" }));
    act(() => { vi.advanceTimersByTime(20_000); });
    expect(screen.getByTestId("view")).toHaveTextContent("Layer 5 of 6");
    unmount();
    setReducedMotion(false);
    render(<ProtectionTour autoPlay />);
    act(() => { vi.advanceTimersByTime(9000); });
    expect(screen.getByTestId("view")).toHaveTextContent("Layer 2 of 6");
  });

  it("states what it does not protect against, on an accessible page", async () => {
    const { container } = render(<ProtectionPage />);
    expect(screen.getByRole("heading", { level: 1, name: "How your data is protected" })).toBeInTheDocument();
    const limits = screen.getByRole("heading", { level: 3, name: "What this does not protect against" }).closest("section")!;
    expect(within(limits).getAllByRole("listitem")).toHaveLength(PROTECTION_LIMITS.length);
    expect(limits).toHaveTextContent("its claims are under evaluation");
    expect(container.querySelector(".hero .evidence")).toHaveTextContent("Designed for defence in depth · claims under evaluation");
    expect((await axe(container)).violations).toHaveLength(0);
  }, 20000);
});
