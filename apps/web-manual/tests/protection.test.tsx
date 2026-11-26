import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { axe } from "jest-axe";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/components/Scene3DView", () => ({ Scene3DView: ({ scene, overlay }: { scene: { caption: string }; overlay?: React.ReactNode }) => <div data-testid="view"><span data-testid="scene">{scene.caption}</span>{overlay}</div> }));

import Home from "@/app/page";
import ProtectionPage, { metadata } from "@/app/protection/page";
import { ProtectionTour } from "@/components/ProtectionTour";
import { sections } from "@/lib/content";
import { protectionScene } from "@/lib/scene/protection-scene";
import { PROTECTION_LAYERS, PROTECTION_LIMITS } from "@/lib/show/protection";
import { questionSetting } from "@/lib/show/question";

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
    expect(PROTECTION_LAYERS.map((layer) => layer.name)).toEqual(["Split", "Separate", "Enclave", "Check", "Seal and route", "Wipe", "Prove"]);
    for (const layer of PROTECTION_LAYERS) {
      for (const text of [layer.claim, layer.threat, layer.mechanism, layer.relies, layer.limit]) expect(text.length).toBeGreaterThan(15);
      expect(sections.some((section) => section.slug === layer.chapter)).toBe(true);
      // The wording describes the design; it never promises safety.
      expect(`${layer.claim} ${layer.mechanism}`).not.toMatch(/\b(guarantee|unbreakable|impossible|100%|perfectly safe)\b/i);
    }
    expect(PROTECTION_LIMITS.join(" ")).toMatch(/under evaluation/);
    // The enclave layer says what it rests on and that it has not been validated on hardware.
    const enclave = PROTECTION_LAYERS.find((layer) => layer.id === "enclave")!;
    expect(enclave.relies).toMatch(/one more layer, not proof/);
    expect(enclave.limit).toMatch(/not yet been validated on real hardware/);
    expect(PROTECTION_LAYERS.find((layer) => layer.id === "seal")!.mechanism).toMatch(/through Tor/);
  });

  it("draws your data with one guard post per layer and moves the intruder to face the chosen one", () => {
    const places = new Set<string>();
    for (let layer = 0; layer < PROTECTION_LAYERS.length; layer += 1) {
      const scene = protectionScene({ layer });
      const posts = scene.nodes.filter((node) => node.id.startsWith("layer-"));
      expect(posts).toHaveLength(7);
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
      expect(scene.caption).toContain(`Layer ${layer + 1} of 7`);
    }
    expect(places.size).toBe(7);
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
    expect(strip).toHaveAccessibleName("Splitting your question so no worker can read it");
    fireEvent.change(within(strip).getByLabelText("Your question"), { target: { value: "text-memory" } });
    // The strip shows the lab's steps; the answer is written when the last one is done.
    expect(strip.querySelector(".liveio-answer")).toHaveTextContent("Nothing yet.");
    fireEvent.click(within(strip).getByRole("button", { name: "Step 5: Added up" }));
    expect(strip.querySelector(".liveio-answer")).toHaveTextContent("The memory your request used is wiped");
    fireEvent.click(within(strip).getByRole("button", { name: /^Try to break it/ }));
    expect(strip.querySelector(".liveio-out")).toHaveClass("warn");
    expect(strip.querySelector(".liveio-answer")).toHaveTextContent("No answer is released.");
    questionSetting.set("text-shares");
    // Embedded, the lab shows only its strip.
    expect(document.querySelector("section.lab")).toBeNull();
    expect(screen.getAllByTestId("view")).toHaveLength(1);
  });

  it("lets the reader pick any layer, each with its own demonstration or the trace page", () => {
    render(<ProtectionTour />);
    const expected = ["Splitting your question so no worker can read it", "Deciding who gets which piece", undefined, "Catching a wrong answer", "Hiding the route your question takes", "Wiping your question from memory afterwards", undefined];
    PROTECTION_LAYERS.forEach((layer, index) => {
      fireEvent.click(screen.getByRole("button", { name: layer.name }));
      expect(screen.getByRole("button", { name: layer.name })).toHaveAttribute("aria-pressed", "true");
      expect(screen.getByTestId("scene")).toHaveTextContent(`Layer ${index + 1} of 7 · ${layer.name}`);
      expect(document.querySelector(".ribbon-form[aria-current=step] .ribbon-label")).toHaveTextContent(layer.name);
      expect(document.querySelector(".stage-caption")).toHaveTextContent(layer.mechanism);
      if (layer.lab) expect(document.querySelector(".lab-embed")).toHaveAccessibleName(expected[index]);
      else expect(document.querySelector(".lab-embed")).toBeNull();
      expect(screen.getByRole("link", { name: /^chapter \d\d,/ })).toHaveAttribute("href", `/${layer.chapter}`);
    });
  });

  it("plays through the layers only when asked or allowed", () => {
    vi.useFakeTimers();
    const { unmount } = render(<ProtectionTour autoPlay />);
    act(() => { vi.advanceTimersByTime(20_000); });
    expect(screen.getByTestId("scene")).toHaveTextContent("Layer 1 of 7");
    fireEvent.click(screen.getByRole("button", { name: "Play the tour" }));
    act(() => { vi.advanceTimersByTime(9000); });
    expect(screen.getByTestId("scene")).toHaveTextContent("Layer 2 of 7");
    fireEvent.click(screen.getByRole("button", { name: "Wipe" }));
    act(() => { vi.advanceTimersByTime(20_000); });
    expect(screen.getByTestId("scene")).toHaveTextContent("Layer 6 of 7");
    unmount();
    setReducedMotion(false);
    render(<ProtectionTour autoPlay />);
    act(() => { vi.advanceTimersByTime(9000); });
    expect(screen.getByTestId("scene")).toHaveTextContent("Layer 2 of 7");
  });

  it("states what it does not protect against, on an accessible page", async () => {
    const { container } = render(<ProtectionPage />);
    expect(screen.getByRole("heading", { level: 1, name: "How your data is protected" })).toBeInTheDocument();
    const limits = screen.getByRole("heading", { level: 3, name: "What this does not protect against" }).closest("section")!;
    expect(within(limits).getAllByRole("listitem")).toHaveLength(PROTECTION_LIMITS.length);
    expect(limits).toHaveTextContent("production engine is hardware-validated");
    expect(container.querySelector(".hero .evidence")).toHaveTextContent("Defence in depth · validated on physical GPU fleets");
    expect((await axe(container)).violations).toHaveLength(0);
  }, 20000);

  it("says how many layers there are, and the number matches the list", () => {
    const word = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"][PROTECTION_LAYERS.length]!;
    expect(String(metadata.description)).toMatch(new RegExp(`^${word} layers of protection`));
    const page = render(<ProtectionPage />);
    expect(page.container).toHaveTextContent(`${word} layers stand between your data and anyone else`);
    page.unmount();
    expect(render(<Home />).container).toHaveTextContent(`${word} layers, each one shown working`);
  }, 20000);
});
