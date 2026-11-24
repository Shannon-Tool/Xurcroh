import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { axe } from "jest-axe";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/scene/stage", () => ({
  createStage: () => ({ rendererName: "Test GPU", setPalette: () => undefined, setQuality: () => undefined, draw: () => ({ labels: [], animating: false }), pick: () => null, dispose: () => undefined }),
}));

import Home from "@/app/page";
import LaboratoryPage from "@/app/laboratory/page";
import { ArchitectureExplorer } from "@/components/ArchitectureExplorer";

function setReducedMotion(reduced: boolean) {
  vi.stubGlobal("matchMedia", (query: string) => ({ matches: reduced && query.includes("reduce"), media: query, addEventListener: () => undefined, removeEventListener: () => undefined }));
}
const caption = (container: HTMLElement) => container.querySelector(".scene3d-caption")!.textContent;

beforeEach(() => {
  vi.stubGlobal("WebGLRenderingContext", class {});
  setReducedMotion(true);
});
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("architecture explorer", () => {
  it("shows the chosen step and marks it as current", () => {
    const { container } = render(<ArchitectureExplorer />);
    expect(caption(container)).toMatch(/^1\. Split/);
    fireEvent.click(screen.getByRole("button", { name: /Return/ }));
    expect(caption(container)).toMatch(/^4\. Return/);
    expect(screen.getByRole("button", { name: /Return/ })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: /Split/ })).toHaveAttribute("aria-pressed", "false");
  });

  it("plays through the steps in a loop and pauses when the reader picks one", () => {
    setReducedMotion(false);
    vi.useFakeTimers();
    const { container } = render(<ArchitectureExplorer autoPlay />);
    expect(screen.getByRole("button", { name: "Pause" })).toBeInTheDocument();
    act(() => { vi.advanceTimersByTime(3700); });
    expect(caption(container)).toMatch(/^2\. Send/);
    act(() => { vi.advanceTimersByTime(3600 * 5); });
    expect(caption(container)).toMatch(/^1\. Split/);
    fireEvent.click(screen.getByRole("button", { name: /Verify/ }));
    expect(screen.getByRole("button", { name: "Play" })).toBeInTheDocument();
    act(() => { vi.advanceTimersByTime(8000); });
    expect(caption(container)).toMatch(/^5\. Verify/);
  });

  it("does not start by itself for readers who prefer reduced motion", () => {
    render(<ArchitectureExplorer autoPlay />);
    expect(screen.getByRole("button", { name: "Play" })).toBeInTheDocument();
  });
});

describe("home page", () => {
  it("opens with the design overview, says what it is, and stays accessible", async () => {
    const { container } = render(<Home />);
    expect(screen.getByRole("heading", { name: "The design in one picture" })).toBeInTheDocument();
    expect(screen.getByText(/not a live system and shows no measurements/)).toBeInTheDocument();
    expect(screen.getByText(/designed for defence\s+in depth with claims under evaluation/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Open the laboratory" })).toHaveAttribute("href", "/laboratory");
    await waitFor(() => expect(container.querySelector(".scene3d canvas")).not.toBeNull());
    expect((await axe(container)).violations).toHaveLength(0);
  }, 20000);
});

describe("laboratory navigator", () => {
  it("links to every laboratory on the page, in order", () => {
    const { container } = render(<LaboratoryPage />);
    const targets = [...container.querySelectorAll(".lab-nav a")].map((link) => link.getAttribute("href"));
    const labs = [...container.querySelectorAll("section.lab")].map((lab) => `#${lab.id}`);
    expect(labs).toHaveLength(11);
    expect(targets).toEqual(labs);
  });
});
