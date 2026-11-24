import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { axe } from "jest-axe";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LiveShow } from "@/components/LiveShow";
import { getModel, seededRng, type TextModel } from "@/lib/simulation/show";

const first = getModel("text-shares") as TextModel;
const SLOW = 5 * 1100;
const answer = () => document.querySelector(".show-answer")!.textContent!.replace("▌", "");

function setReducedMotion(reduced: boolean) {
  vi.stubGlobal("matchMedia", (query: string) => ({ matches: reduced && query.includes("reduce"), media: query, addEventListener: () => undefined, removeEventListener: () => undefined }));
}
// React applies a tick's update when `act` returns, so time moves in small slices to let each beat schedule the next.
function advance(ms: number) {
  for (let elapsed = 0; elapsed < ms; elapsed += 50) act(() => { vi.advanceTimersByTime(Math.min(50, ms - elapsed)); });
}

beforeEach(() => {
  // jsdom has no canvas drawing; the show must run without it. No WebGL either, so the 3D view says so.
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(() => null);
  setReducedMotion(false);
  vi.useFakeTimers();
});
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("live demonstration", () => {
  it("starts by itself, plays the first token slowly and then streams the prepared answer from the share arithmetic", () => {
    render(<LiveShow autoPlay rng={seededRng(1)} />);
    expect(screen.getByText(first.prompt, { selector: ".show-prompt *" })).toBeInTheDocument();
    expect(screen.getByText(/^Split:/)).toBeInTheDocument();
    expect(answer()).toBe("");
    advance(SLOW - 50);
    expect(answer()).toBe("");
    advance(100);
    expect(answer()).toBe(first.tokens[0]);
    expect(screen.getByText(/^Combine:/)).toBeInTheDocument();
    advance(1100 + 300 * 3);
    expect(answer()).toBe(first.tokens.slice(0, 5).join(""));
    expect(screen.getByText(/Token 5 of/)).toBeInTheDocument();
    expect(screen.getByTestId("checks")).toHaveTextContent("20");
    advance(300 * first.tokens.length);
    expect(answer()).toBe(first.answer);
    expect(screen.getByRole("status")).toHaveTextContent(`Answer complete: ${first.answer}`);
  });

  it("moves on to a picture and then a clip while nobody touches it", () => {
    render(<LiveShow autoPlay rng={seededRng(2)} />);
    advance(SLOW + 1100 + 300 * first.tokens.length + 3600 + 100);
    expect(screen.getByText("A sunrise over the water", { selector: ".show-prompt *" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Picture" })).toHaveAttribute("aria-pressed", "true");
    expect(document.querySelector(".show-answer")).toBeNull();
    advance(SLOW + 1100 + 750 * 14 + 3600 + 100);
    expect(screen.getByRole("button", { name: "Clip" })).toHaveAttribute("aria-pressed", "true");
    expect(document.querySelectorAll(".show-film canvas")).toHaveLength(12);
  });

  it("does not start by itself under reduced motion, and steps on request", () => {
    setReducedMotion(true);
    render(<LiveShow autoPlay rng={seededRng(3)} />);
    advance(20_000);
    expect(answer()).toBe("");
    fireEvent.click(screen.getByRole("button", { name: "Next step" }));
    expect(answer()).toBe(first.tokens[0]);
    fireEvent.click(screen.getByRole("button", { name: "Next step" }));
    expect(answer()).toBe(first.tokens.slice(0, 2).join(""));
    advance(20_000);
    expect(answer()).toBe(first.tokens.slice(0, 2).join(""));
  });

  it("lets the reader choose what is produced, and stops the tour", () => {
    render(<LiveShow autoPlay rng={seededRng(4)} />);
    fireEvent.click(screen.getByRole("button", { name: "Picture" }));
    expect(screen.getByText("A sunrise over the water", { selector: ".show-prompt *" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "City at night" }));
    expect(screen.getByText("A city at night under the moon", { selector: ".show-prompt *" })).toBeInTheDocument();
    advance(SLOW + 750 * 14 + 3600 + 5000);
    // Finished and stayed: the tour ended when the reader chose.
    expect(screen.getByText(/Step 14 of 14|Complete/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Picture" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "Play" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Text" }));
    fireEvent.click(screen.getByRole("button", { name: "What does this page not show?" }));
    advance(SLOW + 100);
    expect(answer()).toBe("This");
  });

  it("shows a real rejection when a worker is tampered with, then carries on", () => {
    render(<LiveShow autoPlay rng={seededRng(5)} />);
    advance(SLOW + 100);
    fireEvent.click(screen.getByRole("button", { name: "Tamper with a worker" }));
    expect(screen.getByRole("button", { name: "Tamper with a worker" })).toBeDisabled();
    advance(1100);
    expect(screen.getByText(/returned a product that fails the check/, { selector: ".show-caption" })).toBeInTheDocument();
    expect(document.querySelector(".show-rejected")).toHaveTextContent(/Worker \(\d,\d\) failed the check/);
    expect(document.querySelectorAll(".show-tile.bad")).toHaveLength(1);
    expect(screen.getByTestId("rejected")).toHaveTextContent("1");
    expect(answer()).toBe(first.tokens[0]);
    advance(2400 + 50);
    expect(document.querySelector(".show-rejected")).toBeNull();
    expect(document.querySelectorAll(".show-tile.bad")).toHaveLength(0);
    expect(answer()).toBe(first.tokens.slice(0, 2).join(""));
    expect(screen.getByTestId("rejected")).toHaveTextContent("1");
    expect(screen.getByTestId("checks")).toHaveTextContent("12");
  });

  it("pauses, restarts and changes speed", () => {
    render(<LiveShow autoPlay rng={seededRng(6)} />);
    advance(SLOW + 1100 + 300 * 2 + 50);
    fireEvent.click(screen.getByRole("button", { name: "Pause" }));
    const held = answer();
    advance(5000);
    expect(answer()).toBe(held);
    fireEvent.click(screen.getByRole("button", { name: "Speed: 1×" }));
    expect(screen.getByRole("button", { name: "Speed: 2×" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Restart" }));
    expect(answer()).toBe("");
    advance(SLOW / 2 + 60);
    expect(answer()).toBe(first.tokens[0]);
  });

  it("stands still while the tab is hidden or the show is scrolled away, and resumes", () => {
    let report: (entries: Array<{ isIntersecting: boolean }>) => void = () => undefined;
    vi.stubGlobal("IntersectionObserver", class {
      constructor(callback: (entries: Array<{ isIntersecting: boolean }>) => void) { report = callback; }
      observe() { return undefined; }
      disconnect() { return undefined; }
    });
    render(<LiveShow autoPlay rng={seededRng(9)} />);
    advance(SLOW + 50);
    expect(answer()).toBe(first.tokens[0]);
    act(() => report([{ isIntersecting: false }]));
    advance(5000);
    expect(answer()).toBe(first.tokens[0]);
    act(() => report([{ isIntersecting: true }]));
    advance(1100 + 50);
    expect(answer()).toBe(first.tokens.slice(0, 2).join(""));
    const hidden = vi.spyOn(document, "hidden", "get").mockReturnValue(true);
    act(() => { document.dispatchEvent(new Event("visibilitychange")); });
    advance(5000);
    expect(answer()).toBe(first.tokens.slice(0, 2).join(""));
    hidden.mockReturnValue(false);
    act(() => { document.dispatchEvent(new Event("visibilitychange")); });
    advance(300 + 50);
    expect(answer()).toBe(first.tokens.slice(0, 3).join(""));
  });

  it("says plainly what is real and what is prepared", () => {
    render(<LiveShow rng={seededRng(7)} />);
    const truth = document.querySelector(".show-truth")!;
    expect(truth).toHaveTextContent(/Simulation in your browser/);
    expect(truth).toHaveTextContent(/prepared toy models/);
    expect(truth).toHaveTextContent(/Designed for defence in depth; claims under evaluation/);
    const panel = screen.getByText("What is real here").closest("details")!;
    expect(within(panel).getByText(/No language or image model runs/)).toBeInTheDocument();
    expect(within(panel).getByText(/measures nothing/)).toBeInTheDocument();
  });

  it("has no accessibility violations", async () => {
    vi.useRealTimers();
    setReducedMotion(true);
    const { container } = render(<LiveShow rng={seededRng(8)} />);
    fireEvent.click(screen.getByRole("button", { name: "Next step" }));
    expect((await axe(container)).violations).toHaveLength(0);
  }, 20000);
});
