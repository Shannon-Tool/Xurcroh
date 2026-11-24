import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { axe } from "jest-axe";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { wordsSetting } from "@/lib/scene/settings";
import { questionSetting } from "@/lib/show/question";
import { LiveShow } from "@/components/LiveShow";
import { getModel, seededRng, type TextModel } from "@/lib/simulation/show";

const first = getModel("text-shares") as TextModel;
const SLOW = 7 * 1100;
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
  wordsSetting.set("plain");
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
    expect(screen.getByText(first.prompt, { selector: ".show-prompt .sr-only" })).toBeInTheDocument();
    expect(screen.getByText(/^Step 1 of 8 · Split\./)).toBeInTheDocument();
    expect(screen.getByText(/One piece alone is noise/)).toBeInTheDocument();
    expect(answer()).toBe("");
    advance(SLOW - 50);
    expect(answer()).toBe("");
    advance(100);
    expect(answer()).toBe(first.tokens[0]);
    expect(screen.getByText(/^Step 8 of 8 · Combine\./)).toBeInTheDocument();
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
    expect(screen.getByText("A sunrise over the water", { selector: ".show-prompt .sr-only" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Picture" })).toHaveAttribute("aria-pressed", "true");
    expect(document.querySelector(".show-answer")).toBeNull();
    advance(SLOW + 1100 + 750 * 14 + 3600 + 100);
    expect(screen.getByRole("button", { name: "Clip" })).toHaveAttribute("aria-pressed", "true");
    expect(document.querySelectorAll(".show-film canvas")).toHaveLength(12);
  }, 30000);

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
    expect(screen.getByText("A sunrise over the water", { selector: ".show-prompt .sr-only" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "City at night" }));
    expect(screen.getByText("A city at night under the moon", { selector: ".show-prompt .sr-only" })).toBeInTheDocument();
    advance(SLOW + 750 * 14 + 1000);
    expect(screen.getByText(/^Complete\./)).toBeInTheDocument();
    // The tour ended when the reader chose: the same picture starts again by itself instead of moving on.
    advance(3600 + 750 * 3);
    expect(screen.getByRole("button", { name: "Picture" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "City at night" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText(/^Step [1-4] of 14\./)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Pause" })).toBeInTheDocument();
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
    expect(screen.getByText(/sent back a wrong answer\. Your device caught it/, { selector: ".stage-caption p" })).toBeInTheDocument();
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

  it("works on the reader's own text: typed in, written backwards from the recombined pieces", () => {
    render(<LiveShow autoPlay rng={seededRng(10)} />);
    fireEvent.click(screen.getByRole("button", { name: "Your text" }));
    const input = screen.getByLabelText("Type something");
    expect(input).toHaveValue("hello world");
    fireEvent.change(input, { target: { value: "abc def" } });
    expect(answer()).toBe("");
    advance(300 * 3 + 50);
    expect(answer()).toBe("fed");
    advance(300 * 8);
    expect(answer()).toBe("fed cba");
    expect(screen.getByRole("status")).toHaveTextContent("Answer complete: fed cba");
    expect(screen.getByTestId("checks")).toHaveTextContent("28");
    expect(document.querySelector(".show-toy")).toHaveTextContent("writes your text backwards");
  });

  it("works on the reader's own drawing and offers ready-made ones", () => {
    render(<LiveShow rng={seededRng(11)} />);
    fireEvent.click(screen.getByRole("button", { name: "Your drawing" }));
    expect(screen.getByRole("img", { name: /^Drawing pad/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Your drawing" })).toHaveAttribute("aria-pressed", "true");
    advance(750 * 4 + 50);
    expect(screen.getByTestId("checks")).toHaveTextContent("16");
    expect(screen.getByText(/^Complete\./)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Arrow" }));
    expect(screen.getByTestId("checks")).toHaveTextContent("0");
    advance(750 + 50);
    expect(screen.getByTestId("checks")).toHaveTextContent(/^4$/);
  });

  it("shows what each other party gets: noise, noise, the real thing for matching pieces, ciphertext on the wire", async () => {
    render(<LiveShow autoPlay rng={seededRng(12)} />);
    advance(SLOW + 1100 + 300 * 2 + 50);
    fireEvent.click(screen.getByRole("button", { name: "Pause" }));
    const shown = answer();
    expect(shown.length).toBeGreaterThan(5);
    fireEvent.click(screen.getByRole("button", { name: "One worker" }));
    const data = () => document.querySelector(".show-eye-data")!.textContent!;
    expect(data()).toMatch(/^([0-9a-f]{4} ?){20,}$/);
    expect(document.querySelector(".show-answer")).toBeNull();
    expect(document.querySelector(".show-eye-note")).toHaveTextContent("This is everything it has.");
    const alone = data();
    fireEvent.click(screen.getByRole("button", { name: "Two workers, same piece" }));
    expect(data()).not.toBe(alone);
    expect(document.querySelector(".show-eye-note")).toHaveTextContent("still have only noise");
    fireEvent.click(screen.getByRole("button", { name: "Two workers, matching pieces" }));
    expect(answer()).toBe(shown);
    expect(document.querySelector(".show-output")).toHaveClass("warn");
    expect(document.querySelector(".show-eye-note")).toHaveTextContent("This is where the design stops");
    vi.useRealTimers();
    fireEvent.click(screen.getByRole("button", { name: "The network" }));
    await vi.waitFor(() => expect(document.querySelector(".show-eye-note")).toHaveTextContent(/sees [1-9]\d* encrypted bytes/));
    expect(document.querySelector(".show-eye-data")!.textContent).toMatch(/^([0-9a-f]{2} ?){20,}$/);
    fireEvent.click(screen.getByRole("button", { name: "You" }));
    expect(answer()).toBe(shown);
    expect(document.querySelector(".show-eye-note")).toBeNull();
  });

  it("switches between everyday and technical wording and remembers the choice", () => {
    render(<LiveShow rng={seededRng(13)} />);
    expect(screen.getByText(/^Step 1 of 8 · Split\./)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Words: plain" }));
    expect(screen.getByText(/^Split: the client permutes its input .* into two additive shares\./)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Words: technical" })).toBeInTheDocument();
    expect(wordsSetting.get()).toBe("technical");
    expect(window.localStorage.getItem("xurcroh-words")).toBe("technical");
  });

  it("has a compact form for chapters that follows the question chosen for the labs and diagrams", () => {
    render(<LiveShow compact autoPlay rng={seededRng(14)} />);
    expect(screen.queryByRole("heading", { name: "Live demonstration" })).toBeNull();
    expect(screen.getByRole("region", { name: "Live demonstration of the chosen question" })).toBeInTheDocument();
    expect(screen.getByLabelText("Your question")).toHaveValue("text-shares");
    expect(screen.queryByRole("button", { name: "Your drawing" })).toBeNull();
    expect(screen.queryByText("What is real here")).toBeNull();
    expect(document.querySelector(".show-truth")).toHaveTextContent("Designed for defence in depth; claims under evaluation");
    advance(SLOW + 50);
    expect(answer()).toBe(first.tokens[0]);
    fireEvent.change(screen.getByLabelText("Your question"), { target: { value: "text-operators" } });
    expect(answer()).toBe("");
    expect(screen.getByText("Who runs the workers?", { selector: ".show-prompt .sr-only" })).toBeInTheDocument();
    advance(SLOW + 50);
    expect(answer()).toBe("The");
    questionSetting.set("text-shares");
  });

  it("shows the data in every form it takes and lights the form of the current step", () => {
    render(<LiveShow autoPlay rng={seededRng(15)} />);
    const forms = () => [...document.querySelectorAll(".ribbon-form")];
    const lit = () => forms().filter((form) => form.getAttribute("aria-current") === "step").map((form) => form.querySelector(".ribbon-label")!.textContent);
    expect(forms().map((form) => form.querySelector(".ribbon-label")!.textContent)).toEqual(["Readable", "Numbers", "Shuffled", "Two pieces", "Sealed ×3", "In enclave", "Checked", "Answer"]);
    expect(lit()).toEqual(["Numbers", "Shuffled", "Two pieces"]);
    advance(1100 * 2 + 50);
    expect(lit()).toEqual(["Sealed ×3"]);
    expect(screen.getByText(/^Step 3 of 8 · Seal\./)).toBeInTheDocument();
    advance(SLOW - 1100 * 2);
    expect(lit()).toEqual(["Answer"]);
    // Real values of the step that just ran: two different random pieces, and a result that is neither.
    const value = (label: string) => forms().find((form) => form.querySelector(".ribbon-label")!.textContent === label)!.querySelector(".sr-only")!.textContent!;
    expect(value("Two pieces")).toMatch(/^([0-9a-f]{4} ){2}[0-9a-f]{4} \| ([0-9a-f]{4} ){2}[0-9a-f]{4}$/);
    expect(value("Numbers")).toMatch(/^([0-9a-f]{4} ?){6}$/);
    expect(value("Shuffled")).toMatch(/^([0-9a-f]{4} ?){6}$/);
    expect(value("Checked")).toBe("4 of 4 pass");
    expect(value("Answer")).toBe(first.tokens[0]);
    expect(document.querySelector(".stage-leaves")).toHaveTextContent("What leaves your device");
  });

  it("refuses a worker whose software was swapped, using a real signature check", async () => {
    vi.useRealTimers();
    setReducedMotion(true);
    render(<LiveShow rng={seededRng(16)} />);
    fireEvent.click(screen.getByRole("button", { name: "Next step" }));
    await vi.waitFor(() => expect(document.querySelectorAll(".show-tile")[3]).toHaveTextContent("Worker (1,1) ✓"));
    fireEvent.click(screen.getByRole("button", { name: "Swap a worker's software" }));
    fireEvent.click(screen.getByRole("button", { name: "Next step" }));
    expect(screen.getByText(/is not running the expected software/, { selector: ".stage-caption p" })).toBeInTheDocument();
    expect(document.querySelector(".show-rejected")).toHaveTextContent(/report was refused\. Nothing was sent to it/);
    await vi.waitFor(() => expect(document.querySelectorAll(".show-tile.bad")[0]).toHaveTextContent("✗"));
    await vi.waitFor(() => expect(screen.getByTestId("rejected")).toHaveTextContent("1"));
    expect(answer()).toBe(first.tokens[0]);
    fireEvent.click(screen.getByRole("button", { name: "Next step" }));
    expect(answer()).toBe(first.tokens.slice(0, 2).join(""));
    await vi.waitFor(() => expect(document.querySelectorAll(".show-tile.bad")).toHaveLength(0));
    await vi.waitFor(() => expect([...document.querySelectorAll(".show-tile")].every((tile) => tile.textContent!.includes("✓"))).toBe(true));
  });

  it("stops an altered sealed packet at a relay, using the real authenticity check", async () => {
    vi.useRealTimers();
    setReducedMotion(true);
    render(<LiveShow rng={seededRng(17)} />);
    fireEvent.click(screen.getByRole("button", { name: "Next step" }));
    const sealed = () => [...document.querySelectorAll(".ribbon-form")][4]!;
    await vi.waitFor(() => expect(sealed().querySelector(".sr-only")!.textContent).toMatch(/^([0-9a-f]{2} ?){10}$/));
    expect(sealed()).toHaveClass("ribbon-cipher");
    fireEvent.click(screen.getByRole("button", { name: "Alter a sealed packet" }));
    fireEvent.click(screen.getByRole("button", { name: "Next step" }));
    expect(screen.getByText(/Someone altered a sealed packet\. Relay \d noticed/, { selector: ".stage-caption p" })).toBeInTheDocument();
    await vi.waitFor(() => expect(sealed()).toHaveClass("ribbon-warn"));
    expect(sealed().querySelector(".ribbon-label")).toHaveTextContent("Refused");
    await vi.waitFor(() => expect(screen.getByTestId("rejected")).toHaveTextContent("1"));
    fireEvent.click(screen.getByRole("button", { name: "Next step" }));
    await vi.waitFor(() => expect(sealed()).toHaveClass("ribbon-cipher"));
    expect(answer()).toBe(first.tokens.slice(0, 2).join(""));
  });

  it("shows a relay and the machine's operator only sealed bytes", async () => {
    vi.useRealTimers();
    setReducedMotion(true);
    render(<LiveShow rng={seededRng(18)} />);
    fireEvent.click(screen.getByRole("button", { name: "Next step" }));
    fireEvent.click(screen.getByRole("button", { name: "A Tor relay" }));
    await vi.waitFor(() => expect(document.querySelector(".show-eye-note")).toHaveTextContent(/middle relay sees [1-9]\d* sealed bytes and the next stop/));
    expect(document.querySelector(".show-eye-data")!.textContent).toMatch(/^([0-9a-f]{2} ?){20,}$/);
    fireEvent.click(screen.getByRole("button", { name: "The machine's operator" }));
    await vi.waitFor(() => expect(document.querySelector(".show-eye-note")).toHaveTextContent("one more layer and not as proof"));
  });

  it("says plainly what is real and what is prepared", () => {
    render(<LiveShow rng={seededRng(7)} />);
    const truth = document.querySelector(".show-truth")!;
    expect(truth).toHaveTextContent(/Simulation in your browser/);
    expect(truth).toHaveTextContent(/prepared toy models/);
    expect(truth).toHaveTextContent(/the relays and enclaves are stand-ins inside this page/);
    expect(truth).toHaveTextContent(/Designed for defence in depth; claims under evaluation/);
    const panel = screen.getByText("What is real here").closest("details")!;
    expect(within(panel).getByText(/No language or image model runs/)).toBeInTheDocument();
    expect(within(panel).getByText(/The three relays stand in for a Tor circuit/)).toBeInTheDocument();
    expect(within(panel).getByText(/an enclave is one more layer, not proof/)).toBeInTheDocument();
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
