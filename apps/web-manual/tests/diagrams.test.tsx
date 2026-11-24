import { act, fireEvent, render, screen } from "@testing-library/react";
import { axe } from "jest-axe";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ArchitectureDiagram } from "@/components/diagrams/ArchitectureDiagram";
import { MemoryLifecycleDiagram } from "@/components/diagrams/MemoryLifecycleDiagram";
import { RuntimeFlowDiagram } from "@/components/diagrams/RuntimeFlowDiagram";
import { Figure } from "@/components/Figure";
import { architectureData, memoryData, runtimeData } from "@/lib/show/diagram-data";
import { journeyOf } from "@/lib/show/journey";
import { questionSetting } from "@/lib/show/question";
import { SHOW_PRIME } from "@/lib/simulation/show";

describe("chapter diagrams", () => {
  it("draws the architecture with four role workers and a described picture", async () => {
    const { container } = render(<ArchitectureDiagram caption="Figure 5.1" />);
    const picture = screen.getByRole("img", { name: /Fortress system architecture/ });
    expect(picture).toHaveAccessibleDescription(/never sees the prompt/);
    expect(container.querySelectorAll("text.dg-strong")).toHaveLength(9);
    for (const role of ["(0,0)", "(0,1)", "(1,0)", "(1,1)"]) expect(screen.getByText(`Worker ${role}`, { selector: "text" })).toBeInTheDocument();
    expect(screen.getByText("Figure 5.1")).toBeInTheDocument();
    expect((await axe(container)).violations).toHaveLength(0);
  });

  it("draws the runtime flow and the memory lifecycle with the unsafe branch", async () => {
    const runtime = render(<RuntimeFlowDiagram caption="Figure 7.1" />);
    expect(screen.getByRole("img", { name: /One token through the inference runtime/ })).toBeInTheDocument();
    expect((await axe(runtime.container)).violations).toHaveLength(0);
    runtime.unmount();
    const memory = render(<MemoryLifecycleDiagram caption="Figure 9.1" />);
    expect(screen.getByText(/synchronisation skipped/)).toBeInTheDocument();
    expect(memory.container.querySelector("path.dg-warn")).not.toBeNull();
    expect((await axe(memory.container)).violations).toHaveLength(0);
  });

  it("gives two diagrams on one page different arrow markers", () => {
    const { container } = render(<><RuntimeFlowDiagram caption="a" /><MemoryLifecycleDiagram caption="b" /></>);
    const ids = [...container.querySelectorAll("marker")].map((marker) => marker.id);
    expect(new Set(ids).size).toBe(2);
  });

  it("shows only the caption when a figure has no drawing", () => {
    const { container } = render(<Figure caption="Only words" />);
    expect(container.querySelector("svg")).toBeNull();
    expect(screen.getByText("Only words")).toBeInTheDocument();
  });

  describe("walk-through", () => {
    afterEach(() => vi.useRealTimers());

    it("lights one stage at a time and dims the rest", () => {
      const { container } = render(<ArchitectureDiagram caption="c" />);
      const lit = () => container.querySelectorAll(".dg-on").length;
      const dimmed = () => container.querySelectorAll(".dg-dim").length;
      expect(lit() + dimmed()).toBe(0);
      expect(screen.getByRole("button", { name: "Back" })).toBeDisabled();
      fireEvent.click(screen.getByRole("button", { name: "Next" }));
      expect(screen.getByText(/1 of 7 · Keep the prompt local/)).toBeInTheDocument();
      expect(lit()).toBe(1);
      expect(dimmed()).toBe(14);
      for (let index = 0; index < 3; index += 1) fireEvent.click(screen.getByRole("button", { name: "Next" }));
      expect(screen.getByText(/4 of 7 · Send shares/)).toBeInTheDocument();
      expect(lit()).toBe(6);
      fireEvent.click(screen.getByRole("button", { name: "Back" }));
      expect(screen.getByText(/3 of 7 · Split/)).toBeInTheDocument();
      fireEvent.click(screen.getByRole("button", { name: "Show all" }));
      expect(lit() + dimmed()).toBe(0);
    });

    it("plays through every step and stops on the last one", () => {
      vi.useFakeTimers();
      render(<MemoryLifecycleDiagram caption="c" />);
      fireEvent.click(screen.getByRole("button", { name: "Play walk-through" }));
      expect(screen.getByText(/1 of 7 · Allocate/)).toBeInTheDocument();
      act(() => { vi.advanceTimersByTime(2700); });
      expect(screen.getByText(/2 of 7 · Execute/)).toBeInTheDocument();
      act(() => { vi.advanceTimersByTime(2700 * 8); });
      expect(screen.getByText(/7 of 7 · If synchronisation is skipped/)).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Play walk-through" })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Next" })).toBeDisabled();
    });

    it("lights the arrows of the box the reader points at", () => {
      const { container } = render(<ArchitectureDiagram caption="c" />);
      fireEvent.mouseEnter(screen.getByText("Verify every worker result").closest("g")!);
      // The box itself, the products arriving and the hand-over to the combine step.
      expect(container.querySelectorAll(".dg-on")).toHaveLength(3);
      fireEvent.mouseLeave(screen.getByText("Verify every worker result").closest("g")!);
      expect(container.querySelectorAll(".dg-on")).toHaveLength(0);
    });

    it("stays accessible while a step is selected", async () => {
      const { container } = render(<RuntimeFlowDiagram caption="c" />);
      fireEvent.click(screen.getByRole("button", { name: "Next" }));
      expect((await axe(container)).violations).toHaveLength(0);
    });
  });
});

describe("diagrams that follow the chosen question", () => {
  afterEach(() => questionSetting.set("text-shares"));

  it("quotes the real pieces, results and score of the question, and its full answer", () => {
    const data = architectureData("text-tamper");
    const journey = journeyOf("text-tamper");
    expect(data.question).toBe("What if a worker returns a wrong answer?");
    expect(data.answer).toBe(journey.answer);
    expect(data.values).toHaveLength(7);
    expect(data.values[0]).toBe("“What if a worker returns a wrong answer?”");
    // The first word, "what", is a 1 in the word counts; its two pieces add back to 1.
    const [first, second] = data.values[2]!.split(" + ").map(Number);
    expect((first! + second!) % SHOW_PRIME).toBe(1);
    expect(first).not.toBe(1);
    // The four workers' results add up to the winning score.
    const results = data.values[4]!.split(", ").map(Number);
    const score = Math.max(...journey.scores);
    expect(results.reduce((sum, value) => sum + value, 0) % SHOW_PRIME).toBe(score);
    expect(data.values[5]).toBe("4 of 4 pass");
    expect(data.values[6]).toBe(`score ${score} → “Your device tests every…”`);
    expect(data.parts.map((part) => part.id)).toEqual(["prompt", "split", "directory", "w00", "w01", "w10", "w11", "verify", "combine"]);
    const worker = data.parts.find((part) => part.id === "w10")!;
    expect(worker.holds).toContain(`for “what”: ${second}`);
    expect(worker.cannot).toContain(`the other piece (${first})`);
    expect(architectureData("text-tamper")).toEqual(data);
    expect(architectureData("text-limits").values[2]).not.toBe(data.values[2]);
    expect(runtimeData("text-tamper").values).toHaveLength(7);
    expect(runtimeData("text-tamper").values[5]).toBe("“Your”");
    expect(runtimeData("text-tamper").answer).toBe(journey.answer);
  });

  it("follows the start of the question through the memory lifecycle, wiped or not", () => {
    const data = memoryData("text-shares");
    expect(data.values).toEqual(["empty buffer", "48 6f 77 20", "48 6f 77 20", "48 6f 77 20", "00 00 00 00", "00 00 00 00", "48 6f 77 20 read by the next owner"]);
    expect(data.parts[1]!.holds).toContain("“How ”, the start of your question");
    expect(data.parts[5]!.sees).toBe("The next owner reads 00 00 00 00.");
    expect(data.answer).toBe(journeyOf("text-shares").answer);
  });

  it("shows the chosen question's value at each step, its full answer, and lets the reader open a part", async () => {
    const { container } = render(<ArchitectureDiagram caption="Figure 5.1" />);
    expect(container.querySelector(".dg-answer")).toHaveTextContent(`The model's answer: ${journeyOf("text-shares").answer}`);
    fireEvent.change(screen.getByLabelText("Your question"), { target: { value: "text-memory" } });
    expect(container.querySelector(".dg-answer")).toHaveTextContent(journeyOf("text-memory").answer);
    const next = screen.getByRole("button", { name: "Next" });
    fireEvent.click(next);
    expect(container.querySelector(".dg-value")).toHaveTextContent("With your question: “What happens to my data in memory afterwards?”");
    fireEvent.click(next);
    fireEvent.click(next);
    expect(container.querySelector(".dg-value")).toHaveTextContent(/With your question: \d+ \+ \d+/);
    fireEvent.click(next);
    // The value rides the arrow of the current step.
    expect(container.querySelectorAll(".dg-packet")).toHaveLength(1);
    expect(container.querySelector(".dg-packet text")!.textContent).toMatch(/^\d+ and \d+/);
    fireEvent.click(screen.getByRole("button", { name: "Worker (1,0)" }));
    expect(screen.getByRole("button", { name: "Worker (1,0)" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText("Worker (1,0) holds").nextElementSibling).toHaveTextContent(/Piece 1 of your question \(for “what”: \d+\) and piece 0 of the model\./);
    expect(screen.getByText("It cannot see").nextElementSibling).toHaveTextContent("Your words");
    // Clicking a box in the picture opens it too.
    fireEvent.click(screen.getByText("Signed worker directory", { selector: "text" }));
    expect(screen.getByText("Controller holds").nextElementSibling).toHaveTextContent("The list of workers that are online.");
    expect(container.querySelectorAll(".dg-picked")).toHaveLength(1);
    for (let index = 0; index < 3; index += 1) fireEvent.click(next);
    expect(container.querySelector(".dg-value")).toHaveTextContent(/score \d+ → “The memory your request…”/);
    expect((await axe(container)).violations).toHaveLength(0);
  });

  it("carries the question through the memory and runtime diagrams, with the answer always shown", () => {
    const memory = render(<MemoryLifecycleDiagram caption="Figure 9.1" />);
    const next = screen.getByRole("button", { name: "Next" });
    for (let index = 0; index < 6; index += 1) fireEvent.click(next);
    expect(memory.container.querySelector(".dg-value")).toHaveTextContent("With your question: 00 00 00 00");
    fireEvent.click(next);
    expect(memory.container.querySelector(".dg-value")).toHaveTextContent("48 6f 77 20 read by the next owner");
    expect(memory.container.querySelector(".dg-answer")).toHaveTextContent(journeyOf("text-shares").answer);
    memory.unmount();
    const runtime = render(<RuntimeFlowDiagram caption="Figure 7.1" />);
    expect(screen.getByLabelText("Your question")).toHaveValue("text-shares");
    expect(runtime.container.querySelector(".dg-answer")).toHaveTextContent(journeyOf("text-shares").answer);
    fireEvent.click(screen.getByRole("button", { name: "Workers" }));
    expect(screen.getByText("Workers holds").nextElementSibling).toHaveTextContent("never both");
  });
});
