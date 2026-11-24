import { act, fireEvent, render, screen } from "@testing-library/react";
import { axe } from "jest-axe";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ArchitectureDiagram } from "@/components/diagrams/ArchitectureDiagram";
import { MemoryLifecycleDiagram } from "@/components/diagrams/MemoryLifecycleDiagram";
import { RuntimeFlowDiagram } from "@/components/diagrams/RuntimeFlowDiagram";
import { Figure } from "@/components/Figure";

describe("chapter diagrams", () => {
  it("draws the architecture with four role workers and a described picture", async () => {
    const { container } = render(<ArchitectureDiagram caption="Figure 5.1" />);
    const picture = screen.getByRole("img", { name: /Fortress system architecture/ });
    expect(picture).toHaveAccessibleDescription(/never sees the prompt/);
    expect(container.querySelectorAll("text.dg-strong")).toHaveLength(9);
    for (const role of ["(0,0)", "(0,1)", "(1,0)", "(1,1)"]) expect(screen.getByText(`Worker ${role}`)).toBeInTheDocument();
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
