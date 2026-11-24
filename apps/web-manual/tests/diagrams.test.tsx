import { render, screen } from "@testing-library/react";
import { axe } from "jest-axe";
import { describe, expect, it } from "vitest";
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
});
