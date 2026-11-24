import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { axe } from "jest-axe";
import { ConjugationLab } from "@/components/labs/ConjugationLab";
import { FwhtLab } from "@/components/labs/FwhtLab";
import { MemoryLab } from "@/components/labs/MemoryLab";
import { MoeLab } from "@/components/labs/MoeLab";
import { OnionLab } from "@/components/labs/OnionLab";
import { PipelineLab } from "@/components/labs/PipelineLab";
import { DiffusionLab } from "@/components/labs/DiffusionLab";
import { Search } from "@/components/Search";

describe("interactive manual", () => {
  it("steps conjugation state and changes numerical parameters", async () => {
    render(<ConjugationLab />);
    expect(screen.getByText("Original computation")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Step forward" }));
    expect(screen.getByText("Secret basis")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Reset" }));
    await waitFor(() => expect(screen.getByText("Original computation")).toBeInTheDocument());
    expect(screen.getByText(/Input norm:/)).toBeInTheDocument();
  });

  it("edits FWHT input and steps through actual stages", () => {
    render(<FwhtLab />);
    fireEvent.click(screen.getByRole("button", { name: "Step forward" }));
    expect(screen.getByText("Stage 1")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Input vector"), { target: { value: "1,2,3" } });
    expect(screen.getByText(/Enter 8 finite numbers/)).toBeInTheDocument();
  });

  it("demonstrates synchronized and residual memory states", () => {
    render(<MemoryLab />);
    fireEvent.click(screen.getByRole("checkbox", { name: "Complete wipe before reuse" }));
    fireEvent.click(screen.getByRole("button", { name: "Step forward" }));
    fireEvent.click(screen.getByRole("button", { name: "Step forward" }));
    expect(screen.getByText(/Logical blocks return/)).toBeInTheDocument();
  });

  it("keeps pipeline playback valid when the workload shrinks", () => {
    render(<PipelineLab />);
    fireEvent.change(screen.getByLabelText(/Requests/), { target: { value: "8" } });
    for (let index = 0; index < 8; index += 1) {
      fireEvent.click(screen.getByRole("button", { name: "Step forward" }));
    }
    fireEvent.change(screen.getByLabelText(/Requests/), { target: { value: "1" } });
    expect(screen.getByText(/Event \d+\/\d+/)).toBeInTheDocument();
  });

  it("reports MoE capacity overflow without crashing", () => {
    render(<MoeLab />);
    fireEvent.change(screen.getByLabelText("Top-k"), { target: { value: "3" } });
    fireEvent.change(screen.getByLabelText(/Capacity/), { target: { value: "2" } });
    expect(screen.getByRole("alert")).toHaveTextContent("constant-load capacity exceeded");
  });

  it("can verify an onion layer again after removing tampering", async () => {
    render(<OnionLab />);
    await screen.findByText("Next: client", { exact: false });
    fireEvent.click(screen.getByRole("button", { name: "Step forward" }));
    fireEvent.click(screen.getByRole("button", { name: "Step forward" }));
    const tamper = screen.getByRole("checkbox", { name: "Tamper current ciphertext" });
    fireEvent.click(tamper);
    fireEvent.click(tamper);
    fireEvent.click(screen.getByRole("button", { name: "Decrypt / verify" }));
    await waitFor(() => expect(document.querySelector(".state-inspector")).toHaveTextContent(/"result": "private tensor"/));
  });

  it("updates the diffusion computation and preserves recovery", () => {
    render(<DiffusionLab />);
    fireEvent.change(screen.getByLabelText(/Euler steps/), { target: { value: "4" } });
    fireEvent.click(screen.getByRole("button", { name: "Step forward" }));
    expect(screen.getByText("Rotated and permuted latent")).toBeInTheDocument();
    const recovery = screen.getByText(/Recovery error:/).textContent?.split(": ")[1];
    expect(Number(recovery)).toBeLessThan(1e-12);
  });

  it("searches sections and glossary", () => {
    render(<Search />);
    fireEvent.change(screen.getByRole("searchbox", { name: "Search manual" }), { target: { value: "conjugation" } });
    expect(screen.getByText("Mathematics")).toBeInTheDocument();
  });

  it("has no detectable accessibility violations in the memory lab", async () => {
    const { container } = render(<MemoryLab />);
    const results = await axe(container);
    expect(results.violations).toHaveLength(0);
  });
});


describe("Fortress laboratories", () => {
  it("rejects a tampered share-grid role and preserves exact output", async () => {
    const { ShareGridLab } = await import("@/components/labs/ShareGridLab");
    render(<ShareGridLab />);
    expect(screen.getByText("19, 43")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("checkbox", { name: "Tamper role (0,0)" }));
    expect(screen.getByRole("alert")).toHaveTextContent("role 0:0");
    fireEvent.click(screen.getByRole("button", { name: "Step forward" }));
    expect(screen.getByText("Role (0,1) exact signed product")).toBeInTheDocument();
  });

  it("rejects a modified Freivalds product and validates challenge input", async () => {
    const { FreivaldsLab } = await import("@/components/labs/FreivaldsLab");
    render(<FreivaldsLab />);
    expect(screen.getByText("accepted")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("checkbox", { name: "Add one to the first product" }));
    expect(screen.getByText("rejected")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Challenge vector r"), { target: { value: "invalid" } });
    expect(screen.getByRole("alert")).toHaveTextContent("Enter two nonnegative safe integers");
  });

  it("reshuffles actual operation placement", async () => {
    const { PlacementLab } = await import("@/components/labs/PlacementLab");
    render(<PlacementLab />);
    expect(screen.getByText(/Current epoch:/)).toHaveTextContent("Current epoch: 0; assignments: 16");
    fireEvent.click(screen.getByRole("button", { name: "Reshuffle" }));
    expect(screen.getByText(/Current epoch:/)).toHaveTextContent("Current epoch: 1; assignments: 16");
  });
});
