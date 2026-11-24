import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.mock("@/components/Scene3DView", () => ({ Scene3DView: () => <div data-testid="view" /> }));

import LaboratoryPage from "@/app/laboratory/page";
import { ConjugationLab } from "@/components/labs/ConjugationLab";
import { DiffusionLab } from "@/components/labs/DiffusionLab";
import { FreivaldsLab } from "@/components/labs/FreivaldsLab";
import { FwhtLab } from "@/components/labs/FwhtLab";
import { MemoryLab } from "@/components/labs/MemoryLab";
import { MoeLab } from "@/components/labs/MoeLab";
import { OnionLab } from "@/components/labs/OnionLab";
import { PipelineLab } from "@/components/labs/PipelineLab";
import { PlacementLab } from "@/components/labs/PlacementLab";
import { QuantizedFormatLab } from "@/components/labs/QuantizedFormatLab";
import { ShareGridLab } from "@/components/labs/ShareGridLab";

const card = (name: string) => document.querySelector(`.liveio-${name}`) as HTMLElement;
const attack = () => screen.getByRole("button", { name: /^Try to break it:/ });
const outcome = () => document.querySelector(".liveio-attack [role=status]")!;

describe("every lab as a live story", () => {
  it("gives all eleven labs the in, seen, out strip with its three reasons", () => {
    const { container } = render(<LaboratoryPage />);
    expect(container.querySelectorAll("section.lab .liveio")).toHaveLength(11);
    expect(screen.getAllByRole("heading", { level: 3, name: "What goes in" })).toHaveLength(11);
    expect(screen.getAllByRole("heading", { level: 3, name: "What comes out" })).toHaveLength(11);
    expect(screen.getAllByText("Why this protects you")).toHaveLength(11);
    expect(screen.getAllByText("Where it stops")).toHaveLength(11);
    expect(container.querySelectorAll("section.lab .lab-plain")).toHaveLength(11);
    expect(container.querySelectorAll("details.lab-exact[open]")).toHaveLength(11);
  });

  it("share grid: your numbers decide the result, a worker sees neither, tampering is caught", () => {
    render(<ShareGridLab />);
    expect(card("out")).toHaveTextContent("Result: 19, 43.");
    fireEvent.change(screen.getByLabelText("Your two numbers"), { target: { value: "2, 3" } });
    expect(card("out")).toHaveTextContent("Result: 8, 18. The same as multiplying your numbers directly (8, 18).");
    expect(card("seen")).not.toHaveTextContent(/\b2, 3\b/);
    expect(card("seen")).toHaveTextContent(/\d{6,}/);
    fireEvent.click(attack());
    expect(outcome()).toHaveTextContent(/^Caught/);
    expect(card("out")).toHaveClass("warn");
    expect(card("out")).toHaveTextContent("Nothing.");
    fireEvent.click(attack());
    expect(card("out")).toHaveTextContent("Result: 8, 18.");
    fireEvent.change(screen.getByLabelText("Your two numbers"), { target: { value: "x" } });
    expect(card("in")).toHaveTextContent("Enter two whole numbers");
  });

  it("freivalds: the two sides agree for a right answer and differ after a one-off change", () => {
    render(<FreivaldsLab />);
    const sides = () => [...card("seen").querySelectorAll("code")].slice(1).map((node) => node.textContent);
    expect(sides()[0]).toBe(sides()[1]);
    expect(card("out")).toHaveTextContent("Accepted");
    fireEvent.click(attack());
    expect(sides()[0]).not.toBe(sides()[1]);
    expect(card("out")).toHaveTextContent("Rejected");
    expect(outcome()).toHaveTextContent(/^Caught/);
    expect(card("in")).toHaveTextContent("20, 43");
  });

  it("placement: no provider holds two pieces of one operation, at any shuffle", () => {
    render(<PlacementLab />);
    expect(card("out")).toHaveTextContent("0 of 16 providers hold two pieces of the same operation.");
    const before = card("seen").textContent;
    fireEvent.change(screen.getByLabelText("Shuffle number"), { target: { value: "3" } });
    expect(card("out")).toHaveTextContent("0 of 16 providers");
    expect(card("seen").textContent).not.toBe(before);
  });

  it("memory: your bytes read as zeros after the wipe, and leak when it is skipped", () => {
    render(<MemoryLab />);
    fireEvent.change(screen.getByLabelText("Your secret bytes"), { target: { value: "01 02 03 0a" } });
    expect(card("seen")).toHaveTextContent("00 00 00 00");
    expect(card("out")).toHaveTextContent("Zeros.");
    fireEvent.click(attack());
    expect(card("seen")).toHaveTextContent("01 02 03 0a");
    expect(card("out")).toHaveClass("warn");
    expect(outcome()).toHaveTextContent(/^It worked/);
    fireEvent.change(screen.getByLabelText("Your secret bytes"), { target: { value: "zz" } });
    expect(card("in")).toHaveTextContent("Enter four bytes");
  });

  it("quantized: one piece looks unrelated and both pieces give the numbers back exactly", () => {
    render(<QuantizedFormatLab />);
    fireEvent.change(screen.getByLabelText("Your weight codes"), { target: { value: "3, 1, 2, 0" } });
    const integers = card("in").querySelector("code")!.textContent!;
    expect(card("out")).toHaveTextContent(`Adding both pieces gives ${integers}: exactly the whole numbers that went in.`);
    expect(card("seen").querySelector("code")!.textContent).not.toBe(integers);
    fireEvent.change(screen.getByLabelText("Your weight codes"), { target: { value: "nope" } });
    expect(card("in")).toHaveTextContent("Enter at least two whole numbers");
  });

  it("conjugation: the worker sees rotated numbers and the answer comes back unchanged", () => {
    render(<ConjugationLab />);
    const codes = (name: string) => [...card(name).querySelectorAll("code")].map((node) => node.textContent);
    expect(codes("seen")[0]).not.toBe(codes("in")[0]);
    expect(codes("out")[0]).toBe(codes("out")[1]);
    const seen = codes("seen")[0];
    fireEvent.change(screen.getByLabelText("Secret rotation"), { target: { value: "1.2" } });
    expect(codes("seen")[0]).not.toBe(seen);
    expect(codes("out")[0]).toBe(codes("out")[1]);
    expect(card("seen")).toHaveTextContent("is unchanged");
  });

  it("fwht: changing the input changes the mixed output", () => {
    render(<FwhtLab />);
    const before = card("out").textContent;
    fireEvent.change(screen.getByLabelText("Your numbers"), { target: { value: "8,7,6,5,4,3,2,1" } });
    expect(card("out").textContent).not.toBe(before);
    fireEvent.change(screen.getByLabelText("Your numbers"), { target: { value: "1,2" } });
    expect(card("in")).toHaveTextContent("That is not a list of 8 numbers.");
  });

  it("pipeline: every request is answered, with or without a failed worker", () => {
    render(<PipelineLab />);
    expect(card("out")).toHaveTextContent("2 of 2 requests answered.");
    expect(outcome()).toHaveTextContent("No worker fails in this run.");
    expect(card("out")).toHaveTextContent("0 recoveries after a failure");
    expect(attack()).toHaveAttribute("aria-pressed", "false");
    fireEvent.click(attack());
    expect(outcome()).toHaveTextContent(/^Worker 1 failed\. [1-9]\d* recoveries later, 2 of 2/);
    fireEvent.change(screen.getByLabelText("How many requests to send"), { target: { value: "4" } });
    expect(card("out")).toHaveTextContent("4 of 4 requests answered.");
  });

  it("onion: your message comes out at the far end, a relay sees only sealed bytes, tampering is refused", async () => {
    render(<OnionLab />);
    await waitFor(() => expect(card("out")).toHaveTextContent("“private tensor” comes out at the far end"));
    fireEvent.change(screen.getByLabelText("Your message"), { target: { value: "hello there" } });
    await waitFor(() => expect(card("out")).toHaveTextContent("“hello there” comes out at the far end"));
    expect(card("seen")).toHaveTextContent("Next stop: worker-1");
    expect(card("seen")).not.toHaveTextContent("hello there");
    fireEvent.click(attack());
    await waitFor(() => expect(card("out")).toHaveTextContent("Refused."));
    expect(outcome()).toHaveTextContent(/^Caught/);
    fireEvent.click(attack());
    await waitFor(() => expect(card("out")).toHaveTextContent("comes out at the far end"));
  });

  it("moe: padding makes every expert's load equal, and removing it shows the routing", () => {
    render(<MoeLab />);
    expect(card("seen")).toHaveTextContent("the same for every expert");
    fireEvent.click(attack());
    expect(card("seen")).toHaveTextContent("uneven");
    expect(outcome()).toHaveTextContent(/^Without padding the loads are/);
  });

  it("diffusion: the worker's grid differs from yours and the result is recovered", () => {
    render(<DiffusionLab />);
    const code = (name: string) => card(name).querySelector("code")!.textContent;
    expect(code("seen")).not.toBe(code("in"));
    expect(card("out")).toHaveTextContent("the same as cleaning up directly");
    expect(within(card("in")).getByLabelText("Clean-up steps")).toBeInTheDocument();
  });
});
