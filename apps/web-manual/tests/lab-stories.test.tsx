import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/components/Scene3DView", () => ({ Scene3DView: ({ overlay }: { overlay?: React.ReactNode }) => <div data-testid="view">{overlay}</div> }));

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
import { journeyOf } from "@/lib/show/journey";
import { questionSetting } from "@/lib/show/question";

const card = (name: string) => document.querySelector(`.liveio-${name}`) as HTMLElement;
const answer = () => document.querySelector(".liveio-answer")!;
const attack = () => screen.getByRole("button", { name: /^Try to break it:/ });
const outcome = () => document.querySelector(".liveio-attack [role=status]")!;
const choose = (id: string) => fireEvent.change(screen.getAllByLabelText("Your question")[0]!, { target: { value: id } });
const shares = journeyOf("text-shares");
const tamper = journeyOf("text-tamper");

afterEach(() => questionSetting.set("text-shares"));

describe("every lab as a trip from the question to the answer", () => {
  it("gives all eleven labs an everyday title, the trip bar, the shared question and the model's full answer", () => {
    const { container } = render(<LaboratoryPage />);
    expect(container.querySelectorAll("section.lab .stage")).toHaveLength(11);
    // Every lab marks its own stop on the ribbon, between the question and the answer.
    expect(container.querySelectorAll("section.lab .ribbon")).toHaveLength(11);
    expect([...container.querySelectorAll("section.lab .ribbon")].every((ribbon) => ribbon.querySelectorAll("[aria-current=step]").length === 3)).toBe(true);
    expect(screen.getAllByLabelText("Your question")).toHaveLength(10);
    expect(screen.getAllByLabelText("Your picture request")).toHaveLength(1);
    const answers = [...container.querySelectorAll(".liveio-answer")];
    expect(answers).toHaveLength(11);
    expect(answers.filter((node) => node.textContent!.includes(shares.answer))).toHaveLength(10);
    for (const heading of [...container.querySelectorAll("section.lab h2")]) expect(heading.textContent).toMatch(/your|model|numbers|picture|piece|answer/i);
    expect(screen.getByRole("heading", { level: 2, name: "Splitting your question so no worker can read it" })).toBeInTheDocument();
    // One choice moves every lab.
    choose("text-tamper");
    expect(screen.getAllByLabelText("Your question").every((select) => (select as HTMLSelectElement).value === "text-tamper")).toBe(true);
    expect([...container.querySelectorAll(".liveio-answer")].filter((node) => node.textContent!.includes(tamper.answer))).toHaveLength(10);
  });

  it("share grid: runs the real first step on the question, shows a worker only random pieces, withholds the answer on tampering", () => {
    render(<ShareGridLab />);
    expect(answer()).toHaveTextContent(shares.answer);
    expect(card("in")).toHaveTextContent(`Its ${shares.words.length} words become a list of ${shares.counts.length} numbers`);
    expect(card("out")).toHaveTextContent(`one score for each prepared answer: ${shares.scores.join(", ")}.`);
    expect(card("seen")).toHaveTextContent(/\d{6,}/);
    expect(card("seen")).not.toHaveTextContent(shares.words[3]!);
    choose("text-tamper");
    expect(answer()).toHaveTextContent(tamper.answer);
    expect(card("out")).toHaveTextContent(tamper.scores.join(", "));
    fireEvent.click(attack());
    expect(outcome()).toHaveTextContent(/^Caught/);
    expect(card("out")).toHaveClass("warn");
    expect(answer()).toHaveTextContent("No answer is released.");
    fireEvent.click(attack());
    expect(answer()).toHaveTextContent(tamper.answer);
  });

  it("freivalds: the two sides agree for the workers' real result and differ after a one-off change", () => {
    render(<FreivaldsLab />);
    const sides = () => [...card("seen").querySelectorAll("code")].slice(1).map((node) => node.textContent);
    expect(card("in")).toHaveTextContent(shares.scores.join(", "));
    expect(sides()[0]).toBe(sides()[1]);
    expect(answer()).toHaveTextContent(shares.answer);
    fireEvent.click(attack());
    expect(sides()[0]).not.toBe(sides()[1]);
    expect(card("out")).toHaveTextContent("Rejected");
    expect(answer()).toHaveTextContent("No answer is released.");
    expect(outcome()).toHaveTextContent(/^Caught/);
  });

  it("placement: every question is dealt out so that no provider holds a matching pair", () => {
    render(<PlacementLab />);
    expect(card("in")).toHaveTextContent(`shuffle number ${shares.hash % 21}`);
    expect(card("out")).toHaveTextContent("0 of 16 providers hold two pieces of the same operation.");
    const before = card("seen").textContent;
    choose("text-limits");
    expect(card("in")).toHaveTextContent(`shuffle number ${journeyOf("text-limits").hash % 21}`);
    expect(card("out")).toHaveTextContent("0 of 16 providers");
    expect(card("seen").textContent).not.toBe(before);
    expect(answer()).toHaveTextContent(journeyOf("text-limits").answer);
  });

  it("memory: the start of the question reads as zeros after the wipe, and leaks when it is skipped", () => {
    render(<MemoryLab />);
    expect(card("in")).toHaveTextContent("“How ”, are stored as the bytes 48 6f 77 20");
    expect(card("seen")).toHaveTextContent("00 00 00 00");
    expect(card("out")).toHaveTextContent("Afterwards the memory reads zeros.");
    fireEvent.click(attack());
    expect(card("seen")).toHaveTextContent("48 6f 77 20");
    expect(card("out")).toHaveClass("warn");
    // The answer still arrives; the leak is what went wrong.
    expect(answer()).toHaveTextContent(shares.answer);
    expect(card("out")).toHaveTextContent("part of your question leaked");
    expect(outcome()).toHaveTextContent(/^It worked/);
    choose("text-tamper");
    expect(card("seen")).toHaveTextContent("57 68 61 74");
  });

  it("quantized: the model's codes come back exactly from two pieces that look unrelated", () => {
    render(<QuantizedFormatLab />);
    expect(card("in")).toHaveTextContent(`Four of the codes of the model that answers it: ${shares.codes.join(", ")}.`);
    const integers = card("in").querySelectorAll("code")[1]!.textContent!;
    expect(card("out")).toHaveTextContent(`Adding both pieces gives ${integers}: exactly the whole numbers that went in.`);
    expect(card("seen").querySelector("code")!.textContent).not.toBe(integers);
    choose("text-limits");
    expect(card("in")).toHaveTextContent(journeyOf("text-limits").codes.join(", "));
    expect(card("out")).toHaveTextContent("exactly the whole numbers that went in.");
  });

  it("conjugation: the question's letters are rotated for the worker and come back unchanged", () => {
    render(<ConjugationLab />);
    const codes = (name: string) => [...card(name).querySelectorAll("code")].map((node) => node.textContent);
    expect(codes("in")[0]).toBe("8.00, 15.00");
    expect(codes("seen")[0]).not.toBe(codes("in")[0]);
    expect(codes("out")[0]).toBe(codes("out")[1]);
    choose("text-tamper");
    expect(codes("in")[0]).toBe("23.00, 8.00");
    expect(codes("out")[0]).toBe(codes("out")[1]);
    expect(card("seen")).toHaveTextContent("is unchanged");
  });

  it("fwht: mixes the question's first letters, and a different question gives a different mix", () => {
    render(<FwhtLab />);
    expect(card("in").querySelector("code")).toHaveTextContent(shares.letters.slice(0, 8).join(", "));
    const before = card("out").textContent;
    choose("text-tamper");
    expect(card("in").querySelector("code")).toHaveTextContent(tamper.letters.slice(0, 8).join(", "));
    expect(card("out").textContent).not.toBe(before);
  });

  it("pipeline: the question is answered with or without a failed worker", () => {
    render(<PipelineLab />);
    expect(card("in")).toHaveTextContent("It travels as request 1 of 2, through 4 workers in a row.");
    expect(card("out")).toHaveTextContent("2 of 2 requests answered.");
    expect(outcome()).toHaveTextContent("No worker fails in this run.");
    fireEvent.click(attack());
    expect(outcome()).toHaveTextContent(/^Worker 1 failed\. [1-9]\d* recoveries later, 2 of 2/);
    expect(answer()).toHaveTextContent(shares.answer);
  });

  it("onion: the question itself is wrapped, a relay sees only sealed bytes, tampering stops it", async () => {
    render(<OnionLab />);
    await waitFor(() => expect(card("out")).toHaveTextContent(`“${shares.question}” reaches the far end unchanged`));
    expect(card("seen")).toHaveTextContent("Next stop: worker-1");
    expect(card("seen")).not.toHaveTextContent(shares.question);
    choose("text-tamper");
    await waitFor(() => expect(card("out")).toHaveTextContent(`“${tamper.question}” reaches the far end unchanged`));
    fireEvent.click(attack());
    await waitFor(() => expect(card("out")).toHaveTextContent("Refused."));
    expect(answer()).toHaveTextContent("No answer is released.");
    expect(outcome()).toHaveTextContent(/^Caught/);
    fireEvent.click(attack());
    await waitFor(() => expect(answer()).toHaveTextContent(tamper.answer));
  });

  it("moe: the question's words go to experts with equal loads, and removing the padding shows the routing", () => {
    render(<MoeLab />);
    expect(card("in")).toHaveTextContent(`“${shares.words.slice(0, 6).join("”, “")}”`);
    expect(card("seen")).toHaveTextContent("the same for every expert");
    fireEvent.click(attack());
    expect(card("seen")).toHaveTextContent("uneven");
    expect(card("out")).toHaveClass("warn");
    expect(answer()).toHaveTextContent(shares.answer);
    expect(outcome()).toHaveTextContent(/^Without padding the loads are/);
  });

  it("diffusion: a picture request goes in, the worker sees a different grid, the finished picture comes out", () => {
    render(<DiffusionLab />);
    expect(screen.getByLabelText("Your picture request")).toHaveValue("picture-sunrise");
    const code = (name: string) => card(name).querySelector("code")!.textContent;
    expect(code("seen")).not.toBe(code("in"));
    expect(screen.getByRole("img", { name: "The finished picture: A sunrise over the water" })).toBeInTheDocument();
    expect(card("out")).toHaveTextContent("the same as cleaning it up directly");
    expect(card("out")).toHaveTextContent("toy decoder");
  });
});
