import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
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
import { fwht } from "@/lib/simulation/algebra";
import { freivaldsCheck, placeOperations } from "@/lib/simulation/fortress";
import { getModel } from "@/lib/simulation/show";

const card = (name: string) => document.querySelector(`.liveio-${name}`) as HTMLElement;
const answer = () => document.querySelector(".liveio-answer")!;
const written = () => document.querySelector(".liveio-written")?.textContent ?? null;
const attack = () => screen.getByRole("button", { name: /^Try to break it:/ });
const outcome = () => document.querySelector(".liveio-attack [role=status]")!;
const choose = (id: string) => fireEvent.change(screen.getAllByLabelText("Your question")[0]!, { target: { value: id } });
const titles = () => [...document.querySelectorAll(".track-title")].map((node) => node.textContent);
const cards = () => [...document.querySelectorAll<HTMLElement>(".track-step")];
const now = () => document.querySelector<HTMLElement>(".track-step[aria-current=step]")!;
const said = () => document.querySelector(".track-now [role=status]")!;
const go = (title: string) => fireEvent.click(screen.getByRole("button", { name: new RegExp(`^Step \\d+: ${title.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}$`) }));
const toEnd = () => fireEvent.click([...document.querySelectorAll<HTMLElement>(".track-go")].at(-1)!);
const toStart = () => fireEvent.click(document.querySelector<HTMLElement>(".track-go")!);
const plain = (text: string) => text.replace(/\s+/g, " ");
const shares = journeyOf("text-shares");
const tamper = journeyOf("text-tamper");

afterEach(() => questionSetting.set("text-shares"));

/** What every text lab must do: show its steps, write nothing before the last one, then write the real answer. */
function tellsItsSteps(expected: string[]) {
  expect(titles()).toEqual(expected);
  expect(now()).toHaveTextContent(expected[0]!);
  expect(cards().slice(1).every((entry) => entry.querySelector(".track-empty"))).toBe(true);
  expect(answer()).toHaveClass("waiting");
  expect(answer()).toHaveTextContent("Nothing yet.");
  expect(card("out")).toHaveTextContent(`when all ${expected.length} steps are done. Now at step 1: ${expected[0]}.`);
  expect(written()).toBeNull();
  toEnd();
  expect(cards().every((entry) => !entry.querySelector(".track-empty"))).toBe(true);
  expect(written()).toBe(shares.answer);
  expect(screen.getByTestId("generation")).toHaveTextContent(`Token ${shares.tokens.length} of ${shares.tokens.length}.`);
  // The answer exists only as the result of the steps: back at the start there is none.
  toStart();
  expect(written()).toBeNull();
  choose("text-tamper");
  toEnd();
  expect(written()).toBe(tamper.answer);
  choose("text-shares");
  toStart();
}

describe("every lab as steps from what goes in to what comes out", () => {
  it("gives all eleven labs an everyday title, a row of steps, the shared question, and no answer before the steps are done", async () => {
    const { container } = render(<LaboratoryPage />);
    expect(container.querySelectorAll("section.lab .stage")).toHaveLength(11);
    // Every lab shows its own computation as steps along the bottom of its stage (the route lab once its keys exist); none shows the old dotted ribbon.
    await waitFor(() => expect(container.querySelectorAll("section.lab .stage .track")).toHaveLength(11));
    expect(container.querySelectorAll("section.lab .ribbon")).toHaveLength(0);
    expect([...container.querySelectorAll("section.lab .track")].every((track) => track.querySelectorAll("[aria-current=step]").length === 1)).toBe(true);
    expect([...container.querySelectorAll("section.lab .track")].every((track) => track.querySelectorAll(".track-step").length >= 4)).toBe(true);
    expect(screen.getAllByLabelText("Your question")).toHaveLength(10);
    expect(screen.getAllByLabelText("Your picture request")).toHaveLength(1);
    expect(container.querySelectorAll(".liveio-answer.waiting")).toHaveLength(11);
    expect(container.querySelector(".liveio-written")).toBeNull();
    for (const heading of [...container.querySelectorAll("section.lab h2")]) expect(heading.textContent).toMatch(/your|model|numbers|picture|piece|answer/i);
    expect(screen.getByRole("heading", { level: 2, name: "Splitting your question so no worker can read it" })).toBeInTheDocument();
    // One choice moves every lab.
    choose("text-tamper");
    expect(screen.getAllByLabelText("Your question").every((select) => (select as HTMLSelectElement).value === "text-tamper")).toBe(true);
  });

  it("share grid: runs the real first step on the question, shows a worker only random pieces, withholds the answer on tampering", () => {
    render(<ShareGridLab />);
    tellsItsSteps(["Words to numbers", "A random piece", "The other piece", "Four workers", "Added up"]);
    expect(card("in")).toHaveTextContent(`Its ${shares.words.length} words become a list of ${shares.counts.length} numbers`);
    go("Four workers");
    expect(card("seen")).toHaveTextContent(/\d{3} \d{3} \d{3}/);
    expect(card("seen")).not.toHaveTextContent(shares.words[3]!);
    toEnd();
    expect(card("out")).toHaveTextContent(`one score for each prepared answer: ${shares.scores.join(", ")}.`);
    fireEvent.click(attack());
    expect(outcome()).toHaveTextContent(/^Caught/);
    expect(card("out")).toHaveClass("warn");
    expect(answer()).toHaveTextContent("No answer is released.");
    fireEvent.click(attack());
    expect(written()).toBe(shares.answer);
  });

  it("freivalds: both sides are worked out from the workers' real result, agree, and differ after a one-off change", () => {
    render(<FreivaldsLab />);
    tellsItsSteps(["The workers' result", "A challenge", "Side from the result", "Side from your question", "They match"]);
    expect(card("in")).toHaveTextContent(shares.scores.join(", "));
    const challenge = [11, 13, 17, 19, 23, 29];
    const right = shares.scores.reduce((sum, score, index) => sum + score * challenge[index]!, 0);
    expect(freivaldsCheck(shares.weightRows, shares.counts, shares.scores, challenge)).toBe(true);
    go("Side from the result");
    expect(plain(said().textContent!)).toContain(`${shares.scores[0]}×11 + ${shares.scores[1]}×13`);
    expect(plain(now().textContent!).replace(/ /g, "")).toContain(String(right));
    // The side from the question has not been computed yet.
    expect([...card("seen").querySelectorAll("code")].map((node) => node.textContent)).toEqual([challenge.join(", "), "…", String(right)]);
    toEnd();
    const sides = () => [...card("seen").querySelectorAll("code")].slice(1).map((node) => node.textContent);
    expect(sides()).toEqual([String(right), String(right)]);
    fireEvent.click(attack());
    expect(sides()[0]).not.toBe(sides()[1]);
    expect(titles().at(-1)).toBe("They differ");
    expect(outcome()).toHaveTextContent(/^Caught/);
    expect(answer()).toHaveTextContent("No answer is released.");
    expect(written()).toBeNull();
  });

  it("quantized: the model's codes become whole numbers and come back exactly from two pieces that look unrelated", () => {
    render(<QuantizedFormatLab />);
    tellsItsSteps(["Stored codes", "Decoded", "Whole numbers", "A random piece", "The other piece", "Added back"]);
    expect(card("in")).toHaveTextContent(shares.codes.join(", "));
    expect(card("seen")).toHaveTextContent("Nothing yet");
    go("Whole numbers");
    const whole = plain(now().querySelector(".data-value .sr-only")!.textContent!);
    go("A random piece");
    expect(card("seen")).toHaveTextContent(/\d{6,}/);
    toEnd();
    expect(plain(now().querySelector(".data-value .sr-only")!.textContent!)).toBe(whole);
    expect(now()).toHaveTextContent("= the whole numbers of step 3");
    expect(card("out")).toHaveTextContent("exactly the whole numbers that went in.");
  });

  it("placement: every operation is handed to four providers, and no provider ends up with a matching pair", () => {
    render(<PlacementLab />);
    const roles = placeOperations(shares.hash % 21);
    const operations = [...new Set(roles.map((role) => role.operation))];
    tellsItsSteps(["The work", ...operations, "Nobody holds a pair"]);
    expect(card("seen")).toHaveTextContent("Nothing yet.");
    go(operations[0]!);
    for (const role of roles.filter((entry) => entry.operation === operations[0])) expect(now()).toHaveTextContent(`${role.provider} ${role.country.toUpperCase()}`);
    expect(card("seen").textContent).toMatch(new RegExp(`^What .* holds(${operations[0]} \\(\\d,\\d\\) and nothing else|Nothing yet)`));
    toEnd();
    expect(said()).toHaveTextContent(/Checked: 0 of \d+ providers hold two pieces of the same operation/);
    expect(card("out")).toHaveTextContent(/^What comes out.*0 of \d+ providers hold two pieces/);
    expect(card("out")).not.toHaveClass("warn");
  });

  it("conjugation: the question's letters are rotated for the worker and come back the same as computing directly", () => {
    render(<ConjugationLab />);
    tellsItsSteps(["Your numbers", "Turned in secret", "Worker computes", "Turned back", "Same as direct"]);
    expect(card("in")).toHaveTextContent(`${shares.letters[0]!.toFixed(2)}, ${shares.letters[1]!.toFixed(2)}`);
    expect(card("seen")).toHaveTextContent("Nothing has left your device yet.");
    go("Turned in secret");
    expect(card("seen")).toHaveTextContent("Their overall size");
    expect(card("seen")).not.toHaveTextContent(`${shares.letters[0]!.toFixed(2)}, ${shares.letters[1]!.toFixed(2)}`);
    toEnd();
    const rows = [...now().querySelectorAll(".data-line .data-value")].map((node) => node.textContent);
    expect(rows[0]).toBe(rows[1]);
    expect(card("out")).toHaveTextContent("the same as computing directly");
  });

  it("fwht: mixes the question's first letters round by round, and a different question gives a different mix", () => {
    render(<FwhtLab />);
    tellsItsSteps(["Your numbers", "Round 1", "Round 2", "Round 3"]);
    const letters = Array.from({ length: 8 }, (_, index) => shares.letters[index % shares.letters.length]!);
    const result = fwht(letters);
    go("Round 1");
    expect(said()).toHaveTextContent(`${letters[0]} and ${letters[1]} become ${letters[0]! + letters[1]!} and ${letters[0]! - letters[1]!}`);
    expect(plain(now().querySelector(".data-value .sr-only")!.textContent!)).toContain(result.stages[1]!.slice(0, 3).join(" "));
    toEnd();
    const first = card("out").querySelector(".liveio-result code")!.textContent;
    expect(first).toContain(result.output[0]!.toFixed(2));
    choose("text-tamper");
    expect(card("out").querySelector(".liveio-result code")!.textContent).not.toBe(first);
  });

  it("pipeline: the question is answered with or without a failed worker, and the setback is shown", () => {
    render(<PipelineLab />);
    tellsItsSteps(["Sealed and sent", "Through the workers", "Back on your device", "The other requests"]);
    go("Back on your device");
    expect(written()).toBe(shares.answer);
    const setbacks = () => Number(/setbacks\s*(\d+)/.exec(plain(now().textContent!))![1]);
    const before = setbacks();
    fireEvent.click(attack());
    expect(outcome()).toHaveTextContent(/Worker 1 failed\. \d+ recoveries later/);
    go("Back on your device");
    expect(cards()[1]).toHaveTextContent(/worker 1\s*(lost, sent again, )?failed, recovered, done/);
    expect(setbacks()).toBe(before + 1);
    // A failed worker delays the answer; it is still written.
    expect(written()).toBe(shares.answer);
  });

  it("onion: the question's bytes are wrapped three times, each relay removes one layer, and an altered packet goes no further", async () => {
    render(<OnionLab />);
    await waitFor(() => expect(titles()).toEqual(["Readable", "Sealed three times", "Relay 1", "Relay 2", "Relay 3", "Readable again"]));
    const bytes = new TextEncoder().encode(shares.question);
    const head = Array.from(bytes.slice(0, 6), (value) => value.toString(16).padStart(2, "0")).join(" ");
    expect(now()).toHaveTextContent(`${head} … ${bytes.length} bytes`);
    expect(written()).toBeNull();
    expect(card("seen")).toHaveTextContent("No relay has received anything yet.");
    go("Sealed three times");
    // Sealing adds a 16-byte authenticity tag to what was readable.
    expect(said()).toHaveTextContent(`The packet is now ${bytes.length + 16} bytes and none of them is readable.`);
    expect(now()).not.toHaveTextContent(head);
    go("Relay 1");
    expect(card("seen")).toHaveTextContent("Next stop: worker-1");
    expect(card("seen")).not.toHaveTextContent(shares.question);
    toEnd();
    await waitFor(() => expect(card("out")).toHaveTextContent(`“${shares.question}” reaches the far end unchanged`));
    expect(now()).toHaveTextContent(`${head} … ${bytes.length} bytes`);
    expect(written()).toBe(shares.answer);
    fireEvent.click(attack());
    await waitFor(() => expect(answer()).toHaveTextContent("No answer is released."));
    expect(titles().at(-1)).toBe("Nothing arrives");
    expect(cards()[3]).toHaveTextContent("refused");
    expect(cards()[4]).toHaveTextContent("nothing received");
    expect(outcome()).toHaveTextContent(/^Caught/);
    fireEvent.click(attack());
    await waitFor(() => expect(written()).toBe(shares.answer));
  });

  it("moe: the question's words go to experts, dummy work evens the loads, and removing it shows the routing", () => {
    render(<MoeLab />);
    tellsItsSteps(["Six words", "Router scores", "Real work", "Dummy work added", "Results combined"]);
    expect(card("in")).toHaveTextContent(`“${shares.words.slice(0, 6).join("”, “")}”`);
    go("Real work");
    expect(card("seen")).toHaveTextContent("uneven");
    go("Dummy work added");
    expect(card("seen")).toHaveTextContent("the same for every expert");
    expect(new Set([...now().querySelectorAll(".data-bar-value")].map((node) => node.textContent)).size).toBe(1);
    fireEvent.click(attack());
    expect(titles()[3]).toBe("No dummy work");
    expect(card("seen")).toHaveTextContent("uneven");
    toEnd();
    expect(card("out")).toHaveClass("warn");
    expect(written()).toBe(shares.answer);
  });

  it("diffusion: a picture request goes in, the worker sees a different grid, and the picture forms step by step", () => {
    render(<DiffusionLab />);
    expect(titles()).toEqual(["Random grid", "Turned and shuffled", "Worker cleans up", "Shuffle undone", "Picture decoded"]);
    expect(screen.getByLabelText("Your picture request")).toBeInTheDocument();
    expect(answer()).toHaveTextContent("The model's picture Nothing yet.");
    expect(document.querySelector(".liveio-picture")).toBeNull();
    expect(card("seen")).toHaveTextContent("Nothing has left your device yet.");
    go("Turned and shuffled");
    expect(card("seen")).toHaveTextContent("the grid turned and in a different order");
    toEnd();
    const units = getModel("picture-sunrise").units;
    expect(screen.getByRole("img", { name: "The finished picture: A sunrise over the water" })).toBeInTheDocument();
    expect(screen.getByTestId("generation")).toHaveTextContent(`Step ${units} of ${units}.`);
    expect(screen.getByTestId("generation")).toHaveTextContent(`${units * 4} of ${units * 4} results passed their check`);
    fireEvent.change(screen.getByLabelText("Your picture request"), { target: { value: "picture-city" } });
    expect(screen.getByRole("img", { name: /^The finished picture: A city at night/ })).toBeInTheDocument();
  });

  it("memory: the start of the question sits in memory, reads as zeros after the wipe, and leaks when the wipe is skipped", () => {
    render(<MemoryLab />);
    tellsItsSteps(["Set aside", "In use", "Handed back", "Wipe put first", "Wiped", "Next user"]);
    const hex = shares.bytes.map((value) => value.toString(16).padStart(2, "0")).join(" ");
    expect(now()).toHaveTextContent(`${hex} (4 bytes)`);
    expect(now()).toHaveTextContent(`“${shares.question.slice(0, 4)}”`);
    expect(card("seen")).toHaveTextContent("Nothing yet");
    toEnd();
    expect(now()).toHaveTextContent("00 00 00 00 (4 bytes)");
    expect(card("seen")).toHaveTextContent("00 00 00 00 — nothing of yours.");
    fireEvent.click(attack());
    expect(titles().slice(3)).toEqual(["Wipe only queued", "Not wiped yet", "Next user"]);
    expect(now()).toHaveTextContent(hex);
    expect(card("seen")).toHaveTextContent(`${hex} — the start of your question, still there.`);
    expect(card("out")).toHaveClass("warn");
    expect(within(card("out")).getByText(/part of your question leaked/)).toBeInTheDocument();
  });
});
