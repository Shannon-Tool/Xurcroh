import { act, fireEvent, render, renderHook, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/components/Scene3DView", () => ({ Scene3DView: ({ overlay, scene }: { overlay?: React.ReactNode; scene: { nodes: { label?: string }[] } }) => <div data-testid="view" data-labels={scene.nodes.map((node) => node.label ?? "").join(" | ")}>{overlay}</div> }));

import { ProcessTrack } from "@/components/ProcessTrack";
import { ShareGridLab } from "@/components/labs/ShareGridLab";
import { TOKEN_MS, useGeneration } from "@/lib/show/generation";
import { journeyOf } from "@/lib/show/journey";
import { grouped, hexText, numbersText, viewText, type ProcessStep } from "@/lib/show/process";
import { questionSetting } from "@/lib/show/question";
import { executeShareGrid, mod, PRIME } from "@/lib/simulation/fortress";

const shares = journeyOf("text-shares");
const tamper = journeyOf("text-tamper");

afterEach(() => {
  questionSetting.set("text-shares");
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

const cards = () => [...document.querySelectorAll<HTMLElement>(".track-step")];
const now = () => document.querySelector<HTMLElement>(".track-step[aria-current=step]")!;
const said = () => document.querySelector(".track-now [role=status]")!;
const out = () => document.querySelector<HTMLElement>(".liveio-out")!;
const written = () => document.querySelector(".liveio-written")?.textContent ?? null;
const forward = (times = 1) => { for (let count = 0; count < times; count += 1) fireEvent.click(screen.getByRole("button", { name: "Step forward" })); };
// Numbers are written in groups of three with narrow spaces; compared here with ordinary ones.
const plain = (text: string) => text.replace(/\s+/g, " ");
const big = (value: bigint | number) => plain(grouped(value));
const labels = () => plain(screen.getByTestId("view").getAttribute("data-labels")!);

describe("showing data in a step", () => {
  it("writes long numbers in groups, lists with their length, and bytes as hex with their count", () => {
    expect(grouped(4294967291n)).toBe("4 294 967 291");
    expect(grouped(600)).toBe("600");
    expect(numbersText([1, 2, 3, 4, 5, 6], 3)).toBe("1  2  3  … 6 in all");
    expect(hexText(Uint8Array.from([0, 15, 255]), 2)).toBe("00 0f … 3 bytes");
    expect(hexText(Uint8Array.from([72, 105]))).toBe("48 69 (2 bytes)");
    expect(viewText({ kind: "bytes", bytes: Uint8Array.from([72, 105]), readable: "Hi" })).toBe("48 69 (2 bytes) “Hi”");
    expect(viewText({ kind: "bars", values: [600, 0], names: ["a", "b"] })).toBe("a: 600, b: 0");
  });

  it("shows finished steps with their data, the current one lit, and the ones to come empty", () => {
    const steps: ProcessStep[] = [
      { id: "a", title: "First", does: "Did the first thing.", data: [{ kind: "numbers", values: [1, 2, 3] }] },
      { id: "b", title: "Second", does: "Did the second thing.", data: [{ kind: "bytes", bytes: Uint8Array.from([171, 205]) }], tone: "cipher" },
      { id: "c", title: "Third", does: "Did the third thing.", data: [{ kind: "text", text: "the result" }] },
    ];
    const pressed: number[] = [];
    render(<ProcessTrack steps={steps} step={1} why="Because." onStep={(step) => pressed.push(step)} />);
    expect(cards().map((card) => card.className.match(/done|now|todo/)![0])).toEqual(["done", "now", "todo"]);
    expect(cards()[0]).toHaveTextContent("1 2 3");
    expect(now()).toHaveTextContent("ab cd (2 bytes)");
    expect(now()).toHaveClass("track-cipher");
    expect(cards()[2]).not.toHaveTextContent("the result");
    expect(said()).toHaveTextContent("Step 2 of 3 Did the second thing.");
    expect(screen.getByText("Because.", { exact: false })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Step 3: Third" }));
    expect(pressed).toEqual([2]);
  });
});

describe("writing the answer", () => {
  it("writes nothing until it is switched on, then every token from its own checked product", () => {
    const { result, rerender } = renderHook(({ active }) => useGeneration("text-shares", active), { initialProps: { active: false } });
    expect(result.current).toMatchObject({ tokens: [], checks: 0, passed: 0, complete: false, total: shares.tokens.length });
    rerender({ active: true });
    // Without motion the whole answer is computed at once; each token is still decoded from its product.
    expect(result.current.tokens.join("")).toBe(shares.tokens.join(""));
    expect(result.current).toMatchObject({ checks: shares.tokens.length * 4, passed: shares.tokens.length * 4, complete: true });
    rerender({ active: false });
    expect(result.current.tokens).toEqual([]);
  });

  it("writes one token at a time when motion is allowed", () => {
    vi.useFakeTimers();
    vi.stubGlobal("matchMedia", (query: string) => ({ matches: false, media: query, addEventListener: () => undefined, removeEventListener: () => undefined }));
    const { result } = renderHook(() => useGeneration("text-shares", true));
    expect(result.current.tokens).toEqual([]);
    act(() => { vi.advanceTimersByTime(TOKEN_MS * 3 + 5); });
    expect(result.current.tokens).toEqual(shares.tokens.slice(0, 3));
    expect(result.current).toMatchObject({ checks: 12, passed: 12, complete: false });
    act(() => { vi.advanceTimersByTime(TOKEN_MS * shares.tokens.length); });
    expect(result.current.tokens.join("")).toBe(shares.answer);
    expect(result.current.complete).toBe(true);
  });
});

describe("share-splitting lab, step by step", () => {
  const grid = executeShareGrid(shares.weightRows, shares.counts, 0);
  const [pieceA, pieceB] = grid.activationShares as [bigint[], bigint[]];
  const first = shares.counts.findIndex((count) => count > 0);

  it("starts with the question as numbers, nothing sent and nothing written", () => {
    render(<ShareGridLab />);
    expect(cards().map((card) => card.querySelector(".track-title")!.textContent)).toEqual(["Words to numbers", "A random piece", "The other piece", "Four workers", "Added up"]);
    expect(now()).toHaveTextContent("Words to numbers");
    // One cell lit for every word of the question that the model knows.
    expect(now().querySelectorAll(".data-cell.on")).toHaveLength(shares.counts.filter((count) => count > 0).length);
    expect(cards().slice(1).every((card) => card.querySelector(".track-empty"))).toBe(true);
    expect(out()).toHaveTextContent("Nothing yet.");
    expect(out()).toHaveTextContent("Now at step 1: Words to numbers.");
    expect(written()).toBeNull();
    expect(document.querySelector(".liveio-seen")).toHaveTextContent("Nothing has left your device yet.");
    expect(labels()).toContain(`Your device: ${shares.counts.slice(0, 8).join(" ")}`);
  });

  it("shows the two pieces the engine made, and that they add back to the numbers", () => {
    render(<ShareGridLab />);
    forward();
    expect(now()).toHaveTextContent(`piece A${big(pieceA[0]!)}`);
    forward();
    expect(now()).toHaveTextContent(`piece B${big(pieceB[0]!)}`);
    expect(mod(pieceA[first]! + pieceB[first]!)).toBe(BigInt(shares.counts[first]!));
    expect(said()).toHaveTextContent(`${shares.counts[first]} = ${big(pieceA[first]!)} + ${big(pieceB[first]!)}, counted round at ${big(PRIME)}`);
    expect(labels()).toContain(`pieces ${big(pieceA[first]!)} + ${big(pieceB[first]!)}`);
    expect(written()).toBeNull();
  });

  it("fills in one worker at a time with what that worker really returned", () => {
    render(<ShareGridLab />);
    forward(3);
    const lines = () => [...now().querySelectorAll(".data-line")].map((line) => line.className.replace("data-line ", ""));
    expect(lines()).toEqual(["now", "todo", "todo", "todo"]);
    expect(now()).toHaveTextContent(`${big(mod(grid.roles[0]!.product[0]!))} …`);
    expect(document.querySelector(".liveio-seen h3")).toHaveTextContent("What worker (0,0) sees");
    expect(document.querySelector(".liveio-seen")).toHaveTextContent(big(pieceA[0]!));
    // The worker's panel never shows a word of the question.
    for (const word of shares.words) expect(document.querySelector(".liveio-seen")!.textContent).not.toContain(` ${word} `);
    forward(2);
    expect(lines()).toEqual(["done", "done", "now", "todo"]);
    expect(labels()).toContain(`Worker (1,0): piece ${big(pieceB[first]!)}`);
    expect(out()).toHaveTextContent("Now at step 4: Four workers.");
    expect(written()).toBeNull();
  });

  it("writes the answer only after the four results are added up, from the scores that chose it", () => {
    render(<ShareGridLab />);
    forward(7);
    expect(now()).toHaveTextContent(`scores${shares.scores.join(" ")}`);
    const bars = [...out().querySelectorAll(".data-bar")];
    expect(bars.map((bar) => bar.querySelector(".data-bar-value")!.textContent)).toEqual(shares.scores.map(String));
    expect(bars.findIndex((bar) => bar.classList.contains("top"))).toBe(shares.scores.indexOf(Math.max(...shares.scores)));
    expect(written()).toBe(shares.answer);
    expect(screen.getByTestId("generation")).toHaveTextContent(`Token ${shares.tokens.length} of ${shares.tokens.length}.`);
    expect(screen.getByTestId("generation")).toHaveTextContent(`${shares.tokens.length * 4} of ${shares.tokens.length * 4} results passed their check`);
    // Going back a step takes the answer away again: it only exists as the result of the steps.
    fireEvent.click(screen.getByRole("button", { name: "Step back" }));
    expect(written()).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Step 5: Added up" }));
    expect(written()).toBe(shares.answer);
  });

  it("follows the chosen question, and releases nothing when a worker's result was changed", () => {
    render(<ShareGridLab />);
    fireEvent.change(screen.getByLabelText("Your question"), { target: { value: "text-tamper" } });
    forward(7);
    expect(written()).toBe(tamper.answer);
    expect(now()).toHaveTextContent(tamper.scores.join(" "));
    fireEvent.click(screen.getByRole("button", { name: /^Try to break it:/ }));
    expect(now()).toHaveTextContent("Refused");
    expect(now()).toHaveTextContent("nothing combined");
    expect(out()).toHaveClass("warn");
    expect(out()).toHaveTextContent("No answer is released.");
    expect(written()).toBeNull();
    expect(within(cards()[3]!).getByText("(0,0)").closest(".data-line")).toHaveClass("bad");
    fireEvent.click(screen.getByRole("button", { name: /^Try to break it:/ }));
    expect(written()).toBe(tamper.answer);
  });
});
