import { render, screen } from "@testing-library/react";
import { axe } from "jest-axe";
import { describe, expect, it, vi } from "vitest";

vi.mock("@/components/Scene3DView", () => ({
  Scene3DView: ({ scene, overlay, variant }: { scene: { caption: string }; overlay: React.ReactNode; variant: string }) => <div data-testid="view" data-variant={variant}>{scene.caption}{overlay}</div>,
}));

import { FlowRibbon, type FlowForm } from "@/components/FlowRibbon";
import { ScrambleText } from "@/components/ScrambleText";
import { StageBox } from "@/components/StageBox";
import { hexRun, scrambleFrame } from "@/lib/show/scramble";

const forms: FlowForm[] = [
  { id: "readable", label: "Readable", value: "How is my prompt…", tone: "good" },
  { id: "pieces", label: "Two random pieces", value: "9f3a 01c2", tone: "accent" },
  { id: "sealed", label: "Sealed three times", value: "a1 b2 c3", tone: "cipher" },
];

describe("scramble", () => {
  it("starts as the old text, ends as the new text, and settles from the left in between", () => {
    expect(scrambleFrame("hello world", "9f3a 01c2 77", 0, 3)).toBe("hello world");
    expect(scrambleFrame("hello world", "9f3a 01c2 77", 1, 3)).toBe("9f3a 01c2 77");
    const middle = scrambleFrame("hello world!", "9f3a 01c2 77", 0.5, 3);
    expect(middle).toHaveLength(12);
    expect(middle.slice(0, 6)).toBe("9f3a 0");
    expect(middle.slice(-2)).toBe("d!");
    expect(middle.slice(6, 9)).toMatch(/^[0-9a-f ]{3}$/);
    expect(scrambleFrame("hello world!", "9f3a 01c2 77", 0.5, 4)).not.toBe(middle);
    expect(scrambleFrame("short", "a much longer value", 0.999, 1).length).toBe(19);
  });

  it("writes numbers and bytes as hexadecimal", () => {
    expect(hexRun([255, 1, 4096], 4, 2)).toBe("00ff 0001");
    expect(hexRun(Uint8Array.from([171, 5]), 2, 9)).toBe("ab 05");
  });

  it("gives assistive technology the real value and shows it at once without motion", () => {
    const { container, rerender } = render(<ScrambleText text="readable" />);
    expect(container.querySelector(".sr-only")).toHaveTextContent("readable");
    expect(container.querySelector("[aria-hidden=true]")).toHaveTextContent("readable");
    rerender(<ScrambleText text="9f3a 01c2" />);
    expect(container.querySelector(".sr-only")).toHaveTextContent("9f3a 01c2");
    expect(container.querySelector("[aria-hidden=true]")).toHaveTextContent("9f3a 01c2");
  });
});

describe("flow ribbon and stage box", () => {
  it("shows every form the data takes and marks the current ones", () => {
    render(<FlowRibbon forms={forms} current={["sealed"]} />);
    const items = screen.getAllByRole("listitem");
    expect(items.map((item) => item.querySelector(".ribbon-label")!.textContent)).toEqual(["Readable", "Two random pieces", "Sealed three times"]);
    expect(items[2]).toHaveAttribute("aria-current", "step");
    expect(items[2]).toHaveClass("ribbon-cipher", "on");
    expect(items[0]).not.toHaveAttribute("aria-current");
    expect(items[1]!.querySelector(".sr-only")).toHaveTextContent("9f3a 01c2");
  });

  it("lays out in, out, caption, reason and ribbon over the 3D view", async () => {
    const { container } = render(
      <StageBox
        label="Demo stage"
        scene={{ caption: "scene caption", nodes: [], links: [] }}
        input={<p>question here</p>}
        output={<p>answer here</p>}
        corner={<p>12 checked</p>}
        caption="Step 3 of 8 · Seal."
        why="Everything on the wire is unreadable."
        forms={forms}
        current={["sealed"]}
      />,
    );
    expect(screen.getByRole("group", { name: "Demo stage" })).toBeInTheDocument();
    expect(screen.getByTestId("view")).toHaveAttribute("data-variant", "show");
    const side = container.querySelector(".stage-side")!;
    expect(side).toHaveAttribute("data-inset", "left");
    expect(side).toHaveTextContent("question hereanswer here");
    expect(container.querySelector(".stage-corner")).toHaveTextContent("12 checked");
    expect(container.querySelector(".stage-caption")).toHaveTextContent("Step 3 of 8 · Seal.Why it matters Everything on the wire is unreadable.");
    expect(container.querySelector(".stage-foot .ribbon")).not.toBeNull();
    expect((await axe(container)).violations).toHaveLength(0);
  });
});
