import { fireEvent, render, screen } from "@testing-library/react";
import { axe } from "jest-axe";
import { describe, expect, it, vi } from "vitest";

vi.mock("@/components/Scene3DView", () => ({
  Scene3DView: ({ scene, overlay, variant }: { scene: { caption: string }; overlay?: React.ReactNode; variant?: string }) => (
    <div data-testid="view" data-variant={variant ?? "lab"} data-caption={scene.caption}>{overlay}</div>
  ),
}));

import { LabShell } from "@/components/LabShell";
import type { LiveStory } from "@/components/LiveIO";
import type { Scene3D } from "@/lib/scene/types";
import { journeyOf } from "@/lib/show/journey";
import { questionSetting } from "@/lib/show/question";

const scene: Scene3D = {
  caption: "Two boxes.",
  nodes: [
    { id: "a", kind: "box", position: [-2, 0.5, 1], size: [1, 1, 1], tone: "accent" },
    { id: "b", kind: "box", position: [4, 0.5, -1], size: [1, 1, 1], tone: "plain" },
  ],
  links: [{ id: "ab", from: "a", to: "b", tone: "plain" }],
};

function story(extra: Partial<LiveStory> = {}): LiveStory {
  return {
    plain: "Your question is cut in two.",
    stage: "split",
    input: <p>Its words become numbers.</p>,
    seenTitle: "What one worker sees",
    seen: <code>81234, 99120</code>,
    output: <strong>19, 43</strong>,
    why: "One piece alone is random.",
    relies: "Workers do not pool pieces.",
    limit: "Two workers together can rebuild it.",
    screenSeen: "81234",
    ...extra,
  };
}

describe("lab shell with a story", () => {
  it("tells the lab in plain words, shows in, seen and out, and keeps the exact numbers", async () => {
    const { container } = render(<LabShell id="demo" title="Demo lab" caption="Technical caption." scene={scene} story={story()}><p>exact table</p></LabShell>);
    expect(screen.getByText("Your question is cut in two.")).toBeInTheDocument();
    expect(screen.getByText("Technical caption.")).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 3, name: "What goes in" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 3, name: "What one worker sees" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 3, name: "What comes out" })).toBeInTheDocument();
    // The question comes from the shared list of prepared questions, and the model's full answer comes out.
    expect(screen.getByLabelText("Your question")).toHaveValue("text-shares");
    expect(screen.getByRole("option", { name: "Who runs the workers?" })).toBeInTheDocument();
    expect(document.querySelector(".liveio-answer")).toHaveTextContent(`The model's answer ${journeyOf("text-shares").answer}`);
    expect(document.querySelector(".liveio-result")).toHaveTextContent("19, 43");
    // The ribbon along the bottom of the stage is the trip, with this lab's stop marked.
    expect([...document.querySelectorAll(".ribbon-label")].map((item) => item.textContent)).toEqual(["Your question", "Prepare", "Split · this lab", "Place", "Send", "Compute", "Check", "Wipe", "Answer"]);
    expect([...document.querySelectorAll(".ribbon-form[aria-current=step] .ribbon-label")].map((item) => item.textContent)).toEqual(["Your question", "Split · this lab", "Answer"]);
    expect(screen.getByRole("group", { name: "Demo lab: live stage" })).toBeInTheDocument();
    expect(screen.getByTestId("view")).toHaveAttribute("data-variant", "show");
    expect(document.querySelector(".stage-side:not(.stage-right)")).toHaveTextContent("What goes in");
    expect(document.querySelector(".stage-side:not(.stage-right)")).toHaveTextContent("What one worker sees");
    expect(document.querySelector(".stage-right")).toHaveTextContent("What comes out");
    expect(document.querySelector(".liveio-honest")).toHaveTextContent("no language or image model runs in this manual");
    expect(screen.getByText("Why this protects you").nextElementSibling).toHaveTextContent("One piece alone is random.");
    expect(screen.getByText("Where it stops").nextElementSibling).toHaveTextContent("Two workers together can rebuild it.");
    expect(screen.getByText("exact table").closest("details")).toHaveAttribute("open");
    expect((await axe(container)).violations).toHaveLength(0);
  });

  it("offers the attack and reports what happened to it", () => {
    const onToggle = vi.fn();
    const { rerender } = render(<LabShell id="demo" title="Demo lab" caption="c" scene={scene} story={story({ attack: { label: "tamper with a worker", on: false, onToggle, outcome: "All four products passed." } })}><p>x</p></LabShell>);
    const button = screen.getByRole("button", { name: "Try to break it: tamper with a worker" });
    expect(button).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByRole("status")).toHaveTextContent("All four products passed.");
    fireEvent.click(button);
    expect(onToggle).toHaveBeenCalledTimes(1);
    rerender(<LabShell id="demo" title="Demo lab" caption="c" scene={scene} story={story({ outputTone: "warn", attack: { label: "tamper with a worker", on: true, onToggle, outcome: "Caught." } })}><p>x</p></LabShell>);
    expect(screen.getByRole("button", { name: "Try to break it: tamper with a worker" })).toHaveAttribute("aria-pressed", "true");
    expect(document.querySelector(".liveio-out")).toHaveClass("warn");
  });

  it("follows the shared question, withholds the answer when a step fails, and uses the everyday title", () => {
    const { rerender } = render(<LabShell id="demo" title="Additive share grid" plainTitle="Splitting your question" caption="Technical." story={story()}><p>x</p></LabShell>);
    expect(screen.getByRole("heading", { level: 2, name: "Splitting your question" })).toBeInTheDocument();
    expect(document.querySelector(".lab-caption")).toHaveTextContent("Additive share grid. Technical.");
    fireEvent.change(screen.getByLabelText("Your question"), { target: { value: "text-memory" } });
    expect(questionSetting.get()).toBe("text-memory");
    expect(document.querySelector(".liveio-answer")).toHaveTextContent(journeyOf("text-memory").answer);
    rerender(<LabShell id="demo" title="Additive share grid" plainTitle="Splitting your question" caption="Technical." story={story({ released: false, outputTone: "warn" })}><p>x</p></LabShell>);
    expect(document.querySelector(".liveio-answer")).toHaveTextContent("The model's answer No answer is released.");
    expect(document.querySelector(".liveio-answer")).not.toHaveTextContent(journeyOf("text-memory").answer);
    questionSetting.set("text-shares");
  });

  it("shows a finished picture as the answer to a picture request", () => {
    render(<LabShell id="demo" title="Diffusion" caption="c" story={story({ kind: "picture" })}><p>x</p></LabShell>);
    expect(screen.getByLabelText("Your picture request")).toHaveValue("picture-sunrise");
    expect(screen.getByRole("img", { name: "The finished picture: A sunrise over the water" })).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Your picture request"), { target: { value: "picture-city" } });
    expect(screen.getByRole("img", { name: "The finished picture: A city at night under the moon" })).toBeInTheDocument();
  });

  it("plays the lab from the header and leaves labs without a story as they were", () => {
    const onToggle = vi.fn();
    const { rerender } = render(<LabShell id="demo" title="Demo lab" caption="c" watch={{ playing: false, onToggle }}><p>plain child</p></LabShell>);
    fireEvent.click(screen.getByRole("button", { name: "Watch" }));
    expect(onToggle).toHaveBeenCalledTimes(1);
    expect(screen.getByText("plain child").closest("details")).toBeNull();
    rerender(<LabShell id="demo" title="Demo lab" caption="c" watch={{ playing: true, onToggle }}><p>plain child</p></LabShell>);
    expect(screen.getByRole("button", { name: "Stop" })).toBeInTheDocument();
  });
});
