import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { axe } from "jest-axe";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { LiveScene } from "@/lib/scene/tween";

const draw = vi.fn((live: LiveScene) => { void live; return { labels: [], animating: false }; });
vi.mock("@/lib/scene/stage", () => ({
  createStage: () => ({ rendererName: "Test GPU", setPalette: () => undefined, setQuality: () => undefined, draw, pick: () => null, setScreen: () => undefined, dispose: () => undefined }),
}));

import LaboratoryPage from "@/app/laboratory/page";
import { ConjugationLab } from "@/components/labs/ConjugationLab";
import { QuantizedFormatLab } from "@/components/labs/QuantizedFormatLab";
import { ShareGridLab } from "@/components/labs/ShareGridLab";

describe("labs with a 3D view", () => {
  beforeEach(() => {
    vi.stubGlobal("WebGLRenderingContext", class {});
    // Reduced motion keeps the views still, so the page is quiet while it is checked.
    vi.stubGlobal("matchMedia", (query: string) => ({ matches: query.includes("reduce"), media: query, addEventListener: () => undefined, removeEventListener: () => undefined }));
    draw.mockClear();
  });
  afterEach(() => vi.unstubAllGlobals());

  it("gives every lab on the laboratory page a 3D view and stays accessible", async () => {
    const { container } = render(<LaboratoryPage />);
    const labs = container.querySelectorAll("section.lab");
    expect(labs).toHaveLength(11);
    // The onion lab draws once its keys are generated; the other ten draw at once.
    await waitFor(() => expect(container.querySelectorAll("section.lab .scene3d").length).toBeGreaterThanOrEqual(10));
    expect((await axe(container)).violations).toHaveLength(0);
  }, 20000);

  it("redraws the share grid when the reader steps the playback", async () => {
    const { container } = render(<ShareGridLab />);
    await waitFor(() => expect(draw).toHaveBeenCalled());
    const caption = () => container.querySelector(".track-now [role=status]")!.textContent;
    const before = caption();
    const activeBefore = draw.mock.calls.at(-1)![0].nodes.find((node) => node.emphasis > 0)?.id;
    // On to the first worker: the step sentence and the lit part of the scene both change.
    for (let step = 0; step < 3; step += 1) fireEvent.click(screen.getByRole("button", { name: "Step forward" }));
    await waitFor(() => expect(caption()).not.toBe(before));
    await waitFor(() => expect(draw.mock.calls.at(-1)![0].nodes.some((node) => node.id !== activeBefore && node.emphasis > 0)).toBe(true));
  });

  it("reports bad codes in the quantized lab instead of crashing", () => {
    render(<QuantizedFormatLab />);
    fireEvent.change(screen.getByLabelText("Integer codes"), { target: { value: "x" } });
    expect(screen.getByRole("alert")).toHaveTextContent(/non-negative integer codes/);
  });

  it("computes the rotated norm rather than repeating the input norm", () => {
    render(<ConjugationLab />);
    // The lab works on the first two letters of the default question, H and o: 8 and 15, of length 17.
    expect(screen.getByText(/Rotated norm/)).toHaveTextContent("17.000000");
    expect(screen.getByText(/Input norm/)).toHaveTextContent("17.000000");
  });
});
