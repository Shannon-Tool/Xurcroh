import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { axe } from "jest-axe";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Scene3D } from "@/lib/scene/types";

const update = vi.fn((scene: Scene3D) => { void scene; });
vi.mock("@/lib/scene/stage", () => ({ createStage: () => ({ update, view: () => [], dispose: () => undefined }) }));

import LaboratoryPage from "@/app/laboratory/page";
import { ConjugationLab } from "@/components/labs/ConjugationLab";
import { QuantizedFormatLab } from "@/components/labs/QuantizedFormatLab";
import { ShareGridLab } from "@/components/labs/ShareGridLab";

describe("labs with a 3D view", () => {
  beforeEach(() => {
    vi.stubGlobal("WebGLRenderingContext", class {});
    update.mockClear();
  });
  afterEach(() => vi.unstubAllGlobals());

  it("gives every lab on the laboratory page a 3D view and stays accessible", async () => {
    const { container } = render(<LaboratoryPage />);
    const labs = container.querySelectorAll("section.lab");
    expect(labs).toHaveLength(11);
    // The onion lab draws once its keys are generated; the other ten draw at once.
    await waitFor(() => expect(container.querySelectorAll("section.lab .scene3d").length).toBeGreaterThanOrEqual(10));
    expect((await axe(container)).violations).toHaveLength(0);
  });

  it("redraws the share grid when the reader steps the playback", async () => {
    render(<ShareGridLab />);
    await waitFor(() => expect(update).toHaveBeenCalled());
    const before = update.mock.calls.at(-1)![0].caption;
    fireEvent.click(screen.getByRole("button", { name: "Step forward" }));
    await waitFor(() => expect(update.mock.calls.at(-1)![0].caption).not.toBe(before));
  });

  it("reports bad codes in the quantized lab instead of crashing", () => {
    render(<QuantizedFormatLab />);
    fireEvent.change(screen.getByLabelText("Integer codes"), { target: { value: "x" } });
    expect(screen.getByRole("alert")).toHaveTextContent(/non-negative integer codes/);
  });

  it("computes the rotated norm rather than repeating the input norm", () => {
    render(<ConjugationLab />);
    expect(screen.getByText(/Rotated norm/)).toHaveTextContent("2.236068");
  });
});
