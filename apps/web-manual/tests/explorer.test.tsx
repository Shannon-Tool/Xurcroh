import { render, screen, waitFor } from "@testing-library/react";
import { axe } from "jest-axe";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/scene/stage", () => ({
  createStage: () => ({ rendererName: "Test GPU", setPalette: () => undefined, setQuality: () => undefined, draw: () => ({ labels: [], animating: false }), pick: () => null, setScreen: () => undefined, dispose: () => undefined }),
}));

import Home from "@/app/page";
import LaboratoryPage from "@/app/laboratory/page";
import ShowPage from "@/app/show/page";

function setReducedMotion(reduced: boolean) {
  vi.stubGlobal("matchMedia", (query: string) => ({ matches: reduced && query.includes("reduce"), media: query, addEventListener: () => undefined, removeEventListener: () => undefined }));
}

beforeEach(() => {
  vi.stubGlobal("WebGLRenderingContext", class {});
  // jsdom cannot draw on a canvas; the pages must not depend on it.
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(() => null);
  setReducedMotion(true);
});
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("home page", () => {
  it("opens with the live demonstration, says what it is, and stays accessible", async () => {
    const { container } = render(<Home />);
    expect(screen.getByRole("heading", { name: "Live demonstration" })).toBeInTheDocument();
    expect(container.querySelector(".show-truth")).toHaveTextContent(/prepared toy models, and the relays and enclaves are stand-ins inside this page; the share arithmetic, the three layers of encryption and the signature checks are real/);
    expect(container.querySelector(".show-truth")).toHaveTextContent(/production engine is hardware-validated/);
    expect(screen.getByRole("link", { name: "Open the laboratory" })).toHaveAttribute("href", "/laboratory");
    expect(screen.getByRole("link", { name: "Demonstration on its own page" })).toHaveAttribute("href", "/show");
    await waitFor(() => expect(container.querySelector(".scene3d canvas")).not.toBeNull());
    expect((await axe(container)).violations).toHaveLength(0);
  }, 20000);

  it("offers the demonstration on its own page with the facts open", async () => {
    const { container } = render(<ShowPage />);
    expect(screen.getByRole("heading", { level: 1, name: "Live demonstration" })).toBeInTheDocument();
    expect(container.querySelector("details.show-details")).toHaveAttribute("open");
    expect(screen.getByRole("heading", { level: 2, name: "Prepared or simulated" })).toBeInTheDocument();
    expect((await axe(container)).violations).toHaveLength(0);
  }, 20000);
});

describe("laboratory navigator", () => {
  it("links to every laboratory on the page, in order", () => {
    const { container } = render(<LaboratoryPage />);
    const targets = [...container.querySelectorAll(".lab-nav a")].map((link) => link.getAttribute("href"));
    const labs = [...container.querySelectorAll("section.lab")].map((lab) => `#${lab.id}`);
    expect(labs).toHaveLength(11);
    expect(targets).toEqual(labs);
  });
});
