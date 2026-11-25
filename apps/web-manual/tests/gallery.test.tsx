import { render, screen } from "@testing-library/react";
import { axe } from "jest-axe";
import { describe, expect, it, vi } from "vitest";

vi.mock("@/components/Scene3DView", () => ({ Scene3DView: ({ overlay }: { overlay?: React.ReactNode }) => <div data-testid="view">{overlay}</div> }));

import GalleryPage from "@/app/gallery/page";
import { GALLERY } from "@/components/Gallery";

describe("gallery of every animation", () => {
  it("lists the pipeline, all eleven labs and the tour, each with the files that define it", () => {
    expect(GALLERY).toHaveLength(14);
    expect(new Set(GALLERY.map((item) => item.id)).size).toBe(14);
    expect(GALLERY.filter((item) => item.where.startsWith("/laboratory#"))).toHaveLength(11);
    for (const item of GALLERY) {
      expect(item.files).toMatch(/^components\//);
      expect(item.where).toMatch(/^\//);
    }
  });

  it("puts every stage on one page under its name, file and home, with an index", async () => {
    const { container } = render(<GalleryPage />);
    expect(screen.getByRole("heading", { level: 1, name: "All animations" })).toBeInTheDocument();
    expect(container.querySelectorAll(".gallery-item")).toHaveLength(14);
    // Two pipeline stages, eleven lab stages and the tour's stage.
    expect(container.querySelectorAll(".stage")).toHaveLength(14);
    // Every stage shows its data along the bottom: a ribbon of forms, or a lab's row of steps.
    expect(container.querySelectorAll(".gallery-item .stage .ribbon, .gallery-item .stage .track")).toHaveLength(14);
    const index = [...container.querySelectorAll(".lab-nav a")].map((link) => link.getAttribute("href"));
    expect(index).toEqual(GALLERY.map((item) => `#g-${item.id}`));
    for (const target of index) expect(container.querySelector(target!)).not.toBeNull();
    expect(screen.getByText("components/labs/OnionLab.tsx · lib/scene/builders.ts")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "/laboratory#onion" })).toHaveAttribute("href", "/laboratory#onion");
    // Labs are shown as their stage only: no lab headers or exact-number sections here.
    expect(container.querySelectorAll("section.lab")).toHaveLength(0);
    expect(container.querySelector(".hero")).toHaveTextContent("the relays and enclaves are stand-ins");
    expect((await axe(container)).violations).toHaveLength(0);
  }, 30000);
});
