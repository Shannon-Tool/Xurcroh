import { existsSync, readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/components/Scene3DView", () => ({ Scene3DView: ({ overlay }: { overlay?: React.ReactNode }) => <div data-testid="view">{overlay}</div> }));

import LaboratoryPage from "@/app/laboratory/page";
import { GALLERY } from "@/components/Gallery";
import { LiveShow } from "@/components/LiveShow";
import { journeyOf, pictureIds, pictureJourneyOf, questionIds } from "@/lib/show/journey";
import { PROTECTION_LAYERS } from "@/lib/show/protection";
import { inspectGif } from "@/scripts/lib/gif.mjs";
import { END, START, plainLabel, renderSection, type RecordingFacts } from "@/scripts/lib/readme.mjs";
import { GROUPS, RECORDINGS } from "@/scripts/recordings.mjs";

// The recordings live at the top of the repository, next to the README that shows them.
const ROOT = path.resolve(__dirname, "../../..");
const MEDIA = path.join(ROOT, "docs/media/animations");
const MAX_FILE = 5_000_000;
const MAX_TOTAL = 80_000_000;

const index = JSON.parse(readFileSync(path.join(MEDIA, "index.json"), "utf8")) as Record<string, RecordingFacts>;
const readme = readFileSync(path.join(ROOT, "README.md"), "utf8");
const used = new Set(Object.values(index).flatMap((facts) => facts.controls));
const usedIn = (id: string) => new Set(index[id]?.controls ?? []);

beforeEach(() => {
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(() => null);
  vi.stubGlobal("matchMedia", (query: string) => ({ matches: query.includes("reduce"), media: query, addEventListener: () => undefined, removeEventListener: () => undefined }));
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("recordings of the animations", () => {
  it("has a valid, looping GIF for every listed recording, within the size limits, and no stray files", () => {
    expect(new Set(RECORDINGS.map((recording) => recording.id)).size).toBe(RECORDINGS.length);
    expect(new Set(RECORDINGS.map((recording) => recording.file)).size).toBe(RECORDINGS.length);
    let total = 0;
    for (const recording of RECORDINGS) {
      const facts = index[recording.id];
      expect(facts, `${recording.id} has not been recorded`).toBeDefined();
      const file = path.join(MEDIA, recording.file);
      expect(existsSync(file), `${recording.file} is missing`).toBe(true);
      const bytes = readFileSync(file);
      const gif = inspectGif(bytes);
      expect([gif.width, gif.height], recording.file).toEqual([facts!.width, facts!.height]);
      expect(gif.frames, `${recording.file} does not move`).toBeGreaterThan(1);
      expect(gif.loops, `${recording.file} does not loop`).toBe(0);
      expect(Math.abs(gif.durationMs / 1000 - facts!.seconds), `${recording.file} length`).toBeLessThan(0.5);
      expect(bytes.length, `${recording.file} is too large`).toBeLessThanOrEqual(MAX_FILE);
      expect(bytes.length).toBe(facts!.bytes);
      total += bytes.length;
    }
    expect(total).toBeLessThanOrEqual(MAX_TOTAL);
    expect(readdirSync(MEDIA).filter((name) => name.endsWith(".gif")).sort()).toEqual(RECORDINGS.map((recording) => recording.file).sort());
    expect(Object.keys(index).sort()).toEqual(RECORDINGS.map((recording) => recording.id).sort());
    expect(RECORDINGS.filter((recording) => recording.lead)).toHaveLength(1);
    for (const recording of RECORDINGS) expect(GROUPS.some((group) => group.id === recording.group), recording.id).toBe(true);
  });

  it("records every stage of the gallery, every layer of the tour and every chapter diagram", () => {
    const stages = new Set(RECORDINGS.map((recording) => recording.stage));
    for (const item of GALLERY) expect(stages.has(item.id), `no recording of the stage “${item.name}”`).toBe(true);
    for (const layer of PROTECTION_LAYERS) expect(usedIn("tour").has(`Layer: ${layer.name}`), layer.name).toBe(true);
    expect(usedIn("tour").has("Play the tour")).toBe(true);
    for (const diagram of ["architecture", "runtime", "memory"]) {
      const recording = RECORDINGS.find((entry) => entry.diagram === diagram);
      expect(recording, `no recording of the ${diagram} diagram`).toBeDefined();
      const shown = usedIn(recording!.id);
      for (const control of ["Schematic drawing, step by step", "Play walk-through", "Pause", "Next", "Back", "Show all"]) expect(shown.has(control), `${diagram}: ${control}`).toBe(true);
      expect([...shown].some((control) => control.startsWith("Look inside: ")), `${diagram}: look inside`).toBe(true);
    }
  });

  it("uses every control of the live demonstration in some recording", () => {
    render(<LiveShow factsOpen />);
    const group = (name: string) => within(screen.getByRole("group", { name })).getAllByRole("button").map((button) => button.textContent!.trim());
    const needed = [...group("What to produce"), ...group("Prepared prompts"), ...group("Whose eyes"), ...group("Demonstration controls"), ...group("Try to break it")];
    fireEvent.click(screen.getByRole("button", { name: "Picture" }));
    needed.push(...group("Prepared pictures"));
    fireEvent.click(screen.getByRole("button", { name: "Your text" }));
    expect(screen.getByLabelText("Type something")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Your drawing" }));
    needed.push(...group("Ready-made drawings"));
    // The other state of every switch, the two fields of the reader's own input, and the facts panel.
    needed.push("Pause", "Speed: 2×", "Words: technical", "Exit full screen", "Type something", "Drawing pad", "What is real here");
    expect(needed.length).toBeGreaterThan(30);
    for (const control of needed) expect(used.has(control), `no recording uses “${control}”`).toBe(true);
  });

  it("uses every control of every laboratory in that laboratory's recording", () => {
    const { container } = render(<LaboratoryPage />);
    const labs = [...container.querySelectorAll<HTMLElement>("section.lab")];
    expect(labs).toHaveLength(11);
    for (const lab of labs) {
      const shown = usedIn(`lab-${lab.id}-controls`);
      const needed: string[] = [];
      for (const control of lab.querySelectorAll<HTMLElement>(".lab-controls input, .lab-controls select, .lab-controls button")) {
        const label = control.closest("label");
        const own = label ? [...label.childNodes].filter((node) => node.nodeType === 3).map((node) => node.textContent).join(" ") : control.textContent!;
        const name = plainLabel(own);
        if (control instanceof HTMLSelectElement) needed.push(...[...control.options].map((option) => `${name}: ${option.textContent!.trim()}`));
        else needed.push(name);
      }
      if (lab.querySelector(".lab-controls .playback")) needed.push("Play", "Pause");
      const attack = lab.querySelector(".liveio-attack button");
      if (attack) needed.push(attack.textContent!.trim());
      const header = [...lab.querySelectorAll(".lab-header button")].map((button) => button.textContent!.trim());
      if (header.includes("Watch") || header.includes("Stop")) needed.push("Watch", "Stop");
      needed.push("Hide 3D view", "Show 3D view");
      expect(needed.length, lab.id).toBeGreaterThanOrEqual(4);
      for (const control of needed) expect(shown.has(control), `${lab.id}: its recording does not use “${control}”`).toBe(true);
    }
    // The Fullscreen button is the same shared button on every laboratory; it is recorded once.
    expect(used.has("Fullscreen") && used.has("Exit fullscreen")).toBe(true);
  });

  it("shows every value of every setting", () => {
    const needed = [
      ...questionIds().map((id) => `Your question: ${journeyOf(id).question}`),
      ...pictureIds().map((id) => `Your picture request: ${pictureJourneyOf(id).request}`),
      "Words: plain", "Words: technical",
      "Effects: on", "Effects: off", "Effects: auto",
      "Pause motion", "Play motion",
      "Dark theme", "Light theme",
      "Drag to turn", "Zoom in", "Zoom out", "Reset view",
    ];
    for (const control of needed) expect(used.has(control), `no recording shows “${control}”`).toBe(true);
  });

  it("shows every recording on the repository README, generated from the list, with factual wording", () => {
    const from = readme.indexOf(START);
    const to = readme.indexOf(END);
    expect(from).toBeGreaterThan(0);
    expect(readme.slice(from, to + END.length)).toBe(renderSection(RECORDINGS, GROUPS, index));
    for (const recording of RECORDINGS) expect(readme, recording.file).toContain(`docs/media/animations/${recording.file}"`);
    const section = readme.slice(from, to);
    expect(section).toContain("not the production engine");
    expect(section).toMatch(/the questions and answers are prepared, the models are toys/);
    expect(section).toMatch(/relays stand in for a Tor circuit, and the enclaves are simulated/);
    expect(section).not.toMatch(/\b(guarantee|unbreakable|impossible|100%|perfectly safe|military|state of the art)\b/i);
    // The section sits right under the security status, which is unchanged.
    expect(readme).toContain("> **Security status:** designed for defence in depth; claims under evaluation.");
    expect(readme.indexOf("> **Security status:**")).toBeLessThan(from);
    expect(readme.indexOf("## Public Python API")).toBeGreaterThan(to);
  });
});
