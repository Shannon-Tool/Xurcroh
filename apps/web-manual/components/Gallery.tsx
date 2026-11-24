"use client";

import Link from "next/link";
import { LabEmbedContext } from "@/components/LabShell";
import { LiveShow } from "@/components/LiveShow";
import { ProtectionTour } from "@/components/ProtectionTour";
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

interface GalleryItem {
  id: string;
  name: string;
  /** Where the animation is defined, for whoever polishes it. */
  files: string;
  /** Where readers meet it. */
  where: string;
  render: () => React.ReactNode;
}

const lab = (id: string, name: string, file: string, render: () => React.ReactNode): GalleryItem => ({
  id, name, files: `components/labs/${file}.tsx · lib/scene/builders.ts`, where: `/laboratory#${id}`,
  render: () => <LabEmbedContext.Provider value="stage">{render()}</LabEmbedContext.Provider>,
});

export const GALLERY: GalleryItem[] = [
  { id: "pipeline", name: "The pipeline: text, picture, clip, your text, your drawing", files: "components/LiveShow.tsx · lib/scene/show-scene.ts · lib/show/timeline.ts", where: "/show", render: () => <LiveShow autoPlay /> },
  { id: "pipeline-chapter", name: "The pipeline at chapter size", files: "components/LiveShow.tsx (compact)", where: "/05-system-architecture", render: () => <LiveShow compact autoPlay /> },
  lab("share-grid", "Splitting your question", "ShareGridLab", () => <ShareGridLab />),
  lab("freivalds", "Catching a wrong answer", "FreivaldsLab", () => <FreivaldsLab />),
  lab("quantized-formats", "Storing the model in pieces", "QuantizedFormatLab", () => <QuantizedFormatLab />),
  lab("placement", "Deciding who gets which piece", "PlacementLab", () => <PlacementLab />),
  lab("conjugation", "Disguising the numbers", "ConjugationLab", () => <ConjugationLab />),
  lab("fwht", "Mixing the numbers quickly", "FwhtLab", () => <FwhtLab />),
  lab("pipeline", "Getting your answer when a worker fails", "PipelineLab", () => <PipelineLab />),
  lab("onion", "Hiding the route", "OnionLab", () => <OnionLab />),
  lab("moe", "Hiding which experts were used", "MoeLab", () => <MoeLab />),
  lab("diffusion", "Making a picture", "DiffusionLab", () => <DiffusionLab />),
  lab("memory", "Wiping memory afterwards", "MemoryLab", () => <MemoryLab />),
  { id: "tour", name: "The protection tour", files: "components/ProtectionTour.tsx · lib/scene/protection-scene.ts", where: "/protection", render: () => <ProtectionTour autoPlay /> },
].map((item, index, all) => (all.findIndex((other) => other.id === item.id) === index ? item : { ...item, id: `lab-${item.id}` }));

/** Every animation in the manual on one page, each on its stage, for review and polishing. */
export function Gallery() {
  return (
    <div className="gallery">
      <nav className="lab-nav" aria-label="Animations on this page">
        {GALLERY.map((item) => <a key={item.id} href={`#g-${item.id}`}>{item.name}</a>)}
      </nav>
      {GALLERY.map((item, index) => (
        <section key={item.id} id={`g-${item.id}`} className="gallery-item" aria-labelledby={`g-${item.id}-title`}>
          <h2 id={`g-${item.id}-title`}>{index + 1}. {item.name}</h2>
          <p className="gallery-meta">Defined in <code>{item.files}</code>. Readers see it at <Link href={item.where}>{item.where}</Link>.</p>
          {item.render()}
        </section>
      ))}
    </div>
  );
}
