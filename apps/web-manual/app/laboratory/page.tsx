import type { Metadata } from "next";
import { ConjugationLab } from "@/components/labs/ConjugationLab";
import { FwhtLab } from "@/components/labs/FwhtLab";
import { PipelineLab } from "@/components/labs/PipelineLab";
import { OnionLab } from "@/components/labs/OnionLab";
import { MoeLab } from "@/components/labs/MoeLab";
import { DiffusionLab } from "@/components/labs/DiffusionLab";
import { MemoryLab } from "@/components/labs/MemoryLab";
import { ShareGridLab } from "@/components/labs/ShareGridLab";
import { PlacementLab } from "@/components/labs/PlacementLab";
import { FreivaldsLab } from "@/components/labs/FreivaldsLab";
import { QuantizedFormatLab } from "@/components/labs/QuantizedFormatLab";

export const metadata: Metadata = {
  title: "Interactive Laboratory",
  description: "Executable algebraic, distributed inference, placement, and security laboratories.",
};

/** In page order; each id is the `id` its lab passes to `LabShell`. */
const LABS = [
  { id: "share-grid", name: "Share grid" },
  { id: "freivalds", name: "Freivalds check" },
  { id: "quantized-formats", name: "Quantized formats" },
  { id: "placement", name: "Placement" },
  { id: "conjugation", name: "Conjugation" },
  { id: "fwht", name: "FWHT" },
  { id: "pipeline", name: "Pipeline" },
  { id: "onion", name: "Onion routing" },
  { id: "moe", name: "Mixture of experts" },
  { id: "diffusion", name: "Diffusion" },
  { id: "memory", name: "Memory" },
] as const;

export default function LaboratoryPage() {
  return (
    <article className="article article-wide">
      <header className="hero">
        <span className="evidence">Executable teaching models · not production inference</span>
        <h1>Interactive laboratory</h1>
        <p>Every control changes real computational state. Fortress examples use exact BigInt arithmetic and explicitly separate simulation from measured production evidence. Each laboratory draws its current state as an animated 3D scene; the text beneath it is the exact view.</p>
      </header>
      <nav className="lab-nav" aria-label="Laboratories on this page">
        {LABS.map((lab) => <a key={lab.id} href={`#${lab.id}`}>{lab.name}</a>)}
      </nav>
      <ShareGridLab />
      <FreivaldsLab />
      <QuantizedFormatLab />
      <PlacementLab />
      <ConjugationLab />
      <FwhtLab />
      <PipelineLab />
      <OnionLab />
      <MoeLab />
      <DiffusionLab />
      <MemoryLab />
    </article>
  );
}
