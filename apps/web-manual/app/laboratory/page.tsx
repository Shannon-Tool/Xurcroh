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

export const metadata: Metadata = {
  title: "Interactive Laboratory",
  description: "Executable algebraic, distributed inference, placement, and security laboratories.",
};

export default function LaboratoryPage() {
  return (
    <article className="article">
      <header className="hero">
        <span className="evidence">Executable teaching models · not production inference</span>
        <h1>Interactive laboratory</h1>
        <p>Every control changes real computational state. Fortress examples use exact BigInt arithmetic and explicitly separate simulation from measured production evidence.</p>
      </header>
      <ShareGridLab />
      <FreivaldsLab />
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
