import type { MDXComponents } from "mdx/types";
import { Equation } from "@/components/Equation";
import { Evidence } from "@/components/Evidence";
import { Figure } from "@/components/Figure";
import { Citation } from "@/components/Citation";
import { LabLink } from "@/components/LabLink";
import { LiveShow } from "@/components/LiveShow";
import { MemoryStage } from "@/components/MemoryStage";
import { ArchitectureDiagram } from "@/components/diagrams/ArchitectureDiagram";
import { MemoryLifecycleDiagram } from "@/components/diagrams/MemoryLifecycleDiagram";
import { RuntimeFlowDiagram } from "@/components/diagrams/RuntimeFlowDiagram";

export const mdxComponents: MDXComponents = {
  Equation,
  Evidence,
  Figure,
  Citation,
  LabLink,
  ArchitectureDiagram,
  LiveDemo: () => <LiveShow compact autoPlay />,
  MemoryStage,
  RuntimeFlowDiagram,
  MemoryLifecycleDiagram,
};
