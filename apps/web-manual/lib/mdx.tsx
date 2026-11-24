import type { MDXComponents } from "mdx/types";
import { Equation } from "@/components/Equation";
import { Evidence } from "@/components/Evidence";
import { Figure } from "@/components/Figure";
import { Citation } from "@/components/Citation";
import { LabLink } from "@/components/LabLink";
import { LabEmbedContext } from "@/components/LabShell";
import { LiveShow } from "@/components/LiveShow";
import { MemoryLab } from "@/components/labs/MemoryLab";
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
  MemoryStage: () => <LabEmbedContext.Provider value="stage"><MemoryLab /></LabEmbedContext.Provider>,
  RuntimeFlowDiagram,
  MemoryLifecycleDiagram,
};
