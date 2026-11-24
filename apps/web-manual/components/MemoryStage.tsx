"use client";

import { LabEmbedContext } from "@/components/LabShell";
import { MemoryLab } from "@/components/labs/MemoryLab";

/** The memory laboratory's live stage on its own, for the memory chapter. */
export function MemoryStage() {
  return <LabEmbedContext.Provider value="stage"><MemoryLab /></LabEmbedContext.Provider>;
}
