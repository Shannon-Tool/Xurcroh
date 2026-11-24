"use client";

import { memoryData } from "@/lib/show/diagram-data";
import { Arrow, Box, Diagram, Note, type DiagramLive, type DiagramStep } from "@/components/diagrams/primitives";

const LIVE: DiagramLive = { compute: memoryData };

const STAGES = [
  ["Allocate", "buffer handed out"],
  ["Execute", "secret data written"],
  ["Release", "allocator marks free"],
  ["Synchronise", "wait for the stream"],
  ["Zero", "overwrite the bytes"],
  ["Reuse", "reads back zeros"],
] as const;

const STEPS: DiagramStep[] = [
  { title: "Allocate", text: "A request is handed a buffer on the accelerator.", show: ["s0"] },
  { title: "Execute", text: "Activations and cache entries, which are secret, are written into it.", show: ["s1", "m0"] },
  { title: "Release", text: "The allocator marks the buffer free. Freeing is bookkeeping: the secret bytes are still there.", show: ["s2", "m1"] },
  { title: "Synchronise", text: "The runtime waits until every queued operation on that buffer has finished.", show: ["s3", "m2"] },
  { title: "Zero", text: "Only now are the bytes overwritten, so the wipe cannot be overtaken.", show: ["s4", "m3"] },
  { title: "Reuse", text: "The next owner reads back zeros.", show: ["s5", "m4"] },
  { title: "If synchronisation is skipped", text: "The wipe may still be waiting in the queue when the buffer is handed on, and the next owner reads the previous request's bytes.", show: ["skip", "s2", "s5"] },
];

export function MemoryLifecycleDiagram({ caption }: { caption: string }) {
  return (
    <Diagram
      title="GPU memory lifecycle"
      description="A buffer is allocated, written with secret data, released, then the stream is synchronised and the bytes are zeroed before the buffer is reused. If synchronisation is skipped the wipe may still be queued when the buffer is handed to the next owner, who can then read stale secret bytes."
      width={960}
      height={270}
      caption={caption}
      steps={STEPS}
      live={LIVE}
    >
      {(arrowId) => (
        <>
          {STAGES.map(([name, detail], index) => (
            <Box key={name} id={`s${index}`} x={16 + index * 158} y={60} width={138} height={70} lines={[name, detail]} tone={index === 3 || index === 4 ? "good" : "plain"} />
          ))}
          {STAGES.slice(0, -1).map(([name], index) => (
            <Arrow key={name} id={`m${index}`} from={`s${index}`} to={`s${index + 1}`} arrowId={arrowId} d={`M${154 + index * 158} 95 H${174 + index * 158}`} />
          ))}
          <Arrow id="skip" from="s2" to="s5" arrowId={arrowId} d="M401 130 C401 226 875 226 875 130" tone="warn" dashed label="synchronisation skipped: the wipe is still queued" labelX={638} labelY={226} />
          <Note x={520} y={246} lines={["The next owner can read stale secret bytes."]} />
          <Note x={16} y={30} lines={["Freeing memory is not erasing it."]} />
        </>
      )}
    </Diagram>
  );
}
