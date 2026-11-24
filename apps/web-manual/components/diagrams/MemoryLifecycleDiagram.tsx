import { Arrow, Box, Diagram, Note } from "@/components/diagrams/primitives";

const STEPS = [
  ["Allocate", "buffer handed out"],
  ["Execute", "secret data written"],
  ["Release", "allocator marks free"],
  ["Synchronise", "wait for the stream"],
  ["Zero", "overwrite the bytes"],
  ["Reuse", "next owner reads zeros"],
] as const;

export function MemoryLifecycleDiagram({ caption }: { caption: string }) {
  return (
    <Diagram
      title="GPU memory lifecycle"
      description="A buffer is allocated, written with secret data, released, then the stream is synchronised and the bytes are zeroed before the buffer is reused. If synchronisation is skipped the wipe may still be queued when the buffer is handed to the next owner, who can then read stale secret bytes."
      width={960}
      height={270}
      caption={caption}
    >
      {(arrowId) => (
        <>
          {STEPS.map(([name, detail], index) => (
            <Box key={name} x={16 + index * 158} y={60} width={138} height={70} lines={[name, detail]} tone={index === 3 || index === 4 ? "good" : "plain"} />
          ))}
          {STEPS.slice(0, -1).map(([name], index) => (
            <Arrow key={name} arrowId={arrowId} d={`M${154 + index * 158} 95 H${174 + index * 158}`} />
          ))}
          <Arrow arrowId={arrowId} d="M401 130 C401 226 875 226 875 130" tone="warn" dashed label="synchronisation skipped: the wipe is still queued" labelX={638} labelY={196} />
          <Note x={420} y={246} lines={["The next owner can read stale secret bytes."]} />
          <Note x={16} y={30} lines={["Freeing memory is not erasing it."]} />
        </>
      )}
    </Diagram>
  );
}
