import { Arrow, Box, Diagram, Note, Zone } from "@/components/diagrams/primitives";

const WORKERS = [
  { x: 420, y: 250, role: "(0,0)" },
  { x: 690, y: 250, role: "(0,1)" },
  { x: 420, y: 380, role: "(1,0)" },
  { x: 690, y: 380, role: "(1,1)" },
] as const;

export function ArchitectureDiagram({ caption }: { caption: string }) {
  return (
    <Diagram
      title="Fortress system architecture"
      description="The client splits each activation into additive shares and sends one share to each of four role workers run by independent operators. Each worker holds one weight share, returns an exact product with a proof, and never sees the prompt, the full weights or the output. The client verifies every result before combining it. The controller only tells the client which workers are online and sees ciphertext."
      width={960}
      height={520}
      caption={caption}
    >
      {(arrowId) => (
        <>
          <Zone x={16} y={16} width={300} height={488} label="Your device: plaintext stays here" tone="good" />
          <Box x={40} y={60} width={252} height={64} lines={["Prompt, tokens, embeddings", "kept local"]} tone="good" />
          <Box x={40} y={164} width={252} height={64} lines={["Split into additive shares", "share 0 + share 1 = activation"]} tone="good" />
          <Box x={40} y={298} width={252} height={64} lines={["Verify every worker result", "reject on any mismatch"]} tone="good" />
          <Box x={40} y={412} width={252} height={64} lines={["Combine, apply nonlinearity,", "sample the next token"]} tone="good" />
          <Arrow arrowId={arrowId} d="M166 124 V164" />
          <Arrow arrowId={arrowId} d="M166 362 V412" />
          <Arrow arrowId={arrowId} d="M60 412 C20 380 20 230 40 205" dashed label="next layer" labelX={62} labelY={268} />

          <Zone x={380} y={16} width={564} height={150} label="Controller: a directory that sees only ciphertext" />
          <Box x={520} y={64} width={284} height={72} lines={["Signed worker directory", "sees who is online and ciphertext sizes"]} />
          <Arrow arrowId={arrowId} d="M292 84 H520" both dashed label="look up workers" labelX={406} labelY={76} />

          <Zone x={380} y={196} width={564} height={308} label="Role workers: four independent operators" tone="accent" />
          {WORKERS.map((worker) => (
            <Box key={worker.role} x={worker.x} y={worker.y} width={230} height={92} tone="accent" lines={[`Worker ${worker.role}`, `activation share ${worker.role[1]} × weight share ${worker.role[3]}`.replace(" × ", " · "), "exact integer product"]} />
          ))}
          <Arrow arrowId={arrowId} d="M292 190 C340 190 336 290 380 290" tone="accent" label="one encrypted share" labelX={348} labelY={218} />
          <Arrow arrowId={arrowId} d="M380 410 C336 410 340 334 292 334" label="products + proofs" labelX={348} labelY={392} />
          <Note x={420} y={490} lines={["No worker holds two complementary shares; hiding needs non-collusion."]} />
        </>
      )}
    </Diagram>
  );
}
