import { Arrow, Box, Diagram, Note, Zone } from "@/components/diagrams/primitives";

export function RuntimeFlowDiagram({ caption }: { caption: string }) {
  return (
    <Diagram
      title="One token through the inference runtime"
      description="The prompt is tokenised and embedded on the client. For every model layer the client splits the hidden state into shares, the workers apply the linear maps to their shares, and the client verifies, combines and applies the nonlinear step. After the last layer the client runs the language-model head and samples the next token locally, then repeats."
      width={960}
      height={330}
      caption={caption}
    >
      {(arrowId) => (
        <>
          <Box x={16} y={120} width={150} height={76} lines={["Prompt", "tokenise + embed", "on the client"]} tone="good" />
          <Zone x={196} y={36} width={540} height={258} label="Repeated for every layer" tone="accent" />
          <Box x={220} y={120} width={150} height={76} lines={["Split", "hidden state", "into shares"]} tone="good" />
          <Box x={400} y={120} width={150} height={76} lines={["Workers", "linear maps on", "shares only"]} tone="accent" />
          <Box x={580} y={120} width={136} height={76} lines={["Verify + combine", "nonlinearity", "on the client"]} tone="good" />
          <Box x={786} y={120} width={158} height={76} lines={["LM head + sampling", "local; output never", "leaves the client"]} tone="good" />
          <Arrow arrowId={arrowId} d="M166 158 H220" />
          <Arrow arrowId={arrowId} d="M370 158 H400" tone="accent" />
          <Arrow arrowId={arrowId} d="M550 158 H580" />
          <Arrow arrowId={arrowId} d="M716 158 H786" label="last layer" labelX={751} labelY={150} />
          <Arrow arrowId={arrowId} d="M648 196 C648 262 295 262 295 196" dashed label="next layer" labelX={470} labelY={240} />
          <Arrow arrowId={arrowId} d="M865 120 C865 14 91 14 91 120" dashed label="append the token and continue" labelX={478} labelY={26} />
          <Note x={220} y={284} lines={["Encrypted in transit; each worker sees one share."]} />
        </>
      )}
    </Diagram>
  );
}
