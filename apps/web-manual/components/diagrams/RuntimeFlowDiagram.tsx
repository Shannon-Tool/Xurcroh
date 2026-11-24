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
          <Box x={10} y={120} width={140} height={76} lines={["Prompt", "tokenise + embed", "on the client"]} tone="good" />
          <Zone x={186} y={50} width={518} height={244} label="Repeated for every layer" tone="accent" />
          <Box x={206} y={120} width={140} height={76} lines={["Split", "hidden state", "into shares"]} tone="good" />
          <Box x={372} y={120} width={140} height={76} lines={["Workers", "linear maps on", "shares only"]} tone="accent" />
          <Box x={538} y={120} width={150} height={76} lines={["Verify + combine", "nonlinearity", "on the client"]} tone="good" />
          <Box x={786} y={120} width={166} height={76} lines={["LM head + sampling", "local; output never", "leaves the client"]} tone="good" />
          <Arrow arrowId={arrowId} d="M150 158 H206" />
          <Arrow arrowId={arrowId} d="M346 158 H372" tone="accent" />
          <Arrow arrowId={arrowId} d="M512 158 H538" />
          <Arrow arrowId={arrowId} d="M688 158 H786" label="last layer" labelX={745} labelY={150} />
          <Arrow arrowId={arrowId} d="M613 196 C613 262 276 262 276 196" dashed label="next layer" labelX={444} labelY={240} />
          <Arrow arrowId={arrowId} d="M869 120 C869 6 80 6 80 120" dashed label="append the token and continue" labelX={474} labelY={24} />
          <Note x={206} y={282} lines={["Encrypted in transit; each worker sees one share."]} />
        </>
      )}
    </Diagram>
  );
}
