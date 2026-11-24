"use client";

import { Arrow, Box, Diagram, Note, Zone, type DiagramStep } from "@/components/diagrams/primitives";

const STEPS: DiagramStep[] = [
  { title: "Prompt", text: "The prompt is tokenised and embedded on the client. Nothing has left the device yet.", show: ["prompt"] },
  { title: "Split", text: "The hidden state of the current layer is split into shares.", show: ["split", "to-split"] },
  { title: "Workers", text: "Workers apply the layer's linear maps to the shares they hold. Each sees one share, encrypted in transit.", show: ["workers", "to-workers"] },
  { title: "Verify and combine", text: "The client verifies the results, adds them up and applies the nonlinear step itself.", show: ["combine", "to-combine"] },
  { title: "Next layer", text: "The combined state becomes the input of the next layer, and the same three steps repeat.", show: ["loop-layer", "split", "combine"] },
  { title: "Head and sampling", text: "After the last layer the language-model head and the sampling run locally, so the output never leaves the client.", show: ["head", "to-head"] },
  { title: "Next token", text: "The sampled token is appended to the prompt and the whole loop runs again.", show: ["loop-token", "prompt", "head"] },
];

export function RuntimeFlowDiagram({ caption }: { caption: string }) {
  return (
    <Diagram
      title="One token through the inference runtime"
      description="The prompt is tokenised and embedded on the client. For every model layer the client splits the hidden state into shares, the workers apply the linear maps to their shares, and the client verifies, combines and applies the nonlinear step. After the last layer the client runs the language-model head and samples the next token locally, then repeats."
      width={960}
      height={330}
      caption={caption}
      steps={STEPS}
    >
      {(arrowId) => (
        <>
          <Box id="prompt" x={10} y={120} width={140} height={76} lines={["Prompt", "tokenise + embed", "on the client"]} tone="good" />
          <Zone x={186} y={50} width={518} height={244} label="Repeated for every layer" tone="accent" />
          <Box id="split" x={206} y={120} width={140} height={76} lines={["Split", "hidden state", "into shares"]} tone="good" />
          <Box id="workers" x={372} y={120} width={140} height={76} lines={["Workers", "linear maps on", "shares only"]} tone="accent" />
          <Box id="combine" x={538} y={120} width={150} height={76} lines={["Verify + combine", "nonlinearity", "on the client"]} tone="good" />
          <Box id="head" x={786} y={120} width={166} height={76} lines={["LM head + sampling", "local; output never", "leaves the client"]} tone="good" />
          <Arrow id="to-split" from="prompt" to="split" arrowId={arrowId} d="M150 158 H206" />
          <Arrow id="to-workers" from="split" to="workers" arrowId={arrowId} d="M346 158 H372" tone="accent" />
          <Arrow id="to-combine" from="workers" to="combine" arrowId={arrowId} d="M512 158 H538" />
          <Arrow id="to-head" from="combine" to="head" arrowId={arrowId} d="M688 158 H786" label="last layer" labelX={745} labelY={150} />
          <Arrow id="loop-layer" from="combine" to="split" arrowId={arrowId} d="M613 196 C613 262 276 262 276 196" dashed label="next layer" labelX={444} labelY={240} />
          <Arrow id="loop-token" from="head" to="prompt" arrowId={arrowId} d="M869 120 C869 6 80 6 80 120" dashed label="append the token and continue" labelX={474} labelY={24} />
          <Note x={206} y={282} lines={["Encrypted in transit; each worker sees one share."]} />
        </>
      )}
    </Diagram>
  );
}
