import { simulateMemory } from "@/lib/simulation/memory";
import { SHOW_PRIME, openSession, ownTextModel, runStep, seededRng } from "@/lib/simulation/show";

/** What a diagram shows for the reader's input: the value on the move at each step, and what each part holds. */
export interface DiagramData {
  values: string[];
  parts: Array<{ id: string; name: string; holds: string; sees: string; cannot: string }>;
}

function seedOf(text: string): number {
  let seed = 17;
  for (let index = 0; index < text.length; index += 1) seed = (Math.imul(seed, 31) + text.charCodeAt(index)) >>> 0;
  return seed;
}

/** One real protected step on the reader's text: the first character the toy model works on. */
function firstStep(text: string) {
  const model = ownTextModel(text.trim().length > 0 ? text : "hi");
  const session = openSession(model, seededRng(seedOf(model.prompt)));
  const step = runStep(session, 0);
  const at = model.units - 1;
  const letter = model.prompt[at]!;
  const code = letter.charCodeAt(0);
  const pieces = [step.activationShares[0][at]!, step.activationShares[1][at]!];
  const products = step.roles.map((role) => role.product[0]!);
  return { model, letter, code, pieces, products, out: model.decodeToken(step.combined!), passed: step.roles.filter((role) => role.valid).length };
}

const WORKER_IDS = ["w00", "w01", "w10", "w11"];

export function architectureData(text: string): DiagramData {
  const run = firstStep(text);
  return {
    values: [
      `“${run.model.prompt}”`,
      "who is online?",
      `${run.pieces[0]} + ${run.pieces[1]}`,
      `${run.pieces[0]} and ${run.pieces[1]}`,
      run.products.join(", "),
      `${run.passed} of 4 pass`,
      `“${run.out}”`,
    ],
    parts: [
      { id: "prompt", name: "Your prompt", holds: `Your text “${run.model.prompt}”.`, sees: "Everything: it is your own device.", cannot: "Nothing is hidden from you." },
      { id: "split", name: "Split", holds: `The letter “${run.letter}” as the number ${run.code}, cut into ${run.pieces[0]} and ${run.pieces[1]}. They add up to ${run.code} (counting in a circle of ${SHOW_PRIME}).`, sees: "Both pieces, because this runs on your device.", cannot: "Nothing: this is still your device." },
      { id: "directory", name: "Controller", holds: "The list of workers that are online.", sees: "Who is connected and how big each encrypted message is.", cannot: `Your text, the number ${run.code}, or either piece.` },
      ...WORKER_IDS.map((id, role) => ({
        id,
        name: `Worker (${role >> 1},${role & 1})`,
        holds: `Piece ${role >> 1} of your data, ${run.pieces[role >> 1]}, and piece ${role & 1} of the model.`,
        sees: `Its own answer, ${run.products[role]}. A random-looking number.`,
        cannot: `The letter “${run.letter}”, the other piece ${run.pieces[1 - (role >> 1)]}, or the result.`,
      })),
      { id: "verify", name: "Verify", holds: `The four answers: ${run.products.join(", ")}.`, sees: `Whether each one passes its check: ${run.passed} of 4 do.`, cannot: "Nothing: this is your device." },
      { id: "combine", name: "Combine", holds: `The four checked answers added together: the number ${run.out.charCodeAt(0)}.`, sees: `The result, “${run.out}”.`, cannot: "Nothing: the readable result exists only here." },
    ],
  };
}

export function runtimeData(text: string): DiagramData {
  const run = firstStep(text);
  return {
    values: [
      `“${run.model.prompt}”`,
      `${run.pieces[0]} + ${run.pieces[1]}`,
      run.products.join(", "),
      `${run.passed} of 4 pass → “${run.out}”`,
      `“${run.out}” goes round again`,
      `“${run.out}”`,
      `“${run.out}” is appended`,
    ],
    parts: [
      { id: "prompt", name: "Prompt", holds: `Your text “${run.model.prompt}”.`, sees: "Everything: it is your own device.", cannot: "Nothing is hidden from you." },
      { id: "split", name: "Split", holds: `The number ${run.code} for “${run.letter}”, cut into ${run.pieces[0]} and ${run.pieces[1]}.`, sees: "Both pieces, on your device.", cannot: "Nothing: this is still your device." },
      { id: "workers", name: "Workers", holds: `One piece each: ${run.pieces[0]} or ${run.pieces[1]}, never both.`, sees: `Their own answers: ${run.products.join(", ")}.`, cannot: `The letter “${run.letter}” or the result “${run.out}”.` },
      { id: "combine", name: "Verify + combine", holds: "The four answers and the checks for them.", sees: `${run.passed} of 4 pass, and they add up to “${run.out}”.`, cannot: "Nothing: this is your device." },
      { id: "head", name: "Head and sampling", holds: `The finished result “${run.out}”.`, sees: "The output, which never leaves your device.", cannot: "Nothing: this is your device." },
    ],
  };
}

const hex = (bytes: number[]) => bytes.map((value) => value.toString(16).padStart(2, "0")).join(" ");

export function parseHexBytes(text: string): number[] | null {
  const parts = text.trim().split(/\s+/);
  return parts.length === 4 && parts.every((part) => /^[0-9a-fA-F]{1,2}$/.test(part)) ? parts.map((part) => parseInt(part, 16)) : null;
}

const STAGE_NAMES = ["Allocate", "Execute", "Release", "Synchronise", "Zero", "Reuse"];

export function memoryData(text: string): DiagramData {
  const bytes = parseHexBytes(text) ?? [0xde, 0xad, 0xbe, 0xef];
  const safe = simulateMemory(true, bytes);
  const skipped = simulateMemory(false, bytes);
  return {
    values: [
      "empty buffer",
      ...safe.slice(1).map((stage) => hex(stage.bytes)),
      `${hex(skipped[skipped.length - 1]!.bytes)} read by the next owner`,
    ],
    parts: safe.map((stage, index) => ({
      id: `s${index}`,
      name: STAGE_NAMES[index]!,
      holds: index === 0 ? "A buffer with nothing of yours in it yet." : `The bytes ${hex(stage.bytes)}.`,
      sees: index === 5 ? `The next owner reads ${hex(stage.bytes)}.` : stage.note + ".",
      cannot: index >= 4 ? `Your bytes ${hex(bytes)}: they have been overwritten.` : "Nothing yet: your bytes are still in the buffer.",
    })),
  };
}
