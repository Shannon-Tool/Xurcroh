import { simulateMemory } from "@/lib/simulation/memory";
import { SHOW_PRIME } from "@/lib/simulation/show";
import { journeyOf } from "@/lib/show/journey";

/** What a diagram shows for the chosen question: the value on the move at each step, what each part holds, and the answer. */
export interface DiagramData {
  question: string;
  answer: string;
  values: string[];
  parts: Array<{ id: string; name: string; holds: string; sees: string; cannot: string }>;
}

/** The numbers of one question's real trip that the diagrams quote. */
function trip(id: string) {
  const journey = journeyOf(id);
  const word = journey.words[0]!;
  const at = journey.vocabulary.indexOf(word);
  const row = journey.scores.indexOf(Math.max(...journey.scores));
  return {
    journey,
    word,
    // The first word of the question is a 1 in the list of word counts; these are its two random pieces.
    pieces: [journey.pieces[0][at]!, journey.pieces[1][at]!],
    // Each worker's share of the winning answer's score; the four add up to that score.
    products: journey.products.map((product) => product[row]!),
    score: journey.scores[row]!,
    passed: journey.valid.filter(Boolean).length,
    opening: `${journey.tokens.slice(0, 4).join("").trim()}…`,
  };
}

const WORKER_IDS = ["w00", "w01", "w10", "w11"];

export function architectureData(id: string): DiagramData {
  const run = trip(id);
  const { journey } = run;
  return {
    question: journey.question,
    answer: journey.answer,
    values: [
      `“${journey.question}”`,
      "who is online?",
      `${run.pieces[0]} + ${run.pieces[1]}`,
      `${run.pieces[0]} and ${run.pieces[1]}`,
      run.products.join(", "),
      `${run.passed} of 4 pass`,
      `score ${run.score} → “${run.opening}”`,
    ],
    parts: [
      { id: "prompt", name: "Your question", holds: `Your question, “${journey.question}”, and the list of ${journey.counts.length} word counts made from it.`, sees: "Everything: it is your own device.", cannot: "Nothing is hidden from you." },
      { id: "split", name: "Split", holds: `Every word count cut in two. The 1 that stands for “${run.word}” became ${run.pieces[0]} and ${run.pieces[1]}, which add up to 1 when you count in a circle of ${SHOW_PRIME}.`, sees: "Both pieces, because this runs on your device.", cannot: "Nothing: this is still your device." },
      { id: "directory", name: "Controller", holds: "The list of workers that are online.", sees: "Who is connected and how big each encrypted message is.", cannot: "Your question, its word counts, or either piece." },
      ...WORKER_IDS.map((part, role) => ({
        id: part,
        name: `Worker (${role >> 1},${role & 1})`,
        holds: `Piece ${role >> 1} of your question (for “${run.word}”: ${run.pieces[role >> 1]}) and piece ${role & 1} of the model.`,
        sees: `Its own result, ${run.products[role]}. A random-looking number.`,
        cannot: `Your words, the other piece (${run.pieces[1 - (role >> 1)]}), the score, or the answer.`,
      })),
      { id: "verify", name: "Verify", holds: `The four workers' results: ${run.products.join(", ")}.`, sees: `Whether each one passes its check: ${run.passed} of 4 do.`, cannot: "Nothing: this is your device." },
      { id: "combine", name: "Combine", holds: `The four checked results added together: ${run.score}, the highest score of the ${journey.scores.length} prepared answers.`, sees: `The answer that score selects: “${run.opening}”`, cannot: "Nothing: the readable answer exists only here." },
    ],
  };
}

export function runtimeData(id: string): DiagramData {
  const run = trip(id);
  const { journey } = run;
  return {
    question: journey.question,
    answer: journey.answer,
    values: [
      `“${journey.question}”`,
      `${run.pieces[0]} + ${run.pieces[1]}`,
      run.products.join(", "),
      `${run.passed} of 4 pass → score ${run.score}`,
      "the result goes round again",
      `“${journey.tokens[0]!.trim()}”`,
      `“${journey.tokens.slice(0, 3).join("").trim()}” and so on`,
    ],
    parts: [
      { id: "prompt", name: "Prompt", holds: `Your question, “${journey.question}”, as ${journey.counts.length} word counts.`, sees: "Everything: it is your own device.", cannot: "Nothing is hidden from you." },
      { id: "split", name: "Split", holds: `Each number cut in two. The 1 for “${run.word}” became ${run.pieces[0]} and ${run.pieces[1]}.`, sees: "Both pieces, on your device.", cannot: "Nothing: this is still your device." },
      { id: "workers", name: "Workers", holds: `One piece each, ${run.pieces[0]} or ${run.pieces[1]}, never both.`, sees: `Their own results: ${run.products.join(", ")}.`, cannot: `Your words or the answer “${run.opening}”` },
      { id: "combine", name: "Verify + combine", holds: "The four results and the checks for them.", sees: `${run.passed} of 4 pass, and they add up to ${run.score}.`, cannot: "Nothing: this is your device." },
      { id: "head", name: "Head and sampling", holds: `The answer as it forms, one piece at a time: “${journey.tokens.slice(0, 3).join("").trim()}…”`, sees: "The output, which never leaves your device.", cannot: "Nothing: this is your device." },
    ],
  };
}

const hex = (bytes: number[]) => bytes.map((value) => value.toString(16).padStart(2, "0")).join(" ");
const STAGE_NAMES = ["Allocate", "Execute", "Release", "Synchronise", "Zero", "Reuse"];

export function memoryData(id: string): DiagramData {
  const journey = journeyOf(id);
  const start = journey.question.slice(0, 4);
  const safe = simulateMemory(true, journey.bytes);
  const skipped = simulateMemory(false, journey.bytes);
  return {
    question: journey.question,
    answer: journey.answer,
    values: [
      "empty buffer",
      ...safe.slice(1).map((stage) => hex(stage.bytes)),
      `${hex(skipped[skipped.length - 1]!.bytes)} read by the next owner`,
    ],
    parts: safe.map((stage, index) => ({
      id: `s${index}`,
      name: STAGE_NAMES[index]!,
      holds: index === 0 ? "A buffer with nothing of yours in it yet." : `The bytes ${hex(stage.bytes)}${index < 4 ? `: “${start}”, the start of your question` : ""}.`,
      sees: index === 5 ? `The next owner reads ${hex(stage.bytes)}.` : `${stage.note}.`,
      cannot: index >= 4 ? `The start of your question (${hex(journey.bytes)}): it has been overwritten.` : "Nothing yet: your question is still in the buffer.",
    })),
  };
}
