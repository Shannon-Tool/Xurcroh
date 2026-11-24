import { SHOW_MODELS, getMatcher, getModel, openSession, runStep, seededRng, type PixelModel, type TextModel } from "@/lib/simulation/show";

/**
 * One prepared question followed from start to finish: the numbers it becomes, the two random pieces,
 * what each of the four workers returns, the checks, the scores that pick the answer, and the answer.
 * Everything is computed by the same share engine the live demonstration uses; nothing is made up here.
 */
export interface Journey {
  id: string;
  question: string;
  answer: string;
  /** The answer as the pieces the model produces one at a time. */
  tokens: string[];
  /** The question's words, lower-cased. */
  words: string[];
  vocabulary: string[];
  /** One count per word the model knows. */
  counts: number[];
  /** The two random pieces of `counts`. */
  pieces: [number[], number[]];
  /** What each of the four workers returns: one number per prepared answer. */
  products: number[][];
  valid: boolean[];
  /** The four products added up: one score per prepared answer. */
  scores: number[];
  /** Id of the answer with the highest score. */
  picked: string;
  /** The matching table as plain rows, for the laboratories' big-number arithmetic. */
  weightRows: number[][];
  /** The question's letters as positions in the alphabet (a = 1). */
  letters: number[];
  /** Its first four characters as byte values. */
  bytes: number[];
  /** Four small codes (0–15) taken from the answer's letters, standing in for stored weights. */
  codes: number[];
  /** A stable number for this question, used where a lab needs a shuffle. */
  hash: number;
}

const DEFAULT_QUESTION = "text-shares";
const DEFAULT_PICTURE = "picture-sunrise";
const cache = new Map<string, Journey>();

function hashOf(text: string): number {
  let value = 17;
  for (let index = 0; index < text.length; index += 1) value = (Math.imul(value, 31) + text.charCodeAt(index)) >>> 0;
  return value;
}

function lettersOf(text: string): number[] {
  return (text.toLowerCase().match(/[a-z]/g) ?? ["a"]).map((letter) => letter.charCodeAt(0) - 96);
}

export function questionIds(): string[] {
  return SHOW_MODELS.filter((entry) => entry.kind === "text").map((entry) => entry.id);
}

export function pictureIds(): string[] {
  return SHOW_MODELS.filter((entry) => entry.kind === "picture").map((entry) => entry.id);
}

export function journeyOf(requested: string): Journey {
  const id = questionIds().includes(requested) ? requested : DEFAULT_QUESTION;
  const cached = cache.get(id);
  if (cached) return cached;
  const model = getModel(id) as TextModel;
  const matcher = getMatcher();
  const row = matcher.questions.indexOf(id);
  const hash = hashOf(id);
  const step = runStep(openSession(matcher, seededRng(hash)), row);
  const { rows, cols, data } = matcher.weight;
  const journey: Journey = {
    id,
    question: model.prompt,
    answer: model.answer,
    tokens: model.tokens,
    words: matcher.wordsOf(id),
    vocabulary: matcher.vocabulary,
    counts: Array.from(matcher.input(row)),
    pieces: [Array.from(step.activationShares[0]), Array.from(step.activationShares[1])],
    products: step.roles.map((role) => Array.from(role.product)),
    valid: step.roles.map((role) => role.valid),
    scores: Array.from(step.combined ?? []),
    picked: step.combined ? matcher.pick(step.combined) : id,
    weightRows: Array.from({ length: rows }, (_, index) => Array.from(data.subarray(index * cols, (index + 1) * cols))),
    letters: lettersOf(model.prompt),
    bytes: Array.from({ length: 4 }, (_, index) => model.prompt.charCodeAt(index % model.prompt.length) & 255),
    codes: lettersOf(model.answer).slice(0, 4).map((letter) => letter % 16),
    hash,
  };
  cache.set(id, journey);
  return journey;
}

export interface PictureJourney {
  id: string;
  request: string;
  width: number;
  height: number;
  /** The finished picture as RGB bytes. */
  picture: Uint8ClampedArray;
}

export function pictureJourneyOf(requested: string): PictureJourney {
  const id = pictureIds().includes(requested) ? requested : DEFAULT_PICTURE;
  const model = getModel(id) as PixelModel;
  return { id, request: model.prompt, width: model.width, height: model.height, picture: model.frames[0]! };
}
