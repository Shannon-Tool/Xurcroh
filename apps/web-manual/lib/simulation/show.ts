/**
 * The arithmetic behind the live demonstration.
 *
 * What is real: every output (a token, a picture step, a clip frame) is one matrix product W·x that
 * is computed on additive shares by four roles, checked with a Freivalds test and recombined.
 * What is prepared: the "models". Text uses a lookup table holding one prepared answer; pictures and
 * the clip use a small hand-built linear decoder driven along a prepared path from noise to the image.
 *
 * Arithmetic is modulo a 16-bit prime so that products stay exact in JavaScript numbers at picture
 * size. The engine itself works in a larger ring; see `fortress.ts` for the big-integer version.
 */

export const SHOW_PRIME = 65521;
const HALF = (SHOW_PRIME - 1) / 2;

/** Returns a uniform integer in [0, SHOW_PRIME). */
export type Rng = () => number;

export interface Matrix {
  rows: number;
  cols: number;
  /** Row-major, every entry in [0, SHOW_PRIME). */
  data: Int32Array;
}

/** Repeatable randomness for tests and for building the toy models. */
export function seededRng(seed: number): Rng {
  let state = seed >>> 0;
  return () => {
    for (;;) {
      state = (state + 0x6d2b79f5) >>> 0;
      let t = state;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      const value = ((t ^ (t >>> 14)) >>> 0) >>> 16;
      if (value < SHOW_PRIME) return value;
    }
  };
}

/** Fresh randomness from the browser, used when the show runs for a reader. */
export function cryptoRng(): Rng {
  const buffer = new Uint16Array(4096);
  let next = buffer.length;
  return () => {
    for (;;) {
      if (next >= buffer.length) {
        crypto.getRandomValues(buffer);
        next = 0;
      }
      const value = buffer[next++]!;
      if (value < SHOW_PRIME) return value;
    }
  };
}

function toRing(value: number): number {
  const remainder = value % SHOW_PRIME;
  return remainder < 0 ? remainder + SHOW_PRIME : remainder;
}

function splitValues(values: Int32Array, rng: Rng): [Int32Array, Int32Array] {
  const first = new Int32Array(values.length);
  const second = new Int32Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    const mask = rng();
    first[index] = mask;
    second[index] = toRing(values[index]! - mask);
  }
  return [first, second];
}

/** Two additive shares: the first is uniformly random, the second is the secret minus the first. */
export function splitVector(values: Int32Array, rng: Rng): [Int32Array, Int32Array] {
  return splitValues(values, rng);
}

export function splitMatrix(matrix: Matrix, rng: Rng): [Matrix, Matrix] {
  const [first, second] = splitValues(matrix.data, rng);
  return [{ rows: matrix.rows, cols: matrix.cols, data: first }, { rows: matrix.rows, cols: matrix.cols, data: second }];
}

export function combine(parts: Int32Array[]): Int32Array {
  const out = new Int32Array(parts[0]!.length);
  for (let index = 0; index < out.length; index += 1) {
    let sum = 0;
    for (const part of parts) sum += part[index]!;
    out[index] = sum % SHOW_PRIME;
  }
  return out;
}

/** Exact W·x modulo the prime. Each term is below 2^32, so a row of up to a million terms stays exact. */
export function matVec(weight: Matrix, vector: Int32Array): Int32Array {
  const { rows, cols, data } = weight;
  const out = new Int32Array(rows);
  for (let row = 0; row < rows; row += 1) {
    const base = row * cols;
    let sum = 0;
    for (let col = 0; col < cols; col += 1) sum += data[base + col]! * vector[col]!;
    out[row] = sum % SHOW_PRIME;
  }
  return out;
}

/** challengeᵀ·W, the part of a Freivalds test that depends only on the weights. */
function foldWeights(weight: Matrix, challenge: Int32Array): Int32Array {
  const { rows, cols, data } = weight;
  const sums = new Float64Array(cols);
  for (let row = 0; row < rows; row += 1) {
    const base = row * cols;
    const factor = challenge[row]!;
    for (let col = 0; col < cols; col += 1) sums[col]! += factor * data[base + col]!;
  }
  return Int32Array.from(sums, (value) => value % SHOW_PRIME);
}

function dot(a: Int32Array, b: Int32Array): number {
  let sum = 0;
  for (let index = 0; index < a.length; index += 1) sum += a[index]! * b[index]!;
  return sum % SHOW_PRIME;
}

/** Freivalds test: accepts when challengeᵀ·(W·x) equals challengeᵀ·product. */
export function freivalds(weight: Matrix, vector: Int32Array, product: Int32Array, challenge: Int32Array): boolean {
  if (product.length !== weight.rows || challenge.length !== weight.rows || vector.length !== weight.cols) return false;
  return dot(foldWeights(weight, challenge), vector) === dot(challenge, product);
}

// ---------------------------------------------------------------- models

export type ShowKind = "text" | "picture" | "clip";

/** Anything one protected matrix product can be run on. */
export interface Computable {
  weight: Matrix;
  units: number;
  input(unit: number): Int32Array;
}

interface BaseModel extends Computable {
  id: string;
  kind: ShowKind;
  /** The request shown to the reader. */
  prompt: string;
  /** What the toy model does, when the input is the reader's own. */
  toy?: string;
}

export interface TextModel extends BaseModel {
  kind: "text";
  answer: string;
  tokens: string[];
  vocabulary: string[];
  decodeToken(combined: Int32Array): string;
}

export interface PixelModel extends BaseModel {
  kind: "picture" | "clip";
  width: number;
  height: number;
  /** Finished frames as RGB bytes: one for a picture, several for a clip. */
  frames: Uint8ClampedArray[];
  stepsPerFrame: number;
  /** Fixed-point scale of the decoder output. */
  scale: number;
  /** Distance, in ring units, between the largest possible intermediate value and wrap-around. */
  headroom: number;
  /** True when each unit yields one part of the picture and the device adds the parts up. */
  accumulate?: boolean;
}

export type ShowModel = TextModel | PixelModel;

export interface ShowModelInfo {
  id: string;
  kind: ShowKind;
  title: string;
}

const TEXTS: Array<{ id: string; prompt: string; answer: string }> = [
  {
    id: "text-shares",
    prompt: "How is my prompt kept from the workers?",
    answer: "Your device splits every number into two random shares. Each worker gets one share of your data and one share of the weights, and a single share is uniformly random on its own. Your device checks the four products and adds them to get the result. This holds as long as the workers do not pool their shares.",
  },
  {
    id: "text-tamper",
    prompt: "What if a worker returns a wrong answer?",
    answer: "Your device tests every returned product against a random challenge. A product that is wrong by even one fails the test, except with a tiny probability. The step is then rejected and the work is sent again. Press Tamper with a worker to watch that happen here.",
  },
  {
    id: "text-worker-view",
    prompt: "What does a worker actually see?",
    answer: "A worker sees one random piece of your data, one random piece of the model, and the answer it computes from them. All three look like noise. It never sees your question, the other pieces, or the final result.",
  },
  {
    id: "text-operators",
    prompt: "Who runs the workers?",
    answer: "The design calls for four workers run by independent operators, so that no single operator holds two pieces that fit together. In this manual the four workers are four roles inside the page. Real operators and hardware checks are still to come.",
  },
  {
    id: "text-memory",
    prompt: "What happens to my data in memory afterwards?",
    answer: "The memory your request used is wiped, and the wipe is confirmed finished, before that memory is handed to anyone else. A software wipe cannot prove that every register inside the chip is cleared, and the manual says so.",
  },
  {
    id: "text-limits",
    prompt: "What does this page not show?",
    answer: "This page shows only the share arithmetic, with toy models and a prepared answer. Encryption in transit, anonymous routing, attestation and memory wiping are separate layers with their own laboratories. The design is built for defence in depth, and its claims are still under evaluation.",
  },
];

export const SHOW_MODELS: ShowModelInfo[] = [
  ...TEXTS.map((entry) => ({ id: entry.id, kind: "text" as const, title: entry.prompt })),
  { id: "picture-sunrise", kind: "picture", title: "Sunrise over the water" },
  { id: "picture-city", kind: "picture", title: "City at night" },
  { id: "clip-sunrise", kind: "clip", title: "Sunrise, twelve frames" },
];

function hash(a: number, b: number, seed = 0): number {
  let value = Math.imul(a + 1, 0x9e3779b1) ^ Math.imul(b + 1, 0x85ebca6b) ^ Math.imul(seed + 1, 0xc2b2ae35);
  value ^= value >>> 16;
  value = Math.imul(value, 0x7feb352d);
  value ^= value >>> 15;
  return (value >>> 0) / 4294967296;
}

function buildText(entry: { id: string; prompt: string; answer: string }): TextModel {
  const tokens = entry.answer.match(/\s*\S+/g) ?? [];
  const vocabulary = [...new Set(tokens)].sort();
  const rows = vocabulary.length;
  const cols = tokens.length;
  const data = new Int32Array(rows * cols);
  for (let col = 0; col < cols; col += 1) {
    const right = vocabulary.indexOf(tokens[col]!);
    for (let row = 0; row < rows; row += 1) data[row * cols + col] = row === right ? 900 : Math.floor(hash(row, col, 7) * 400);
  }
  return {
    id: entry.id,
    kind: "text",
    prompt: entry.prompt,
    answer: entry.answer,
    tokens,
    vocabulary,
    weight: { rows, cols, data },
    units: cols,
    input(unit) {
      const vector = new Int32Array(cols);
      vector[unit] = 1;
      return vector;
    },
    decodeToken(combined) {
      let best = 0;
      for (let row = 1; row < rows; row += 1) if (combined[row]! > combined[best]!) best = row;
      return vocabulary[best]!;
    },
  };
}

// ---- reading the question: a matching table from words to prepared answers

export interface Matcher extends Computable {
  /** Ids of the prepared questions, in row order. */
  questions: string[];
  vocabulary: string[];
  wordsOf(id: string): string[];
  /** The id whose row scored highest. */
  pick(combined: Int32Array): string;
}

function wordsIn(text: string): string[] {
  return text.toLowerCase().match(/[a-z]+/g) ?? [];
}

let matcher: Matcher | null = null;

/**
 * The first step of answering: the question becomes a list of word counts, and one matrix product scores
 * it against every prepared question. Run on shares like any other step; the highest score picks the answer.
 */
export function getMatcher(): Matcher {
  if (matcher) return matcher;
  const words = TEXTS.map((entry) => wordsIn(entry.prompt));
  const vocabulary = [...new Set(words.flat())].sort();
  const rows = TEXTS.length;
  const cols = vocabulary.length;
  const data = new Int32Array(rows * cols);
  words.forEach((list, row) => {
    const distinct = new Set(list);
    // Every question's own words add up to about the same total, so a short question is not outvoted.
    for (const word of distinct) data[row * cols + vocabulary.indexOf(word)] = Math.floor(600 / distinct.size);
  });
  matcher = {
    questions: TEXTS.map((entry) => entry.id),
    vocabulary,
    weight: { rows, cols, data },
    units: rows,
    wordsOf: (id) => words[Math.max(0, TEXTS.findIndex((entry) => entry.id === id))]!,
    input(unit) {
      const vector = new Int32Array(cols);
      for (const word of words[unit]!) vector[vocabulary.indexOf(word)]! += 1;
      return vector;
    },
    pick(combined) {
      let best = 0;
      for (let row = 1; row < rows; row += 1) if (combined[row]! > combined[best]!) best = row;
      return TEXTS[best]!.id;
    },
  };
  return matcher;
}

// ---- drawing the prepared pictures (formulas only; no image files)

type Rgb = [number, number, number];
type Layer = (u: number, v: number, x: number, y: number) => [Rgb, number] | null;

function mix(a: Rgb, b: Rgb, t: number): Rgb {
  const k = Math.max(0, Math.min(1, t));
  return [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k];
}

function smooth(value: number): number {
  const t = Math.max(0, Math.min(1, value));
  return t * t * (3 - 2 * t);
}

/** Paints `layer` over `under` and returns the new picture (RGB bytes). */
function paintOver(under: Uint8ClampedArray, width: number, height: number, layer: Layer): Uint8ClampedArray {
  const out = new Uint8ClampedArray(under);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const hit = layer(x / (width - 1), y / (height - 1), x, y);
      if (!hit) continue;
      const [colour, alpha] = hit;
      const base = (y * width + x) * 3;
      for (let channel = 0; channel < 3; channel += 1) out[base + channel] = Math.round(under[base + channel]! + (colour[channel]! - under[base + channel]!) * Math.max(0, Math.min(1, alpha)));
    }
  }
  return out;
}

function flat(width: number, height: number, colour: Rgb): Uint8ClampedArray {
  const out = new Uint8ClampedArray(width * height * 3);
  for (let index = 0; index < out.length; index += 3) out.set(colour, index);
  return out;
}

const GREY: Rgb = [116, 116, 124];

/** `rise` runs from 0 (sun below the horizon, dark) to 1 (sun up, bright). */
function sunriseLayers(aspect: number, rise: number, boat: number | null): Layer[] {
  const horizon = 0.64;
  const sunY = horizon + 0.1 - 0.34 * rise;
  const light = 0.35 + 0.65 * rise;
  const far = (u: number) => horizon - 0.07 - 0.05 * Math.sin(u * 7 + 1) - 0.025 * Math.sin(u * 19 + 0.5);
  const near = (u: number) => 0.9 - 0.06 * Math.sin(u * 5 + 2) - 0.03 * Math.sin(u * 13);
  const fromSun = (u: number, v: number) => Math.hypot((u - 0.5) * aspect, v - sunY);
  return [
    (u, v) => {
      if (v > horizon) return [mix([24, 44, 84], [8, 16, 38], (v - horizon) / (1 - horizon)), 1];
      const high = mix([10, 16, 52], [36, 60, 130], rise);
      const low = mix([70, 40, 90], [255, 168, 92], rise);
      return [mix(high, low, smooth(v / horizon) ** 1.6), 1];
    },
    (u, v) => (v > horizon ? null : [[255, 196, 120], 0.75 * light * Math.exp(-(fromSun(u, v) ** 2) / 0.06)]),
    (u, v) => (v > horizon ? null : [[255, 244, 212], 1 - smooth((fromSun(u, v) - 0.085) / 0.02)]),
    (u, v) => (v > horizon || v < far(u) ? null : [mix([34, 30, 66], [84, 66, 118], rise), 1]),
    (u, v) => {
      if (v <= horizon) return null;
      const streak = Math.exp(-((u - 0.5) ** 2) / (0.004 + 0.03 * (v - horizon))) * (0.55 + 0.45 * Math.sin(v * 150));
      return [[255, 190, 120], 0.8 * light * streak];
    },
    (u, v, x, y) => {
      if (v > near(u)) return [[10, 12, 24], 1];
      if (boat !== null && Math.abs(u - boat) < 0.05 && v > 0.74 && v < 0.78) return [[14, 14, 26], 1];
      if (boat !== null && Math.abs(u - boat) < 0.012 && v > 0.66 && v <= 0.74) return [[14, 14, 26], 1];
      if (v < 0.34 && hash(x, y, 3) > 0.985) return [[255, 255, 255], 0.9 * (1 - rise * 0.8)];
      return null;
    },
  ];
}

function cityLayers(width: number): Layer[] {
  const ground = 0.8;
  const farTop = (x: number) => ground - 0.14 - 0.24 * hash(Math.floor(x / 6), 1, 11);
  const nearTop = (x: number) => ground + 0.02 - 0.2 - 0.3 * hash(Math.floor((x + 4) / 9), 2, 12);
  const gap = (x: number) => hash(Math.floor((x + 4) / 9), 5, 12) < 0.22;
  const aspect = 1.6;
  return [
    (u, v) => (v > ground ? [mix([18, 18, 30], [8, 8, 14], (v - ground) / (1 - ground)), 1] : [mix([6, 8, 30], [112, 62, 112], smooth(v / ground) ** 2), 1]),
    (u, v) => {
      const distance = Math.hypot((u - 0.78) * aspect, v - 0.2);
      return [[238, 238, 218], Math.max(1 - smooth((distance - 0.075) / 0.02), 0.45 * Math.exp(-(distance ** 2) / 0.03))];
    },
    (u, v, x) => (v <= ground && v >= farTop(x) ? [[40, 38, 74], 1] : null),
    (u, v, x) => (v <= ground + 0.02 && v >= nearTop(x) && !gap(x) ? [[14, 14, 30], 1] : null),
    (u, v, x, y) => {
      const onNear = v <= ground - 0.02 && v >= nearTop(x) + 0.03 && !gap(x);
      if (onNear) return x % 3 === 1 && y % 3 === 1 && hash(x, y, 21) > 0.42 ? [[255, 214, 120], 1] : null;
      const onFar = v <= ground - 0.02 && v >= farTop(x) + 0.03;
      return onFar && x % 2 === 0 && y % 3 === 0 && hash(x, y, 22) > 0.72 ? [[196, 170, 118], 0.9] : null;
    },
    (u, v, x, y) => {
      if (v > ground + 0.03) {
        const glow = hash(Math.floor(x / 2), 9, 23) > 0.55 ? 0.5 * (1 - (v - ground) / (1 - ground)) * (0.5 + 0.5 * Math.sin(y * 2.4)) : 0;
        return [[255, 200, 120], glow];
      }
      return v < Math.min(farTop(x), nearTop(x)) - 0.03 && hash(x, y, 24) > 0.988 && width > 0 ? [[255, 255, 255], 0.9] : null;
    },
  ];
}

function composites(width: number, height: number, layers: Layer[]): Uint8ClampedArray[] {
  const stack = [flat(width, height, GREY)];
  for (const layer of layers) stack.push(paintOver(stack[stack.length - 1]!, width, height, layer));
  return stack;
}

function noiseColumns(rows: number, count: number, seed: number): Int32Array[] {
  const rng = seededRng(seed);
  return Array.from({ length: count }, () => Int32Array.from({ length: rows }, () => (rng() & 255) - 128));
}

const SCALE = 32;
const NOISE_GAIN = SCALE / 4;

function assemble(columns: ArrayLike<number>[], rows: number): Matrix {
  const cols = columns.length;
  const data = new Int32Array(rows * cols);
  for (let col = 0; col < cols; col += 1) {
    const column = columns[col]!;
    for (let row = 0; row < rows; row += 1) data[row * cols + col] = toRing(column[row]!);
  }
  return { rows, cols, data };
}

function difference(a: Uint8ClampedArray, b: Uint8ClampedArray): Int32Array {
  return Int32Array.from(a, (value, index) => value - b[index]!);
}

/** Largest noise any pixel can receive, used to prove that nothing wraps around the prime. */
function headroomFor(noise: Int32Array[], rows: number): number {
  let worst = 0;
  for (let row = 0; row < rows; row += 1) {
    let sum = 0;
    for (const column of noise) sum += Math.abs(column[row]!) * NOISE_GAIN;
    worst = Math.max(worst, sum);
  }
  // Content is a weighted average of pictures, so it stays within 0..255·SCALE.
  return HALF - (255 * SCALE + worst);
}

function noiseInput(vector: Int32Array, offset: number, count: number, unit: number, strength: number, seed: number): void {
  for (let index = 0; index < count; index += 1) {
    vector[offset + index] = toRing(Math.round(NOISE_GAIN * strength * Math.cos(hash(index, 0, seed) * 6.283 + unit * 0.9)));
  }
}

function buildPicture(id: string, prompt: string, layers: Layer[], seed: number): PixelModel {
  const width = 96;
  const height = 60;
  const rows = width * height * 3;
  const stack = composites(width, height, layers);
  const noise = noiseColumns(rows, 16, seed);
  const content = stack.slice(1).map((picture, index) => difference(picture, stack[index]!));
  const weight = assemble([stack[0]!, ...content, ...noise], rows);
  const units = 14;
  return {
    id, kind: "picture", prompt, width, height, weight, units,
    frames: [stack[stack.length - 1]!],
    stepsPerFrame: units,
    scale: SCALE,
    headroom: headroomFor(noise, rows),
    input(unit) {
      const progress = unit / (units - 1);
      const vector = new Int32Array(weight.cols);
      vector[0] = SCALE;
      // Earlier layers arrive first, so the content is always a weighted average of real pictures.
      for (let layer = 0; layer < content.length; layer += 1) vector[1 + layer] = Math.round(SCALE * smooth((progress - layer * 0.08) / 0.5));
      noiseInput(vector, 1 + content.length, noise.length, unit, (1 - progress) ** 1.2, seed);
      return vector;
    },
  };
}

function buildClip(): PixelModel {
  const width = 64;
  const height = 40;
  const rows = width * height * 3;
  const count = 12;
  const base = flat(width, height, GREY);
  const frames = Array.from({ length: count }, (_, index) => {
    const t = index / (count - 1);
    const layers = sunriseLayers(width / height, t, 0.18 + 0.64 * t);
    return layers.reduce((picture, layer) => paintOver(picture, width, height, layer), base);
  });
  const noise = noiseColumns(rows, 12, 77);
  const weight = assemble([base, ...frames.map((frame) => difference(frame, base)), ...noise], rows);
  return {
    id: "clip-sunrise", kind: "clip", prompt: "A sunrise over the water, twelve frames", width, height, weight,
    units: count * 2,
    frames,
    stepsPerFrame: 2,
    scale: SCALE,
    headroom: headroomFor(noise, rows),
    input(unit) {
      const frame = Math.floor(unit / 2);
      const vector = new Int32Array(weight.cols);
      vector[0] = SCALE;
      if (unit % 2 === 1) {
        vector[1 + frame] = SCALE;
        return vector;
      }
      // First pass of a frame: half-way from the previous frame (or from grey), still noisy.
      vector[1 + frame] = SCALE / 2;
      if (frame > 0) vector[frame] = SCALE / 2;
      noiseInput(vector, 1 + count, noise.length, unit, frame === 0 ? 0.7 : 0.4, 77);
      return vector;
    },
  };
}

// ---- the reader's own input

const OWN_TEXT_LIMIT = 40;

/** A toy model for typed text: a permutation matrix that writes it backwards, one character per step. */
export function ownTextModel(text: string): TextModel {
  const chars = (text.length > 0 ? text : " ").slice(0, OWN_TEXT_LIMIT).split("");
  const count = chars.length;
  const codes = chars.map((char) => Math.min(char.charCodeAt(0), SHOW_PRIME - 1));
  const data = new Int32Array(count * count);
  for (let row = 0; row < count; row += 1) data[row * count + (count - 1 - row)] = 1;
  const tokens = [...chars].reverse();
  return {
    id: "own-text",
    kind: "text",
    prompt: chars.join(""),
    toy: "Toy model: it writes your text backwards.",
    answer: tokens.join(""),
    tokens,
    vocabulary: [...new Set(chars)].sort(),
    weight: { rows: count, cols: count, data },
    units: count,
    input(unit) {
      // One character per step, in its own position; every other entry is zero.
      const vector = new Int32Array(count);
      vector[count - 1 - unit] = codes[count - 1 - unit]!;
      return vector;
    },
    decodeToken(combined) {
      let best = 0;
      for (let row = 1; row < count; row += 1) if (combined[row]! > combined[best]!) best = row;
      return String.fromCharCode(combined[best]!);
    },
  };
}

/** Size of the drawing pad, in pixels. */
export const PAD = { width: 32, height: 20 };
const PAD_STRIPS = 4;
const PAD_SCALE = 4;
const PAD_BASE = [16, 24, 40];
const PAD_GAIN = [2, 3, 4];

/**
 * A toy model for a drawing (one ink value per pixel): it mirrors the drawing and raises its edges,
 * which is a linear map plus a constant. The drawing goes in as four strips; the device adds the parts.
 */
export function ownDrawingModel(ink: ArrayLike<number>): PixelModel {
  const { width, height } = PAD;
  const pixels = width * height;
  const rows = pixels * 3;
  const cols = pixels + 1;
  const data = new Int32Array(rows * cols);
  const picture = new Uint8ClampedArray(rows);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const here = y * width + (width - 1 - x);
      const behind = x >= 1 && y >= 1 ? (y - 1) * width + (width - x) : -1;
      for (let channel = 0; channel < 3; channel += 1) {
        const row = (y * width + x) * 3 + channel;
        const gain = PAD_GAIN[channel]!;
        data[row * cols + here] = toRing(3 * gain);
        if (behind >= 0) data[row * cols + behind] = toRing(-2 * gain);
        data[row * cols + pixels] = PAD_BASE[channel]! * PAD_SCALE;
        picture[row] = Math.round((PAD_BASE[channel]! * PAD_SCALE + gain * (3 * ink[here]! - 2 * (behind >= 0 ? ink[behind]! : 0))) / PAD_SCALE);
      }
    }
  }
  const stripRows = height / PAD_STRIPS;
  return {
    id: "own-drawing",
    kind: "picture",
    prompt: "Your drawing",
    toy: "Toy model: it mirrors your drawing and raises its edges.",
    width, height,
    weight: { rows, cols, data },
    units: PAD_STRIPS,
    frames: [picture],
    stepsPerFrame: PAD_STRIPS,
    scale: PAD_SCALE,
    headroom: HALF - PAD_SCALE * (PAD_BASE[2]! + 3 * 255),
    accumulate: true,
    input(unit) {
      const vector = new Int32Array(cols);
      for (let index = unit * stripRows * width; index < (unit + 1) * stripRows * width; index += 1) vector[index] = ink[index]!;
      if (unit === 0) vector[pixels] = 1;
      return vector;
    },
  };
}

const cache = new Map<string, ShowModel>();

export function getModel(id: string): ShowModel {
  const cached = cache.get(id);
  if (cached) return cached;
  const text = TEXTS.find((entry) => entry.id === id);
  let model: ShowModel;
  if (text) model = buildText(text);
  else if (id === "picture-sunrise") model = buildPicture(id, "A sunrise over the water", sunriseLayers(1.6, 0.8, null), 41);
  else if (id === "picture-city") model = buildPicture(id, "A city at night under the moon", cityLayers(96), 42);
  else if (id === "clip-sunrise") model = buildClip();
  else throw new Error(`unknown show model ${id}`);
  cache.set(id, model);
  return model;
}

/** Turns a recombined decoder output into RGB bytes. */
export function decodePixels(model: PixelModel, combined: Int32Array): Uint8ClampedArray {
  const out = new Uint8ClampedArray(combined.length);
  for (let index = 0; index < combined.length; index += 1) {
    const value = combined[index]!;
    out[index] = Math.round((value > HALF ? value - SHOW_PRIME : value) / model.scale);
  }
  return out;
}

/** How a worker's numbers look when drawn as pixels: the low byte of each value. */
export function workerBytes(values: Int32Array): Uint8ClampedArray {
  return Uint8ClampedArray.from(values, (value) => value & 255);
}

// ---------------------------------------------------------------- one protected step

export interface ShowSession {
  model: Computable;
  /** The two weight shares, split once; workers hold them for the whole run. */
  weights: [Matrix, Matrix];
  rng: Rng;
}

export interface ShowRole {
  activationShare: number;
  weightShare: number;
  product: Int32Array;
  valid: boolean;
}

export interface ShowStep {
  unit: number;
  activationShares: [Int32Array, Int32Array];
  roles: ShowRole[];
  rejected: boolean;
  /** W·x, or null when a product failed its check and nothing may be combined. */
  combined: Int32Array | null;
}

export function openSession(model: Computable, rng: Rng): ShowSession {
  return { model, weights: splitMatrix(model.weight, rng), rng };
}

/**
 * One matrix product on shares. Role `2a + w` multiplies activation share `a` by weight share `w`.
 * `tamper` names a role whose first product entry is changed by one, to show the check at work.
 */
export function runStep(session: ShowSession, unit: number, tamper?: number): ShowStep {
  const { model, weights, rng } = session;
  const activationShares = splitVector(model.input(unit), rng);
  // Non-zero challenge entries, so that a change to any single entry is always caught.
  const challenge = Int32Array.from({ length: model.weight.rows }, () => rng() || 1);
  const folded = [foldWeights(weights[0], challenge), foldWeights(weights[1], challenge)];
  const roles: ShowRole[] = [];
  for (let a = 0; a < 2; a += 1) {
    for (let w = 0; w < 2; w += 1) {
      const product = matVec(weights[w]!, activationShares[a]!);
      if (tamper === roles.length) product[0] = (product[0]! + 1) % SHOW_PRIME;
      const valid = dot(folded[w]!, activationShares[a]!) === dot(challenge, product);
      roles.push({ activationShare: a, weightShare: w, product, valid });
    }
  }
  const rejected = roles.some((role) => !role.valid);
  return { unit, activationShares, roles, rejected, combined: rejected ? null : combine(roles.map((role) => role.product)) };
}

// ---------------------------------------------------------------- what other parties could work out

export interface PooledView {
  /** Your input, if the two workers hold both pieces of it. */
  input: Int32Array | null;
  /** The output, if they also hold both pieces of the weights. */
  output: Int32Array | null;
  /** Their two products added together: the best they can do otherwise. */
  sum: Int32Array;
}

/** What two workers learn if they pool everything they hold in one step. */
export function pooledView(session: ShowSession, step: ShowStep, first: number, second: number): PooledView {
  const a = step.roles[first]!;
  const b = step.roles[second]!;
  const sum = combine([a.product, b.product]);
  if (a.activationShare === b.activationShare) return { input: null, output: null, sum };
  const input = combine([step.activationShares[0], step.activationShares[1]]);
  if (a.weightShare === b.weightShare) return { input, output: null, sum };
  const weight: Matrix = { rows: session.model.weight.rows, cols: session.model.weight.cols, data: combine([session.weights[0].data, session.weights[1].data]) };
  return { input, output: matVec(weight, input), sum };
}

/** A worker's numbers as they travel: encrypted with AES-GCM under a fresh key. Returns the ciphertext. */
export async function sealForWire(values: Int32Array): Promise<Uint8Array> {
  const key = await crypto.subtle.generateKey({ name: "AES-GCM", length: 256 }, false, ["encrypt"]);
  const nonce = crypto.getRandomValues(new Uint8Array(12));
  const plain = new Uint8Array(new Uint16Array(values).buffer);
  return new Uint8Array(await crypto.subtle.encrypt({ name: "AES-GCM", iv: nonce }, key, plain));
}

// ---------------------------------------------------------------- disguise and sealed routing

/** A model whose input order has been shuffled with a secret order, and its table shuffled to match. */
export interface Shuffled extends Computable {
  /** `order[k]` is the original position of what now sits at position `k`. */
  order: Int32Array;
}

/**
 * Disguise, not encryption: the numbers keep their values and only change places. Because the model's
 * columns move with them, every product comes out exactly as before.
 */
export function shuffled(model: Computable, rng: Rng): Shuffled {
  const { rows, cols, data } = model.weight;
  const order = Int32Array.from({ length: cols }, (_, index) => index);
  for (let index = cols - 1; index > 0; index -= 1) {
    const other = rng() % (index + 1);
    [order[index], order[other]] = [order[other]!, order[index]!];
  }
  const moved = new Int32Array(rows * cols);
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) moved[row * cols + col] = data[row * cols + order[col]!]!;
  }
  return {
    weight: { rows, cols, data: moved },
    units: model.units,
    order,
    input(unit) {
      const plain = model.input(unit);
      return Int32Array.from(order, (from) => plain[from]!);
    },
  };
}

export interface SealedRoute {
  /** What each relay receives, in order. A relay that receives a bad packet is the last entry. */
  views: Uint8Array[];
  /** The bytes that come out after the last relay, or null if a relay refused the packet. */
  delivered: Uint8Array | null;
  /** Index of the relay whose authenticity check failed, if any. */
  failedAt: number | null;
}

const RELAYS = 3;

/**
 * Wraps a piece in three layers of AES-GCM, one per relay, then really sends it through: every relay
 * removes its own layer and forwards the rest. `alterAt` flips one bit of what that relay receives.
 */
export async function sealRoute(values: Int32Array, alterAt?: number): Promise<SealedRoute> {
  const keys = await Promise.all(Array.from({ length: RELAYS }, () => crypto.subtle.generateKey({ name: "AES-GCM", length: 256 }, false, ["encrypt", "decrypt"])));
  const nonces = keys.map(() => crypto.getRandomValues(new Uint8Array(12)));
  let packet = new Uint8Array(new Uint16Array(values).buffer);
  // Innermost layer last: the first relay's layer is the outermost.
  for (let hop = RELAYS - 1; hop >= 0; hop -= 1) {
    packet = new Uint8Array(await crypto.subtle.encrypt({ name: "AES-GCM", iv: nonces[hop]! as unknown as BufferSource }, keys[hop]!, packet as unknown as BufferSource));
  }
  const views: Uint8Array[] = [];
  for (let hop = 0; hop < RELAYS; hop += 1) {
    if (alterAt === hop) {
      packet = new Uint8Array(packet);
      packet[0]! ^= 1;
    }
    views.push(packet);
    try {
      packet = new Uint8Array(await crypto.subtle.decrypt({ name: "AES-GCM", iv: nonces[hop]! as unknown as BufferSource }, keys[hop]!, packet as unknown as BufferSource));
    } catch {
      return { views, delivered: null, failedAt: hop };
    }
  }
  return { views, delivered: packet, failedAt: null };
}
