import { describe, expect, it } from "vitest";
import { PRIME, freivaldsCheck } from "@/lib/simulation/fortress";
import {
  SHOW_MODELS, SHOW_PRIME, combine, decodePixels, freivalds, getModel, matVec, openSession, runStep, seededRng, splitVector,
  workerBytes, type Matrix, type PixelModel, type TextModel,
} from "@/lib/simulation/show";

function matrix(rows: number[][]): Matrix {
  return { rows: rows.length, cols: rows[0]!.length, data: Int32Array.from(rows.flat()) };
}

describe("share arithmetic for the show", () => {
  it("splits a vector into two shares that are not the secret and add back to it", () => {
    const secret = Int32Array.from([0, 1, 2, 65520, 12345]);
    const [first, second] = splitVector(secret, seededRng(7));
    expect(Array.from(first)).not.toEqual(Array.from(secret));
    expect(Array.from(second)).not.toEqual(Array.from(secret));
    expect(Array.from(combine([first, second]))).toEqual(Array.from(secret));
    for (const value of [...first, ...second]) expect(value >= 0 && value < SHOW_PRIME).toBe(true);
  });

  it("draws share values evenly across the whole range", () => {
    const rng = seededRng(11);
    const buckets = new Array<number>(16).fill(0);
    for (let index = 0; index < 32_000; index += 1) buckets[Math.floor((rng() / SHOW_PRIME) * 16)]! += 1;
    for (const count of buckets) expect(Math.abs(count - 2000)).toBeLessThan(200);
  });

  it("multiplies exactly modulo the prime", () => {
    const weight = matrix([[65520, 2], [3, 65519]]);
    expect(Array.from(matVec(weight, Int32Array.from([65520, 5])))).toEqual([(65520 * 65520 + 10) % SHOW_PRIME, (3 * 65520 + 65519 * 5) % SHOW_PRIME]);
  });

  it("agrees with the laboratory's big-integer Freivalds check", () => {
    const weight = matrix([[5, 7, 11], [13, 17, 19]]);
    const activation = Int32Array.from([3, 4, 6]);
    const product = matVec(weight, activation);
    const challenge = Int32Array.from([23, 29]);
    const rows = [[5, 7, 11], [13, 17, 19]];
    expect(freivalds(weight, activation, product, challenge)).toBe(true);
    expect(freivaldsCheck(rows, [3, 4, 6], Array.from(product), [23, 29])).toBe(true);
    const wrong = Int32Array.from(product);
    wrong[1] = (wrong[1]! + 1) % SHOW_PRIME;
    expect(freivalds(weight, activation, wrong, challenge)).toBe(false);
    expect(freivaldsCheck(rows, [3, 4, 6], Array.from(wrong), [23, 29])).toBe(false);
    expect(PRIME > BigInt(SHOW_PRIME)).toBe(true);
  });
});

describe("text shown by the show", () => {
  const texts = SHOW_MODELS.filter((entry) => entry.kind === "text");

  it("offers three prepared prompts", () => {
    expect(texts).toHaveLength(3);
  });

  it.each(texts.map((entry) => entry.id))("decodes %s token by token from the recombined products", (id) => {
    const model = getModel(id) as TextModel;
    const session = openSession(model, seededRng(3));
    const produced: string[] = [];
    for (let unit = 0; unit < model.units; unit += 1) {
      const step = runStep(session, unit);
      expect(step.rejected).toBe(false);
      expect(step.roles).toHaveLength(4);
      expect(step.roles.every((role) => role.valid)).toBe(true);
      expect(Array.from(step.combined!)).toEqual(Array.from(matVec(model.weight, model.input(unit))));
      produced.push(model.decodeToken(step.combined!));
    }
    expect(produced).toEqual(model.tokens);
    expect(produced.join("")).toBe(model.answer);
    expect(model.answer.length).toBeGreaterThan(80);
  });

  it("gives each worker numbers that do not reveal which token is being produced", () => {
    const model = getModel(texts[0]!.id) as TextModel;
    const first = runStep(openSession(model, seededRng(1)), 4);
    const second = runStep(openSession(model, seededRng(2)), 4);
    // Same step, different randomness: every worker's view changes although the token does not.
    for (let role = 0; role < 4; role += 1) expect(Array.from(first.roles[role]!.product)).not.toEqual(Array.from(second.roles[role]!.product));
    expect(model.decodeToken(first.combined!)).toBe(model.decodeToken(second.combined!));
    expect(Array.from(first.activationShares[0])).not.toEqual(Array.from(model.input(4)));
  });

  it("rejects a step when one worker's product is off by one, and accepts the retry", () => {
    const model = getModel(texts[1]!.id) as TextModel;
    const session = openSession(model, seededRng(5));
    for (let role = 0; role < 4; role += 1) {
      const bad = runStep(session, 2, role);
      expect(bad.rejected).toBe(true);
      expect(bad.combined).toBeNull();
      expect(bad.roles.map((entry) => entry.valid)).toEqual([0, 1, 2, 3].map((index) => index !== role));
    }
    const retry = runStep(session, 2);
    expect(retry.rejected).toBe(false);
    expect(model.decodeToken(retry.combined!)).toBe(model.tokens[2]);
  });
});

describe("pictures and the clip shown by the show", () => {
  const pictures = SHOW_MODELS.filter((entry) => entry.kind === "picture");
  const clip = SHOW_MODELS.find((entry) => entry.kind === "clip")!;

  it("offers two pictures and one clip", () => {
    expect(pictures).toHaveLength(2);
    expect(clip).toBeDefined();
  });

  it.each(pictures.map((entry) => entry.id))("builds %s step by step from recombined products, ending on the exact picture", (id) => {
    const model = getModel(id) as PixelModel;
    const session = openSession(model, seededRng(9));
    let last: Uint8ClampedArray | null = null;
    let firstError = 0;
    for (let unit = 0; unit < model.units; unit += 1) {
      const step = runStep(session, unit);
      expect(step.rejected).toBe(false);
      expect(Array.from(step.combined!.subarray(0, 300))).toEqual(Array.from(matVec(model.weight, model.input(unit)).subarray(0, 300)));
      last = decodePixels(model, step.combined!);
      if (unit === 0) firstError = meanError(last, model.frames[0]!);
    }
    expect(last).toHaveLength(model.width * model.height * 3);
    expect(Array.from(last!)).toEqual(Array.from(model.frames[0]!));
    // It starts as noise, far from the picture.
    expect(firstError).toBeGreaterThan(25);
  });

  it("never lets an intermediate value wrap around the prime", () => {
    for (const entry of [...pictures, clip]) {
      const model = getModel(entry.id) as PixelModel;
      expect(model.headroom).toBeGreaterThan(0);
    }
  });

  it("shows each worker noise that is unrelated to the picture", () => {
    const model = getModel(pictures[0]!.id) as PixelModel;
    const step = runStep(openSession(model, seededRng(21)), model.units - 1);
    const picture = model.frames[0]!;
    for (const role of step.roles) {
      const view = workerBytes(role.product);
      expect(view).toHaveLength(picture.length);
      const mean = view.reduce((sum, value) => sum + value, 0) / view.length;
      expect(Math.abs(mean - 127.5)).toBeLessThan(4);
      expect(Math.abs(correlation(view, picture))).toBeLessThan(0.05);
    }
  });

  it("produces every frame of the clip exactly, two steps per frame", () => {
    const model = getModel(clip.id) as PixelModel;
    expect(model.frames.length).toBe(12);
    expect(model.units).toBe(24);
    const session = openSession(model, seededRng(13));
    for (let unit = 0; unit < model.units; unit += 1) {
      const step = runStep(session, unit);
      expect(step.rejected).toBe(false);
      if (unit % 2 === 1) expect(Array.from(decodePixels(model, step.combined!))).toEqual(Array.from(model.frames[(unit - 1) / 2]!));
    }
    // The frames differ from one another: something moves.
    expect(meanError(model.frames[0]!, model.frames[11]!)).toBeGreaterThan(3);
  });

  it("catches a tampered product even though it changes one number by one", () => {
    const model = getModel(pictures[1]!.id) as PixelModel;
    const step = runStep(openSession(model, seededRng(4)), 3, 2);
    expect(step.rejected).toBe(true);
    expect(step.roles[2]!.valid).toBe(false);
  });
});

function meanError(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let total = 0;
  for (let index = 0; index < a.length; index += 1) total += Math.abs(a[index]! - b[index]!);
  return total / a.length;
}

function correlation(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const count = a.length;
  let meanA = 0;
  let meanB = 0;
  for (let index = 0; index < count; index += 1) { meanA += a[index]!; meanB += b[index]!; }
  meanA /= count;
  meanB /= count;
  let cross = 0;
  let varA = 0;
  let varB = 0;
  for (let index = 0; index < count; index += 1) {
    const da = a[index]! - meanA;
    const db = b[index]! - meanB;
    cross += da * db;
    varA += da * da;
    varB += db * db;
  }
  return cross / Math.sqrt(varA * varB);
}
