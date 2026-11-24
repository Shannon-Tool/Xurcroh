import { describe, expect, it } from "vitest";
import {
  PAD, SHOW_PRIME, decodePixels, getModel, matVec, openSession, ownDrawingModel, ownTextModel, pooledView, runStep, sealForWire,
  seededRng, type PixelModel,
} from "@/lib/simulation/show";

function run(model: ReturnType<typeof ownTextModel>, seed = 1) {
  const session = openSession(model, seededRng(seed));
  return Array.from({ length: model.units }, (_, unit) => runStep(session, unit));
}

describe("your own text", () => {
  it("is written backwards, one character per step, from the recombined products", () => {
    const model = ownTextModel("Hello, wörld!");
    expect(model.units).toBe(13);
    const produced = run(model).map((step) => {
      expect(step.rejected).toBe(false);
      return model.decodeToken(step.combined!);
    });
    expect(produced.join("")).toBe("!dlröw ,olleH");
    expect(model.answer).toBe("!dlröw ,olleH");
    expect(model.prompt).toBe("Hello, wörld!");
  });

  it("never shows a worker the character it is working on", () => {
    const model = ownTextModel("secret");
    for (const step of run(model, 7)) {
      const code = model.decodeToken(step.combined!).charCodeAt(0);
      for (const share of step.activationShares) expect(Array.from(share)).not.toContain(code);
    }
    // Same text, fresh randomness: what the workers hold changes, the result does not.
    const again = run(model, 8);
    expect(Array.from(again[0]!.roles[0]!.product)).not.toEqual(Array.from(run(model, 7)[0]!.roles[0]!.product));
  });

  it("copes with empty and over-long text", () => {
    expect(ownTextModel("").units).toBe(1);
    expect(ownTextModel("x".repeat(200)).units).toBe(40);
  });
});

describe("your own drawing", () => {
  const ink = new Uint8ClampedArray(PAD.width * PAD.height);
  for (let y = 4; y < 16; y += 1) ink[y * PAD.width + 5] = 255;
  for (let x = 5; x < 14; x += 1) ink[4 * PAD.width + x] = 255;

  it("is mirrored and embossed in four strips whose verified results add up to the direct calculation", () => {
    const model = ownDrawingModel(ink);
    expect(model.units).toBe(4);
    expect(model.accumulate).toBe(true);
    expect(model.headroom).toBeGreaterThan(0);
    const session = openSession(model, seededRng(3));
    const total = new Int32Array(model.weight.rows);
    for (let unit = 0; unit < model.units; unit += 1) {
      const step = runStep(session, unit);
      expect(step.rejected).toBe(false);
      for (let row = 0; row < total.length; row += 1) total[row] = (total[row]! + step.combined![row]!) % SHOW_PRIME;
    }
    const picture = decodePixels(model, total);
    expect(Array.from(picture)).toEqual(Array.from(model.frames[0]!));
    // The vertical stroke at x = 5 ends up at the mirrored column, and blank areas keep the background.
    const at = (x: number, y: number) => picture[(y * PAD.width + x) * 3 + 2]!;
    expect(at(PAD.width - 1 - 5, 10)).toBeGreaterThan(at(2, 10) + 60);
    expect(at(2, 10)).toBe(at(3, 18));
  });

  it("gives a worker a product unrelated to the drawing", () => {
    const model = ownDrawingModel(ink);
    const step = runStep(openSession(model, seededRng(5)), 0);
    const direct = matVec(model.weight, model.input(0));
    expect(Array.from(step.roles[0]!.product.subarray(0, 60))).not.toEqual(Array.from(direct.subarray(0, 60)));
  });
});

describe("whose eyes", () => {
  const model = getModel("picture-sunrise") as PixelModel;
  const session = openSession(model, seededRng(11));
  const step = runStep(session, model.units - 1);

  it("lets two workers with matching pieces rebuild the input and the output", () => {
    const view = pooledView(session, step, 0, 3);
    expect(Array.from(view.input!)).toEqual(Array.from(model.input(model.units - 1)));
    expect(Array.from(decodePixels(model, view.output!))).toEqual(Array.from(model.frames[0]!));
  });

  it("lets two workers with both pieces of your data but one piece of the weights rebuild only the input", () => {
    const view = pooledView(session, step, 0, 2);
    expect(view.input).not.toBeNull();
    expect(view.output).toBeNull();
  });

  it("gives two workers holding the same piece of your data nothing but noise", () => {
    const view = pooledView(session, step, 0, 1);
    expect(view.input).toBeNull();
    expect(view.output).toBeNull();
    expect(Array.from(view.sum.subarray(0, 200))).not.toEqual(Array.from(step.combined!.subarray(0, 200)));
  });

  it("puts only ciphertext on the wire, different every time and longer than the data by the tag", async () => {
    const product = step.roles[0]!.product.subarray(0, 100);
    const first = await sealForWire(product);
    const second = await sealForWire(product);
    expect(first.length).toBe(product.length * 2 + 16);
    expect(Array.from(first)).not.toEqual(Array.from(second));
    const plain = new Uint8Array(new Uint16Array(product).buffer);
    expect(Array.from(first.subarray(0, plain.length))).not.toEqual(Array.from(plain));
  });
});
