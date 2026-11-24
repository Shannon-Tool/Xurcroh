import { describe, expect, it } from "vitest";
import {
  additiveShares,
  combineShares,
  decodeExl2Mixed,
  decodeQ4_0,
  FIELD_MODULUS,
  procedural3inst,
  quantizeFortress,
  quantizationError,
} from "@/lib/simulation/quantized";

describe("quantized format laboratory", () => {
  it("decodes Q4_0 and EXL2 mixed-width source values", () => {
    expect(decodeQ4_0(0.5, [0, 8, 15])).toEqual([-4, 0, 3.5]);
    expect(decodeExl2Mixed([
      { bits: 2, scale: 0.25, codes: [0, 1, 2, 3] },
      { bits: 4, scale: 0.5, codes: [0, 8, 15] },
    ])).toEqual([-0.5, -0.25, 0, 0.25, -4, 0, 3.5]);
  });

  it("combines exact BigInt additive shares without revealing the source in one share", () => {
    const values = [1, -2, 7];
    const shares = additiveShares(values, 2, 731);
    expect(combineShares(shares)).toEqual([1n, FIELD_MODULUS - 2n, 7n]);
    expect(shares[0]).not.toEqual(values.map(BigInt));
  });

  it("measures quantization error and computes a real procedural codebook value", () => {
    const quantized = quantizeFortress([0.1, -0.2, 0.7]);
    const recovered = quantized.integers.map((value) => value * quantized.scale);
    expect(quantizationError([0.1, -0.2, 0.7], recovered)).toBeGreaterThan(0);
    expect(Number.isFinite(procedural3inst(731))).toBe(true);
  });
});

describe("signed values in the field", () => {
  it("recombines shares of negative integers to the same negative integers", async () => {
    const { additiveShares, combineShares, signedField, FIELD_MODULUS } = await import("@/lib/simulation/quantized");
    const values = [-4, -6, 0, 7];
    expect(combineShares(additiveShares(values, 2, 99)).map(signedField)).toEqual(values);
    expect(signedField(FIELD_MODULUS - 1n)).toBe(-1);
    expect(signedField(5n)).toBe(5);
  });
});
