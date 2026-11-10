import { describe, expect, it } from "vitest";
import { conjugateLinear, eulerStep, frameDeltaStep, fwht, matMul, rotate2d } from "@/lib/simulation/algebra";

describe("algebra simulation", () => {
  it("computes every FWHT butterfly and matches the direct reference", () => {
    const { stages, output } = fwht([1, 2, 3, 4]);
    expect(stages).toHaveLength(3);
    expect(output).toEqual([5, -1, -2, 0]);
  });

  it("rejects non-power-of-two dimensions", () => {
    expect(() => fwht([1, 2, 3])).toThrow("power of two");
  });

  it("recovers conjugated linear output within floating error", () => {
    const basis = rotate2d(Math.PI / 3);
    const weight = [[1, 2], [3, 4]];
    const transformed = conjugateLinear(weight, basis, basis);
    const x = [[2, -1]];
    const rotated = matMul(x, basis[0]!.map((_, index) => basis.map((row) => row[index]!)));
    const y = matMul(rotated, transformed.map((row) => row.slice()));
    const recovered = matMul(y, basis);
    const expected = matMul(x, weight);
    expect(recovered[0]![0]).toBeCloseTo(expected[0]![0]!, 10);
    expect(recovered[0]![1]).toBeCloseTo(expected[0]![1]!, 10);
  });

  it("keeps Euler steps linear", () => {
    expect(eulerStep([[1, 2]], [[0.5, -1]], 0.5)).toEqual([[0.75, 2.5]]);
  });

  it("solves the frame DeltaNet normal equation", () => {
    const { state, output } = frameDeltaStep(
      [[0, 0], [0, 0]],
      [[1, 0], [0, 1]],
      [[1, 2], [3, 4]],
      [[1, 0], [0, 1]],
      [1, 1],
    );
    expect(state[0]![0]).toBeCloseTo(0.5);
    expect(state[0]![1]).toBeCloseTo(1.5);
    expect(state[1]![0]).toBeCloseTo(1);
    expect(state[1]![1]).toBeCloseTo(2);
    expect(output[0]![0]).toBeCloseTo(0.5);
    expect(output[1]![1]).toBeCloseTo(2);
  });
});
