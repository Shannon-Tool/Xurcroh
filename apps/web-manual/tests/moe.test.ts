import { describe, expect, it } from "vitest";
import { combineMoe, routeMoe } from "@/lib/simulation/moe";

describe("MoE routing simulator", () => {
  it("pads every expert and preserves weighted output", () => {
    const hidden = [[1, 0], [0, 1], [1, 1]];
    const router = [[1, 0], [0, 1], [1, 1]];
    const plan = routeMoe(hidden, router, 1, 3);
    expect(plan.loads).toEqual([3, 3, 3]);
    expect(plan.dummy.flat().some(Boolean)).toBe(true);
    expect(plan.weights.flat().every((weight) => weight === 0 || weight === 1)).toBe(true);
    const outputs = [
      [[1, 1], [2, 2], [9, 9]],
      [[3, 3], [4, 4], [9, 9]],
      [[5, 5], [6, 6], [9, 9]],
    ];
    const result = combineMoe(plan, outputs);
    expect(result[0]).toEqual([1, 1]);
    expect(result[1]).toEqual([3, 3]);
    expect(result[2]).toEqual([5, 5]);
  });

  it("rejects capacity overflow instead of dropping tokens", () => {
    expect(() => routeMoe([[1], [1]], [[1], [1]], 1, 1)).toThrow("capacity");
  });
});
