import { describe, expect, it } from "vitest";
import { simulateMemory } from "@/lib/simulation/memory";

describe("GPU memory lifecycle simulation", () => {
  it("demonstrates zeroized reuse when synchronized", () => {
    expect(simulateMemory(true).at(-1)!.bytes).toEqual([0, 0, 0, 0]);
  });

  it("demonstrates residual disclosure when not synchronized", () => {
    expect(simulateMemory(false).at(-1)!.bytes).toEqual([0xde, 0xad, 0xbe, 0xef]);
  });
});
