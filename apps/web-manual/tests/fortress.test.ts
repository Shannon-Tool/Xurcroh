import { describe, expect, it } from "vitest";
import { executeShareGrid, freivaldsCheck, placeOperations } from "@/lib/simulation/fortress";

describe("Fortress simulations", () => {
  it("reconstructs the exact ring product and attributes a tampered role", () => {
    const weight = [[1, 2], [3, 4]];
    const activation = [5, 7];
    const grid = executeShareGrid(weight, activation, 0);
    expect(grid.combined).toEqual([19n, 43n]);
    expect(grid.roles).toHaveLength(4);
    expect(() => executeShareGrid(weight, activation, 0, "0:0")).toThrow(/role 0:0/);
  });

  it("verifies exact Freivalds products and rejects modification", () => {
    const weight = [[1, 2], [3, 4]];
    expect(freivaldsCheck(weight, [5, 7], [19, 43], [11, 13])).toBe(true);
    expect(freivaldsCheck(weight, [5, 7], [20, 43], [11, 13])).toBe(false);
  });

  it("places adjacent and paired operations on disjoint diverse providers", () => {
    const placement = placeOperations(0);
    expect(placement).toHaveLength(16);
    for (const operation of ["query", "key", "gate", "up"]) {
      const providers = placement.filter((role) => role.operation === operation).map((role) => role.provider);
      expect(new Set(providers).size).toBe(4);
    }
    const query = new Set(placement.filter((role) => role.operation === "query").map((role) => role.provider));
    const key = new Set(placement.filter((role) => role.operation === "key").map((role) => role.provider));
    expect([...query].every((provider) => !key.has(provider))).toBe(true);
    expect(placeOperations(1)).not.toEqual(placement);
  });
});
