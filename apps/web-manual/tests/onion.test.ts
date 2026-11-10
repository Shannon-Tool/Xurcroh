import { describe, expect, it } from "vitest";
import { openOnion, ReplayWindow, sealOnion } from "@/lib/simulation/onion";

describe("onion routing simulation", () => {
  it("decrypts the innermost layer with the correct key", async () => {
    const secret = new TextEncoder().encode("protected");
    const layers = await sealOnion([{ nextHop: "worker-b" }, { nextHop: "client" }], secret);
    expect(layers).toHaveLength(2);
    expect(layers[0]!.returnSecret).toBeNull();
    expect(layers[1]!.returnSecret).not.toBeNull();
    expect(Array.from(await openOnion(layers[1]!))).toEqual(Array.from(secret));
  });

  it("rejects tampered ciphertext", async () => {
    const layers = await sealOnion([{ nextHop: "client" }], new Uint8Array([1, 2, 3]));
    layers[0]!.ciphertext[0]! ^= 1;
    await expect(openOnion(layers[0]!)).rejects.toThrow();
  });

  it("rejects replayed sequence numbers", () => {
    const window = new ReplayWindow();
    expect(window.accept(4)).toBe(true);
    expect(window.accept(4)).toBe(false);
    expect(window.accept(5)).toBe(true);
  });
});
