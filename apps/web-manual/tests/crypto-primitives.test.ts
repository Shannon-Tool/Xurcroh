// @vitest-environment node
import { afterEach, describe, expect, it, vi } from "vitest";
import { browserPrimitives, javascriptPrimitives, primitives, randomBytes, type Primitives } from "@/lib/crypto/primitives";
import { attest, makeEnclave, verifyReport, WORKER_CODE } from "@/lib/simulation/enclave";
import { openOnion, sealOnion } from "@/lib/simulation/onion";
import { sealRoute } from "@/lib/simulation/show";

const text = (value: string) => new TextEncoder().encode(value);
const hex = (bytes: Uint8Array) => Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");

afterEach(() => {
  vi.unstubAllGlobals();
});

// A page opened over plain HTTP: random numbers are there, `crypto.subtle` is not.
const real = globalThis.crypto;
const withoutSubtle = () => vi.stubGlobal("crypto", { getRandomValues: <T extends ArrayBufferView>(array: T) => real.getRandomValues(array as never) });

/** The page's own cryptography, for a browser that offers none (any page opened over plain HTTP). */
async function both(): Promise<[Primitives, Primitives]> {
  return [browserPrimitives, await javascriptPrimitives()];
}

describe("cryptography with and without the browser's own", () => {
  it("hashes the same in both", async () => {
    for (const each of await both()) {
      expect(hex(await each.sha256(text("abc")))).toBe("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }
  });

  it("opens with one what the other sealed, and refuses a wrong key or a changed byte", async () => {
    const [browser, javascript] = await both();
    const key = randomBytes(32);
    const nonce = randomBytes(12);
    const plain = text("How is my prompt kept from the workers?");
    for (const [sealer, opener] of [[browser, javascript], [javascript, browser]] as const) {
      const sealed = await sealer.aesSeal(key, nonce, plain);
      expect(sealed).toHaveLength(plain.length + 16);
      expect(hex(sealed)).not.toContain(hex(plain).slice(0, 16));
      expect(hex(await opener.aesOpen(key, nonce, sealed))).toBe(hex(plain));
      const wrongKey = Uint8Array.from(key);
      wrongKey[0]! ^= 1;
      await expect(opener.aesOpen(wrongKey, nonce, sealed)).rejects.toThrow();
      const changed = Uint8Array.from(sealed);
      changed[3]! ^= 1;
      await expect(opener.aesOpen(key, nonce, changed)).rejects.toThrow();
    }
    // The same key, nonce and message give the same sealed bytes in both.
    expect(hex(await browser.aesSeal(key, nonce, plain))).toBe(hex(await javascript.aesSeal(key, nonce, plain)));
  });

  it("agrees on a shared key whichever side computes it", async () => {
    const [browser, javascript] = await both();
    const here = await browser.ecdhPair();
    const there = await javascript.ecdhPair();
    expect(here.publicKey).toHaveLength(65);
    expect(there.publicKey).toHaveLength(65);
    const one = await browser.ecdhShared(here.secretKey, there.publicKey);
    const other = await javascript.ecdhShared(there.secretKey, here.publicKey);
    expect(one).toHaveLength(32);
    expect(hex(one)).toBe(hex(other));
  });

  it("verifies with one what the other signed, and refuses a changed message", async () => {
    const [browser, javascript] = await both();
    const message = text("measurement and nonce");
    for (const [signer, verifier] of [[browser, javascript], [javascript, browser], [javascript, javascript]] as const) {
      const pair = await signer.ecdsaPair();
      const signature = await signer.ecdsaSign(pair.secretKey, message);
      expect(signature).toHaveLength(64);
      expect(await verifier.ecdsaVerify(pair.publicKey, signature, message)).toBe(true);
      expect(await verifier.ecdsaVerify(pair.publicKey, signature, text("measurement and nonce!"))).toBe(false);
    }
  });

  it("uses the browser's when it is there and its own when it is not", async () => {
    expect((await primitives()).name).toBe("browser");
    withoutSubtle();
    expect(globalThis.crypto.subtle).toBeUndefined();
    expect((await primitives()).name).toBe("javascript");
  });
});

describe("the manual's engines on a page without the browser's cryptography", () => {

  it("wraps a message for three relays and opens every layer", async () => {
    withoutSubtle();
    const layers = await sealOnion([{ nextHop: "worker-1" }, { nextHop: "worker-2" }, { nextHop: "client" }], text("my question"));
    expect(layers).toHaveLength(3);
    expect(new TextDecoder().decode(await openOnion(layers[2]!))).toBe("my question");
    const changed = { ...layers[0]!, ciphertext: Uint8Array.from(layers[0]!.ciphertext) };
    changed.ciphertext[0]! ^= 1;
    await expect(openOnion(changed)).rejects.toThrow();
  });

  it("signs and checks a worker's report", async () => {
    withoutSubtle();
    const enclave = await makeEnclave();
    const nonce = randomBytes(16);
    expect((await verifyReport(await attest(enclave, nonce), enclave.publicKey, nonce)).ok).toBe(true);
    const swapped = await verifyReport(await attest(enclave, nonce, `${WORKER_CODE} + something else`), enclave.publicKey, nonce);
    expect(swapped).toEqual({ signed: true, expected: false, ok: false });
    expect((await verifyReport(await attest(enclave, nonce), enclave.publicKey, randomBytes(16))).ok).toBe(false);
  });

  it("seals a piece in three layers, delivers it, and stops an altered packet", async () => {
    withoutSubtle();
    const values = Int32Array.from([1, 2, 3, 65520]);
    const route = await sealRoute(values);
    expect(route.views.map((view) => view.length)).toEqual([8 + 48, 8 + 32, 8 + 16]);
    expect(Array.from(new Uint16Array(route.delivered!.buffer.slice(0)))).toEqual([1, 2, 3, 65520]);
    const altered = await sealRoute(values, 1);
    expect(altered.delivered).toBeNull();
    expect(altered.failedAt).toBe(1);
  });
});
