import { describe, expect, it } from "vitest";
import { attest, makeEnclave, verifyReport, WORKER_CODE } from "@/lib/simulation/enclave";
import { getMatcher, getModel, matVec, openSession, runStep, sealRoute, seededRng, shuffled } from "@/lib/simulation/show";

describe("secret shuffle before splitting", () => {
  it("reorders the numbers and the model's table together, so the result is unchanged", () => {
    const matcher = getMatcher();
    const disguised = shuffled(matcher, seededRng(8));
    expect(Array.from(disguised.order).sort((a, b) => a - b)).toEqual(Array.from({ length: matcher.weight.cols }, (_, index) => index));
    expect(Array.from(disguised.order)).not.toEqual(Array.from({ length: matcher.weight.cols }, (_, index) => index));
    for (let unit = 0; unit < matcher.units; unit += 1) {
      expect(Array.from(disguised.input(unit))).not.toEqual(Array.from(matcher.input(unit)));
      expect(Array.from(disguised.input(unit)).sort()).toEqual(Array.from(matcher.input(unit)).sort());
      expect(Array.from(matVec(disguised.weight, disguised.input(unit)))).toEqual(Array.from(matVec(matcher.weight, matcher.input(unit))));
      const step = runStep(openSession(disguised, seededRng(unit + 1)), unit);
      expect(matcher.pick(step.combined!)).toBe(matcher.questions[unit]);
    }
  });

  it("works for the picture decoder too", () => {
    const model = getModel("picture-city");
    const disguised = shuffled(model, seededRng(3));
    expect(Array.from(matVec(disguised.weight, disguised.input(5)).subarray(0, 400))).toEqual(Array.from(matVec(model.weight, model.input(5)).subarray(0, 400)));
  });
});

describe("three sealed layers through three relays", () => {
  const piece = Int32Array.from([1234, 65000, 7, 42, 31999, 2]);
  const plain = Array.from(new Uint8Array(new Uint16Array(piece).buffer));

  it("shows each relay different bytes, none of them the piece, and delivers the piece at the end", async () => {
    const route = await sealRoute(piece);
    expect(route.views).toHaveLength(3);
    // One layer (a 16-byte tag) comes off at every relay.
    expect(route.views.map((view) => view.length)).toEqual([plain.length + 48, plain.length + 32, plain.length + 16]);
    for (const view of route.views) expect(Array.from(view.subarray(0, plain.length))).not.toEqual(plain);
    expect(Array.from(route.views[0]!.subarray(0, 12))).not.toEqual(Array.from(route.views[1]!.subarray(0, 12)));
    expect(route.failedAt).toBeNull();
    expect(Array.from(route.delivered!)).toEqual(plain);
    const again = await sealRoute(piece);
    expect(Array.from(again.views[0]!)).not.toEqual(Array.from(route.views[0]!));
  });

  it("stops at the relay that receives an altered packet", async () => {
    for (const hop of [0, 1, 2]) {
      const route = await sealRoute(piece, hop);
      expect(route.failedAt).toBe(hop);
      expect(route.delivered).toBeNull();
      expect(route.views).toHaveLength(hop + 1);
    }
  });
});

describe("enclave reports", () => {
  it("accepts a signed report for the expected software and a fresh nonce", async () => {
    const enclave = await makeEnclave();
    const nonce = crypto.getRandomValues(new Uint8Array(16));
    const report = await attest(enclave, nonce);
    expect(report.measurement).toHaveLength(64);
    expect(await verifyReport(report, enclave.publicKey, nonce)).toEqual({ signed: true, expected: true, ok: true });
  });

  it("refuses a report from swapped software, a replayed nonce, a forged signature or another enclave's key", async () => {
    const enclave = await makeEnclave();
    const other = await makeEnclave();
    const nonce = crypto.getRandomValues(new Uint8Array(16));
    const swapped = await attest(enclave, nonce, `${WORKER_CODE} + a logger`);
    expect(await verifyReport(swapped, enclave.publicKey, nonce)).toEqual({ signed: true, expected: false, ok: false });
    const honest = await attest(enclave, nonce);
    expect((await verifyReport(honest, enclave.publicKey, crypto.getRandomValues(new Uint8Array(16)))).ok).toBe(false);
    expect((await verifyReport(honest, other.publicKey, nonce)).ok).toBe(false);
    const forged = { ...honest, signature: honest.signature.map((byte, index) => (index === 3 ? byte ^ 1 : byte)) };
    expect((await verifyReport(forged, enclave.publicKey, nonce)).signed).toBe(false);
    // Claiming the right measurement without the software to back it changes what was signed.
    const lying = { ...swapped, measurement: honest.measurement };
    expect((await verifyReport(lying, enclave.publicKey, nonce)).ok).toBe(false);
  });
});
