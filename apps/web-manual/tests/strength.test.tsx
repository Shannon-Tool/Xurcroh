import { act, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { StrengthPanel } from "@/components/StrengthPanel";
import { primitives } from "@/lib/crypto/primitives";
import { journeyOf, questionIds } from "@/lib/show/journey";
import {
  GUESSES_PER_SECOND, KEYS, YEARS_FOR_HALF, bitsChanged, everyQuestionFits, fingerprintFacts, scientific, sealFacts, tryWrongKeys,
} from "@/lib/show/strength";
import { WORKER_CODE } from "@/lib/simulation/enclave";
import { executeShareGrid, mod, PRIME } from "@/lib/simulation/fortress";

const shares = journeyOf("text-shares");
const BANNED = /\b(guarantee|unbreakable|impossible|100%|perfectly safe|military|state of the art|uncrackable|bulletproof)\b/i;

afterEach(() => undefined);

describe("facts about undoing what a stage hides", () => {
  it("counts changed bits and writes large numbers as powers of ten", () => {
    expect(bitsChanged(Uint8Array.from([0b1010, 0xff]), Uint8Array.from([0b0110, 0x0f]))).toBe(2 + 4);
    expect(scientific(KEYS)).toBe("1.2 × 10⁷⁷");
    expect(scientific(GUESSES_PER_SECOND)).toBe("1.0 × 10¹⁸");
    // Half of 2²⁵⁶ keys at 10¹⁸ a second, in years of 365.25 days.
    expect(YEARS_FOR_HALF).toBe((1n << 255n) / 10n ** 18n / 31_557_600n);
    expect(scientific(YEARS_FOR_HALF)).toBe("1.8 × 10⁵¹");
  });

  it("finds, for one worker's piece, another piece that fits every prepared question", () => {
    const grid = executeShareGrid(shares.weightRows, shares.counts, 0);
    const position = shares.counts.findIndex((count) => count > 0);
    const piece = grid.activationShares[0]![position]!;
    const fits = everyQuestionFits(piece, position);
    expect(fits.map((fit) => fit.question)).toEqual(questionIds().map((id) => journeyOf(id).question));
    for (const [index, fit] of fits.entries()) {
      // The worker's piece plus that other piece really is that question's number.
      expect(mod(piece + fit.other)).toBe(BigInt(journeyOf(questionIds()[index]!).counts[position]!));
      expect(fit.other >= 0n && fit.other < PRIME).toBe(true);
    }
    // The real other piece is the one listed for the real question.
    expect(fits[questionIds().indexOf("text-shares")]!.other).toBe(grid.activationShares[1]![position]);
    // Questions that differ at that position need different other pieces; the worker cannot see which exists.
    expect(new Set(fits.map((fit) => fit.other)).size).toBe(new Set(fits.map((fit) => fit.value)).size);
  });

  it("seals bytes, and measures what one changed bit of the key or of the sealed bytes does", async () => {
    const plain = new TextEncoder().encode(shares.question);
    const facts = await sealFacts(plain);
    expect(facts.sealed).toHaveLength(plain.length + 16);
    expect(facts.sealedBits).toBe((plain.length + 16) * 8);
    // An unrelated key changes about half the bits; one bit of the key does the same.
    expect(facts.keyBitChanged).toBeGreaterThan(facts.sealedBits * 0.35);
    expect(facts.keyBitChanged).toBeLessThan(facts.sealedBits * 0.65);
    expect(facts.alteredOpens).toBe(false);
    const crypt = await primitives();
    expect(new TextDecoder().decode(await crypt.aesOpen(facts.key, facts.nonce, facts.sealed))).toBe(shares.question);
    expect(await tryWrongKeys(facts, 40)).toBe(0);
  });

  it("shows that one changed character changes about half of a fingerprint", async () => {
    const facts = await fingerprintFacts(WORKER_CODE);
    expect(facts.before).toHaveLength(64);
    expect(facts.after).not.toBe(facts.before);
    expect(facts.bitsChanged).toBeGreaterThan(90);
    expect(facts.bitsChanged).toBeLessThan(166);
  });
});

describe("the panel under a stage", () => {
  const panel = () => screen.getByRole("group", { name: "How hard is this to undo?" });

  it("random pieces: lists a fitting other piece for every prepared question, and says where that stops", () => {
    const grid = executeShareGrid(shares.weightRows, shares.counts, 0);
    const position = shares.counts.findIndex((count) => count > 0);
    render(<StrengthPanel spec={{ kind: "pieces", piece: grid.activationShares[0]![position]!, position, word: shares.vocabulary[position]!, holder: "Worker (0,0)" }} />);
    expect(within(panel()).getAllByRole("row")).toHaveLength(questionIds().length + 1);
    expect(panel()).toHaveTextContent(`All ${questionIds().length} fit`);
    expect(panel()).toHaveTextContent("Whoever holds both matching pieces adds them and reads the value.");
    expect(panel()).toHaveTextContent("non-collusion condition is part of its security model");
    expect(panel().textContent).not.toMatch(BANNED);
  });

  it("sealed bytes: shows readable beside sealed, really tries wrong keys, and says that guessing is not how it fails", async () => {
    render(<StrengthPanel spec={{ kind: "seal", plain: new TextEncoder().encode(shares.question), layers: 3, what: "The packet" }} />);
    await waitFor(() => expect(Number(screen.getByTestId("keys-tried").textContent)).toBeGreaterThanOrEqual(64));
    expect(screen.getByTestId("keys-opened")).toHaveTextContent("0");
    expect(panel()).toHaveTextContent("48 6f 77 20");
    expect(panel()).toHaveTextContent("2²⁵⁶ ≈ 1.2 × 10⁷⁷ possible keys");
    expect(panel()).toHaveTextContent("about 1.8 × 10⁵¹ years");
    expect(panel()).toHaveTextContent("refused to open them");
    expect(panel()).toHaveTextContent("Guessing the key is not how such protection fails.");
    expect(panel()).toHaveTextContent("the relays stand in for a Tor circuit inside the page");
    expect(panel()).toHaveTextContent("production engine is hardware-validated");
    expect(panel().textContent).not.toMatch(BANNED);
    await act(async () => undefined);
  });

  it("worker reports: shows two fingerprints and keeps the limits of enclaves in view", async () => {
    render(<StrengthPanel spec={{ kind: "report", code: WORKER_CODE }} />);
    await waitFor(() => expect(panel()).toHaveTextContent(/changed \d+ of the 256 bits/));
    expect(panel()).toHaveTextContent("an enclave is one more layer, not proof");
    expect(panel()).toHaveTextContent("has not yet been validated on real hardware");
    expect(panel().textContent).not.toMatch(BANNED);
  });

  it("disguises and wiped memory: say plainly what they are", () => {
    const { unmount } = render(<StrengthPanel spec={{ kind: "disguise", what: "The worker gets your numbers turned by an angle it does not know." }} />);
    expect(panel()).toHaveTextContent("This is a disguise, not encryption.");
    expect(panel()).toHaveTextContent("Nothing in the design relies on it");
    unmount();
    const wiped = render(<StrengthPanel spec={{ kind: "wipe", before: [72, 111, 119, 32], after: [0, 0, 0, 0] }} />);
    expect(panel()).toHaveTextContent("48 6f 77 20");
    expect(panel()).toHaveTextContent("00 00 00 00");
    expect(panel()).toHaveTextContent("does not prove that every register and cache inside the chip was cleared");
    wiped.unmount();
    render(<StrengthPanel spec={{ kind: "wipe", before: [72, 111, 119, 32], after: [72, 111, 119, 32] }} />);
    expect(panel()).toHaveTextContent("The wipe was skipped, so the next user reads your bytes as they were.");
  });
});
