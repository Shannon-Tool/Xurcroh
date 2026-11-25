import { primitives, randomBytes } from "@/lib/crypto/primitives";
import { journeyOf, questionIds } from "@/lib/show/journey";
import { PRIME, mod } from "@/lib/simulation/fortress";

/**
 * Facts about how hard the hidden data is to recover, each one computed by the page: nothing here is
 * a rating or an adjective. Every fact comes with the place where it stops.
 */

/** Bits that differ between two byte strings of the same length. */
export function bitsChanged(a: Uint8Array, b: Uint8Array): number {
  let count = 0;
  for (let index = 0; index < Math.min(a.length, b.length); index += 1) {
    let diff = a[index]! ^ b[index]!;
    while (diff) {
      count += diff & 1;
      diff >>= 1;
    }
  }
  return count;
}

/** A whole number as "1.2 × 10⁷⁷". */
export function scientific(value: bigint): string {
  const digits = value.toString();
  const exponent = digits.length - 1;
  const rounded = (Number(digits.padEnd(4, "0").slice(0, 4)) / 1000).toFixed(1);
  const up = (text: string) => text.replace(/[0-9]/g, (digit) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[Number(digit)]!);
  return `${rounded} × 10${up(String(exponent))}`;
}

/** How many 256-bit keys there are, and how long half of them take at a stated rate of guessing. */
export const KEYS = 1n << 256n;
export const GUESSES_PER_SECOND = 10n ** 18n;
const SECONDS_PER_YEAR = 31_557_600n;
export const YEARS_FOR_HALF = KEYS / 2n / GUESSES_PER_SECOND / SECONDS_PER_YEAR;

export interface PieceFit {
  question: string;
  /** The other piece that, added to the worker's piece, gives this question's number at that position. */
  other: bigint;
  /** That question's number at the position: what the two pieces add up to. */
  value: number;
}

/**
 * One worker holds `piece` for the position of a word. For every prepared question there is another
 * piece that makes the worker's piece fit that question, so the worker's piece cannot tell them apart.
 */
export function everyQuestionFits(piece: bigint, position: number): PieceFit[] {
  return questionIds().map((id) => {
    const journey = journeyOf(id);
    const value = journey.counts[position] ?? 0;
    return { question: journey.question, other: mod(BigInt(value) - piece), value };
  });
}

export const POSSIBLE_PIECES = PRIME;

export interface SealFacts {
  plain: Uint8Array;
  sealed: Uint8Array;
  key: Uint8Array;
  nonce: Uint8Array;
  /** Sealed again with one bit of the key changed: how many of the sealed bits came out different. */
  keyBitChanged: number;
  sealedBits: number;
  /** One bit of the sealed bytes changed: does it still open with the right key? */
  alteredOpens: boolean;
}

/** Seals `plain` with a fresh AES-256-GCM key and measures what a one-bit change does. */
export async function sealFacts(plain: Uint8Array): Promise<SealFacts> {
  const crypt = await primitives();
  const key = randomBytes(32);
  const nonce = randomBytes(12);
  const sealed = await crypt.aesSeal(key, nonce, plain);
  const nearKey = Uint8Array.from(key);
  nearKey[0]! ^= 1;
  const nearSealed = await crypt.aesSeal(nearKey, nonce, plain);
  const altered = Uint8Array.from(sealed);
  altered[0]! ^= 1;
  const alteredOpens = await crypt.aesOpen(key, nonce, altered).then(() => true, () => false);
  return { plain, sealed, key, nonce, keyBitChanged: bitsChanged(sealed, nearSealed), sealedBits: sealed.length * 8, alteredOpens };
}

/** Tries `count` random wrong keys against a sealed packet; answers how many opened it. */
export async function tryWrongKeys(facts: Pick<SealFacts, "sealed" | "nonce">, count: number): Promise<number> {
  const crypt = await primitives();
  let opened = 0;
  for (let attempt = 0; attempt < count; attempt += 1) {
    if (await crypt.aesOpen(randomBytes(32), facts.nonce, facts.sealed).then(() => true, () => false)) opened += 1;
  }
  return opened;
}

export interface FingerprintFacts {
  before: string;
  after: string;
  bitsChanged: number;
}

/** The SHA-256 fingerprint of a piece of software, and of the same software with one character changed. */
export async function fingerprintFacts(code: string): Promise<FingerprintFacts> {
  const crypt = await primitives();
  const encoder = new TextEncoder();
  const before = await crypt.sha256(encoder.encode(code));
  const after = await crypt.sha256(encoder.encode(`${code.slice(0, -1)}${code.endsWith("t") ? "s" : "t"}`));
  const hex = (bytes: Uint8Array) => Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
  return { before: hex(before), after: hex(after), bitsChanged: bitsChanged(before, after) };
}
