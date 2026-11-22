/** SHA-256, hybrid Ed25519 + ML-DSA-65 verification and RFC 6962 Merkle proofs. */

import { ed25519 } from "@noble/curves/ed25519.js";
import { sha256 } from "@noble/hashes/sha2.js";
import { ml_dsa65 } from "@noble/post-quantum/ml-dsa.js";

export const ALGORITHM = "Ed25519+ML-DSA-65";
const ED25519_PUBLIC = 32;
const ED25519_SIGNATURE = 64;
const PUBLIC_KEY_BYTES = ED25519_PUBLIC + 1952;
const SIGNATURE_BYTES = ED25519_SIGNATURE + 3309;

const encoder = new TextEncoder();

export function utf8(text: string): Uint8Array {
  return encoder.encode(text);
}

export function concat(...parts: Uint8Array[]): Uint8Array {
  const result = new Uint8Array(parts.reduce((total, part) => total + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    result.set(part, offset);
    offset += part.length;
  }
  return result;
}

export function toHex(bytes: Uint8Array): string {
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

export function fromHex(text: unknown): Uint8Array {
  if (typeof text !== "string" || text.length % 2 !== 0 || !/^[0-9a-fA-F]*$/.test(text)) {
    throw new Error("expected hexadecimal text");
  }
  const bytes = new Uint8Array(text.length / 2);
  for (let index = 0; index < bytes.length; index += 1) {
    bytes[index] = parseInt(text.slice(2 * index, 2 * index + 2), 16);
  }
  return bytes;
}

export function sha256Hex(...parts: Uint8Array[]): string {
  return toHex(sha256(concat(...parts)));
}

export function equalBytes(left: Uint8Array, right: Uint8Array): boolean {
  if (left.length !== right.length) return false;
  let difference = 0;
  for (let index = 0; index < left.length; index += 1) difference |= (left[index] as number) ^ (right[index] as number);
  return difference === 0;
}

/** True only when both halves of a hybrid signature verify for this context. */
export function verifyHybrid(
  publicKey: Uint8Array,
  context: string,
  message: Uint8Array,
  signature: Uint8Array,
): boolean {
  if (publicKey.length !== PUBLIC_KEY_BYTES || signature.length !== SIGNATURE_BYTES) return false;
  const label = utf8(context);
  const length = new Uint8Array(4);
  new DataView(length.buffer).setUint32(0, label.length, false);
  const signed = concat(utf8("xurcroh-hybrid-signature-v1"), length, label, message);
  try {
    const classical = ed25519.verify(
      signature.subarray(0, ED25519_SIGNATURE),
      signed,
      publicKey.subarray(0, ED25519_PUBLIC),
    );
    const postQuantum = ml_dsa65.verify(
      signature.subarray(ED25519_SIGNATURE),
      signed,
      publicKey.subarray(ED25519_PUBLIC),
    );
    return classical && postQuantum;
  } catch {
    return false;
  }
}

export function leafHash(leaf: Uint8Array): Uint8Array {
  return sha256(concat(Uint8Array.of(0), leaf));
}

function node(left: Uint8Array, right: Uint8Array): Uint8Array {
  return sha256(concat(Uint8Array.of(1), left, right));
}

/** True when the hashed leaf is entry `index` of the tree of `size` with this root. */
export function verifyInclusion(
  leaf: Uint8Array,
  index: bigint,
  size: bigint,
  path: Uint8Array[],
  root: Uint8Array,
): boolean {
  if (index < 0n || index >= size) return false;
  let position = index;
  let last = size - 1n;
  let current = leaf;
  for (const sibling of path) {
    if (last === 0n) return false;
    if (position & 1n || position === last) {
      current = node(sibling, current);
      while (!(position & 1n) && position) {
        position >>= 1n;
        last >>= 1n;
      }
    } else {
      current = node(current, sibling);
    }
    position >>= 1n;
    last >>= 1n;
  }
  return last === 0n && equalBytes(current, root);
}
