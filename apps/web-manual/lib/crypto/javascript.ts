import { gcm } from "@noble/ciphers/aes.js";
import { p256 } from "@noble/curves/nist.js";
import { sha256 } from "@noble/hashes/sha2.js";
import type { Bytes, KeyPair, Primitives, SecretKey } from "@/lib/crypto/primitives";

function bytesOf(secretKey: SecretKey): Bytes {
  if (!secretKey.bytes) throw new Error("This secret key was made by the browser and cannot be used here");
  return secretKey.bytes;
}

function pair(): KeyPair {
  const secret = p256.utils.randomSecretKey();
  return { publicKey: p256.getPublicKey(secret, false), secretKey: { bytes: secret } };
}

/** The same operations as the browser's, computed by JavaScript that ships with the page. */
export const javascriptPrimitives: Primitives = {
  name: "javascript",
  async sha256(data) {
    return sha256(data);
  },
  async aesSeal(key, nonce, plain) {
    if (key.length !== 32) throw new Error("An AES-256 key is 32 bytes");
    return gcm(key, nonce).encrypt(plain);
  },
  async aesOpen(key, nonce, sealed) {
    if (key.length !== 32) throw new Error("An AES-256 key is 32 bytes");
    return gcm(key, nonce).decrypt(sealed);
  },
  async ecdhPair() {
    return pair();
  },
  async ecdhShared(secretKey, peerPublicKey) {
    // The shared point, compressed: one byte for the sign, then the 32 bytes both sides use.
    return p256.getSharedSecret(bytesOf(secretKey), peerPublicKey, true).slice(1);
  },
  async ecdsaPair() {
    return pair();
  },
  async ecdsaSign(secretKey, message) {
    return p256.sign(message, bytesOf(secretKey));
  },
  async ecdsaVerify(publicKey, signature, message) {
    try {
      // The browser may produce either of the two equivalent forms of a signature; both are accepted.
      return p256.verify(signature, message, publicKey, { lowS: false });
    } catch {
      return false;
    }
  },
};
