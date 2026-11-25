/**
 * The four pieces of cryptography the manual's demonstrations run: SHA-256, AES-256-GCM, ECDH and
 * ECDSA on the P-256 curve. A browser only offers its own implementation (`crypto.subtle`) on HTTPS
 * and on localhost. On any other address the same operations are done by a JavaScript implementation
 * that ships with the page, so that a demonstration never has a step it cannot compute.
 *
 * Keys are plain bytes here (or, for a secret key made by the browser, the browser's own handle):
 * everything is a teaching simulation inside one page, and the demonstrations need the bytes to show
 * what a changed key does.
 */

export type Bytes = Uint8Array;

/** A secret key: the browser's handle when the browser made it, the bytes when the page did. */
export interface SecretKey {
  handle?: CryptoKey;
  bytes?: Bytes;
}

export interface KeyPair {
  /** The public point, uncompressed: 65 bytes. */
  publicKey: Bytes;
  secretKey: SecretKey;
}

export interface Primitives {
  name: "browser" | "javascript";
  sha256(data: Bytes): Promise<Bytes>;
  /** AES-256-GCM. Returns the sealed bytes followed by the 16-byte authenticity tag. */
  aesSeal(key: Bytes, nonce: Bytes, plain: Bytes): Promise<Bytes>;
  /** Rejects when the key is wrong or a single bit of the sealed bytes was changed. */
  aesOpen(key: Bytes, nonce: Bytes, sealed: Bytes): Promise<Bytes>;
  ecdhPair(): Promise<KeyPair>;
  /** The 32 bytes both sides arrive at. */
  ecdhShared(secretKey: SecretKey, peerPublicKey: Bytes): Promise<Bytes>;
  ecdsaPair(): Promise<KeyPair>;
  /** ECDSA over SHA-256 of the message; the signature is r followed by s, 64 bytes. */
  ecdsaSign(secretKey: SecretKey, message: Bytes): Promise<Bytes>;
  ecdsaVerify(publicKey: Bytes, signature: Bytes, message: Bytes): Promise<boolean>;
}

/** Random bytes from the browser. Unlike `crypto.subtle`, this is available on every address. */
export function randomBytes(length: number): Bytes {
  return crypto.getRandomValues(new Uint8Array(length));
}

const source = (bytes: Bytes) => bytes as unknown as BufferSource;
const P256 = { namedCurve: "P-256" } as const;

async function browserPair(name: "ECDH" | "ECDSA"): Promise<KeyPair> {
  const pair = await crypto.subtle.generateKey({ name, ...P256 }, true, name === "ECDH" ? ["deriveBits"] : ["sign", "verify"]);
  return { publicKey: new Uint8Array(await crypto.subtle.exportKey("raw", pair.publicKey)), secretKey: { handle: pair.privateKey } };
}

async function aesHandle(key: Bytes, use: "encrypt" | "decrypt"): Promise<CryptoKey> {
  if (key.length !== 32) throw new Error("An AES-256 key is 32 bytes");
  return crypto.subtle.importKey("raw", source(key), "AES-GCM", false, [use]);
}

function handleOf(secretKey: SecretKey): CryptoKey {
  if (!secretKey.handle) throw new Error("This secret key was not made by the browser");
  return secretKey.handle;
}

/** The browser's own implementation. */
export const browserPrimitives: Primitives = {
  name: "browser",
  async sha256(data) {
    return new Uint8Array(await crypto.subtle.digest("SHA-256", source(data)));
  },
  async aesSeal(key, nonce, plain) {
    return new Uint8Array(await crypto.subtle.encrypt({ name: "AES-GCM", iv: source(nonce) }, await aesHandle(key, "encrypt"), source(plain)));
  },
  async aesOpen(key, nonce, sealed) {
    return new Uint8Array(await crypto.subtle.decrypt({ name: "AES-GCM", iv: source(nonce) }, await aesHandle(key, "decrypt"), source(sealed)));
  },
  ecdhPair: () => browserPair("ECDH"),
  async ecdhShared(secretKey, peerPublicKey) {
    const peer = await crypto.subtle.importKey("raw", source(peerPublicKey), { name: "ECDH", ...P256 }, false, []);
    return new Uint8Array(await crypto.subtle.deriveBits({ name: "ECDH", public: peer }, handleOf(secretKey), 256));
  },
  ecdsaPair: () => browserPair("ECDSA"),
  async ecdsaSign(secretKey, message) {
    return new Uint8Array(await crypto.subtle.sign({ name: "ECDSA", hash: "SHA-256" }, handleOf(secretKey), source(message)));
  },
  async ecdsaVerify(publicKey, signature, message) {
    const key = await crypto.subtle.importKey("raw", source(publicKey), { name: "ECDSA", ...P256 }, false, ["verify"]);
    return crypto.subtle.verify({ name: "ECDSA", hash: "SHA-256" }, key, source(signature), source(message));
  },
};

let loading: Promise<Primitives> | null = null;

/** The implementation that ships with the page; loaded only when it is needed. */
export function javascriptPrimitives(): Promise<Primitives> {
  loading ??= import("@/lib/crypto/javascript").then((module) => module.javascriptPrimitives);
  return loading;
}

/** The browser's implementation when this page has one, the page's own otherwise. */
export async function primitives(): Promise<Primitives> {
  return typeof crypto !== "undefined" && crypto.subtle ? browserPrimitives : javascriptPrimitives();
}
