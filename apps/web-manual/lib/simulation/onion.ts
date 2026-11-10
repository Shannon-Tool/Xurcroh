export interface OnionLayer {
  publicKey: CryptoKey;
  privateKey: CryptoKey;
  nonce: Uint8Array;
  ciphertext: Uint8Array;
  nextHop: string;
  returnSecret: Uint8Array | null;
}

export interface OnionHop {
  nextHop: string;
}

async function deriveSharedKey(publicKey: CryptoKey, privateKey: CryptoKey): Promise<CryptoKey> {
  const bits = await crypto.subtle.deriveBits({ name: "ECDH", public: publicKey }, privateKey, 256);
  return crypto.subtle.importKey("raw", bits, "AES-GCM", false, ["encrypt", "decrypt"]);
}

export async function sealOnion(hops: OnionHop[], payload: Uint8Array): Promise<OnionLayer[]> {
  let inner = payload;
  const layers: OnionLayer[] = [];
  for (let index = hops.length - 1; index >= 0; index -= 1) {
    const pair = await crypto.subtle.generateKey({ name: "ECDH", namedCurve: "P-256" }, true, ["deriveBits"]);
    const shared = await deriveSharedKey(pair.publicKey, pair.privateKey);
    const nonce = crypto.getRandomValues(new Uint8Array(12));
    const ciphertext = await crypto.subtle.encrypt(
      { name: "AES-GCM", iv: nonce },
      shared,
      inner as unknown as BufferSource,
    );
    layers.unshift({
      publicKey: pair.publicKey,
      privateKey: pair.privateKey,
      nonce,
      ciphertext: new Uint8Array(ciphertext),
      nextHop: hops[index]!.nextHop,
      returnSecret: index === hops.length - 1 ? new Uint8Array([7, 7, 7, 7]) : null,
    });
    inner = new TextEncoder().encode(JSON.stringify({ ciphertext: Array.from(new Uint8Array(ciphertext)), nextHop: hops[index]!.nextHop }));
  }
  return layers;
}

export async function openOnion(layer: OnionLayer): Promise<Uint8Array> {
  const shared = await deriveSharedKey(layer.publicKey, layer.privateKey);
  const plaintext = await crypto.subtle.decrypt(
    { name: "AES-GCM", iv: layer.nonce as unknown as BufferSource },
    shared,
    layer.ciphertext as unknown as BufferSource,
  );
  return new Uint8Array(plaintext);
}

export class ReplayWindow {
  private readonly seen = new Set<number>();

  accept(sequence: number): boolean {
    if (this.seen.has(sequence)) return false;
    this.seen.add(sequence);
    return true;
  }
}
