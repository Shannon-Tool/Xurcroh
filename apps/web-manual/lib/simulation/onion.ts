import { primitives, randomBytes, type Bytes, type SecretKey } from "@/lib/crypto/primitives";

export interface OnionLayer {
  publicKey: Bytes;
  secretKey: SecretKey;
  nonce: Uint8Array;
  ciphertext: Uint8Array;
  nextHop: string;
  returnSecret: Uint8Array | null;
}

export interface OnionHop {
  nextHop: string;
}

export async function sealOnion(hops: OnionHop[], payload: Uint8Array): Promise<OnionLayer[]> {
  const crypt = await primitives();
  let inner = payload;
  const layers: OnionLayer[] = [];
  for (let index = hops.length - 1; index >= 0; index -= 1) {
    const pair = await crypt.ecdhPair();
    const shared = await crypt.ecdhShared(pair.secretKey, pair.publicKey);
    const nonce = randomBytes(12);
    const ciphertext = await crypt.aesSeal(shared, nonce, inner);
    layers.unshift({
      publicKey: pair.publicKey,
      secretKey: pair.secretKey,
      nonce,
      ciphertext,
      nextHop: hops[index]!.nextHop,
      returnSecret: index === hops.length - 1 ? new Uint8Array([7, 7, 7, 7]) : null,
    });
    inner = new TextEncoder().encode(JSON.stringify({ ciphertext: Array.from(ciphertext), nextHop: hops[index]!.nextHop }));
  }
  return layers;
}

export async function openOnion(layer: OnionLayer): Promise<Uint8Array> {
  const crypt = await primitives();
  const shared = await crypt.ecdhShared(layer.secretKey, layer.publicKey);
  return crypt.aesOpen(shared, layer.nonce, layer.ciphertext);
}

export class ReplayWindow {
  private readonly seen = new Set<number>();

  accept(sequence: number): boolean {
    if (this.seen.has(sequence)) return false;
    this.seen.add(sequence);
    return true;
  }
}
