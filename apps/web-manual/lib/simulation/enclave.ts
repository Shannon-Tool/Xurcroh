/**
 * A stand-in for a hardware enclave's attestation, for the manual only. The cryptography is real:
 * an ECDSA key signs a report over the hash of the worker's code and a fresh nonce, and the device
 * verifies the signature and compares the hash with the one it expects. What is NOT real is the
 * hardware: here the key lives in the page, whereas a real enclave's key is held by the processor
 * and vouched for by its maker. Hardware attestation of workers has not been validated yet.
 */

/** The software every worker is expected to run. Its hash is the expected measurement. */
export const WORKER_CODE = "xurcroh-worker: multiply one activation share by one weight share; return the product";

export interface Enclave {
  publicKey: CryptoKey;
  privateKey: CryptoKey;
}

export interface EnclaveReport {
  /** SHA-256 of the code the enclave says it runs, as hex. */
  measurement: string;
  signature: number[];
}

const encoder = new TextEncoder();

async function sha256Hex(text: string): Promise<string> {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", encoder.encode(text)));
  return Array.from(digest, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function signedBytes(measurement: string, nonce: Uint8Array): Uint8Array {
  const head = encoder.encode(measurement);
  const out = new Uint8Array(head.length + nonce.length);
  out.set(head);
  out.set(nonce, head.length);
  return out;
}

export async function makeEnclave(): Promise<Enclave> {
  const pair = await crypto.subtle.generateKey({ name: "ECDSA", namedCurve: "P-256" }, false, ["sign", "verify"]);
  return { publicKey: pair.publicKey, privateKey: pair.privateKey };
}

/** The enclave measures the code it runs and signs that measurement together with the device's nonce. */
export async function attest(enclave: Enclave, nonce: Uint8Array, code: string = WORKER_CODE): Promise<EnclaveReport> {
  const measurement = await sha256Hex(code);
  const signature = await crypto.subtle.sign({ name: "ECDSA", hash: "SHA-256" }, enclave.privateKey, signedBytes(measurement, nonce) as unknown as BufferSource);
  return { measurement, signature: Array.from(new Uint8Array(signature)) };
}

export interface ReportCheck {
  /** The signature is valid for this enclave's key, this measurement and this nonce. */
  signed: boolean;
  /** The measurement is the one the device expects. */
  expected: boolean;
  ok: boolean;
}

export async function verifyReport(report: EnclaveReport, publicKey: CryptoKey, nonce: Uint8Array): Promise<ReportCheck> {
  const signed = await crypto.subtle.verify(
    { name: "ECDSA", hash: "SHA-256" },
    publicKey,
    Uint8Array.from(report.signature) as unknown as BufferSource,
    signedBytes(report.measurement, nonce) as unknown as BufferSource,
  );
  const expected = report.measurement === await sha256Hex(WORKER_CODE);
  return { signed, expected, ok: signed && expected };
}
