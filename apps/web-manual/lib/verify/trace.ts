/**
 * Check a Xurcroh task trace entirely in the browser.
 *
 * This is a second implementation of `xurcroh verify` (`python/xurcroh/receipts.py`).
 * It runs on the reader's machine; the trace, prompt and output are never sent anywhere.
 */

import {
  ALGORITHM,
  concat,
  equalBytes,
  fromHex,
  leafHash,
  sha256Hex,
  utf8,
  verifyHybrid,
  verifyInclusion,
} from "./crypto";
import {
  JsonNumber,
  type JsonObject,
  type JsonValue,
  canonicalBytes,
  isObject,
  parseJson,
  toDisplay,
  without,
} from "./json";

export const TRACE_FORMAT = "xurcroh-task-trace-v2";
const MANIFEST_FIELDS = [
  "repository",
  "revision",
  "source_format",
  "source_digest",
  "layout_digest",
  "architecture",
  "encodings",
  "share_count",
  "weight_shares",
  "weight_bits",
  "activation_bits",
  "reap_keep_fraction",
  "role_artifacts",
];

export class TraceError extends Error {}

export type LogReport = { treeSize: string; witnesses: string[] };

export type TraceReport = {
  signature: true;
  /** SHA-256 of the key that signed the trace. */
  signer: string;
  /** False when no expected key was given: the trace then only vouches for itself. */
  signerPinned: boolean;
  prompt: true | null;
  output: true | null;
  workerCheckpoints: number | null;
  log: LogReport | null;
  mode: string | null;
  assurance: string | null;
  facts: unknown;
};

export type VerifyOptions = {
  /** Hex key the trace must be signed by. */
  publicKey?: string;
  /** Chat messages as JSON text, to check the trace refers to them. */
  messages?: string;
  /** Image or video request (`prompt`, `image`, `mask`) as JSON text. */
  media?: string;
  output?: string;
  /** Inclusion proof as JSON text; needs `logKey`. */
  proof?: string;
  logKey?: string;
  /** Hex witness keys; every one must have countersigned the proof's tree head. */
  witnessKeys?: string[];
};

function text(value: JsonValue | undefined): string | null {
  return typeof value === "string" ? value : null;
}

function member(value: JsonValue | undefined, name: string): JsonValue | undefined {
  return isObject(value) ? value.get(name) : undefined;
}

function opens(commitment: JsonValue | undefined, payload: Uint8Array): boolean {
  try {
    const salt = fromHex(member(commitment, "salt"));
    return sha256Hex(salt, payload) === member(commitment, "sha256");
  } catch {
    return false;
  }
}

function normalizedMessages(source: string): Uint8Array {
  const messages = parseJson(source);
  if (!Array.isArray(messages)) throw new TraceError("messages must be a JSON list");
  for (const message of messages) {
    const content = member(message, "content");
    for (const part of Array.isArray(content) ? content : []) {
      // The request schema fills in the one default it has before the prompt is committed.
      const audio = member(part, "input_audio");
      if (isObject(audio) && !audio.has("format")) audio.set("format", "wav");
    }
  }
  return canonicalBytes(messages, false);
}

function mediaPayload(source: string): Uint8Array {
  const media = parseJson(source);
  if (!isObject(media)) throw new TraceError("media input must be a JSON object");
  const selected: JsonObject = new Map(
    ["prompt", "image", "mask"].map((name) => [name, media.get(name) ?? null]),
  );
  return canonicalBytes(selected, false);
}

function verifyManifest(document: JsonObject): void {
  const model = document.get("model");
  if (!isObject(model)) return;
  const manifest: JsonObject = new Map(
    MANIFEST_FIELDS.map((name) => [name, model.get(name) ?? null]),
  );
  const expected = sha256Hex(utf8("xurcroh-model-manifest-v1"), canonicalBytes(manifest, false));
  if (model.get("manifest_sha256") !== expected) {
    throw new TraceError("trace model manifest digest does not match its model facts");
  }
}

function verifyCheckpoints(document: JsonObject): number | null {
  const checkpoints = member(document.get("transport"), "worker_checkpoints");
  if (checkpoints === undefined || checkpoints === null) return null;
  if (!Array.isArray(checkpoints)) throw new TraceError("worker checkpoints are malformed");
  const pinned = member(document.get("model"), "role_artifacts");
  for (const checkpoint of checkpoints) {
    const worker = text(member(checkpoint, "worker"));
    const body = member(checkpoint, "body");
    let valid = false;
    try {
      const identity = fromHex(member(checkpoint, "identity_public"));
      valid =
        isObject(body) &&
        body.get("format") === "xurcroh-worker-checkpoint-v1" &&
        body.get("identity") === sha256Hex(identity) &&
        verifyHybrid(
          identity,
          "xurcroh-worker-checkpoint-v1",
          canonicalBytes(body, true),
          fromHex(member(checkpoint, "signature")),
        );
    } catch {
      valid = false;
    }
    if (!valid || worker === null) {
      throw new TraceError(`worker checkpoint signature does not verify (${worker ?? "unnamed"})`);
    }
    const expected = member(pinned, worker);
    if (expected !== undefined && expected !== null && member(body, "artifacts") !== expected) {
      throw new TraceError(
        `worker ${worker} signed for artifacts other than the ones in the model manifest`,
      );
    }
  }
  return checkpoints.length;
}

function verifyProof(document: JsonObject, options: VerifyOptions): LogReport {
  if (!options.logKey) throw new TraceError("a log proof needs the log's public key");
  const fail = (reason: string) => new TraceError(`trace is not in the log: ${reason}`);
  let proof: JsonValue;
  let logKey: Uint8Array;
  try {
    proof = parseJson(options.proof as string);
    logKey = fromHex(options.logKey.trim());
  } catch {
    throw fail("the proof or log key is malformed");
  }
  const head = member(proof, "tree_head");
  if (!isObject(head) || !isObject(proof)) throw fail("the proof is malformed");
  const size = head.get("tree_size");
  const index = proof.get("leaf_index");
  const path = proof.get("path");
  try {
    if (
      head.get("format") !== "xurcroh-tree-head-v2" ||
      !equalBytes(fromHex(head.get("log_key")), logKey) ||
      !verifyHybrid(
        logKey,
        "xurcroh-tree-head",
        canonicalBytes(without(head, "signature", "cosignatures"), true),
        fromHex(head.get("signature")),
      )
    ) {
      throw fail("tree head signature is missing or does not verify");
    }
    if (!(size instanceof JsonNumber) || !(index instanceof JsonNumber) || !Array.isArray(path)) {
      throw fail("the proof is malformed");
    }
    const digest = sha256Hex(canonicalBytes(document, false));
    const leaf = leafHash(concat(utf8("trace:"), fromHex(digest)));
    const included = verifyInclusion(
      leaf,
      BigInt(index.text),
      BigInt(size.text),
      path.map(fromHex),
      fromHex(head.get("root_hash")),
    );
    if (!included) throw fail("inclusion proof does not lead to the tree root");
  } catch (error) {
    if (error instanceof TraceError) throw error;
    throw fail("the proof is malformed");
  }
  const statement: JsonObject = new Map(
    ["log_key", "tree_size", "root_hash"].map((name) => [name, head.get(name) as JsonValue]),
  );
  const witnessed = canonicalBytes(statement, true);
  const cosignatures = head.get("cosignatures");
  const witnesses: string[] = [];
  for (const wanted of options.witnessKeys ?? []) {
    const key = wanted.trim().toLowerCase();
    const found = (Array.isArray(cosignatures) ? cosignatures : []).some((cosignature) => {
      try {
        return (
          member(cosignature, "witness_key") === key &&
          verifyHybrid(
            fromHex(key),
            "xurcroh-log-witness",
            witnessed,
            fromHex(member(cosignature, "signature")),
          )
        );
      } catch {
        return false;
      }
    });
    if (!found) throw new TraceError("the log's tree head is not signed by the required witness");
    witnesses.push(sha256Hex(fromHex(key)));
  }
  return { treeSize: (size as JsonNumber).text, witnesses: witnesses.sort() };
}

/** Verify a trace given as JSON text. Throws `TraceError` naming the part that failed. */
export function verifyTrace(source: string, options: VerifyOptions = {}): TraceReport {
  let document: JsonValue;
  try {
    document = parseJson(source);
  } catch (error) {
    throw new TraceError(`trace is not valid JSON: ${(error as Error).message}`);
  }
  if (!isObject(document) || document.get("format") !== TRACE_FORMAT) {
    throw new TraceError(`trace format is not ${TRACE_FORMAT}`);
  }
  const signer = document.get("signer");
  let embedded: Uint8Array;
  let signerPinned = false;
  try {
    embedded = fromHex(member(signer, "public_key"));
    if (member(signer, "algorithm") !== ALGORITHM) throw new Error("unsupported algorithm");
    if (options.publicKey?.trim()) {
      signerPinned = true;
      if (!equalBytes(fromHex(options.publicKey.trim()), embedded)) {
        throw new Error("signer does not match the expected key");
      }
    }
    const body = canonicalBytes(without(document, "signature"), false);
    if (!verifyHybrid(embedded, "xurcroh-task-trace", body, fromHex(document.get("signature")))) {
      throw new Error("signature does not verify");
    }
  } catch {
    throw new TraceError("trace signature is missing or does not verify");
  }
  verifyManifest(document);
  const report: TraceReport = {
    signature: true,
    signer: sha256Hex(embedded),
    signerPinned,
    prompt: null,
    output: null,
    workerCheckpoints: verifyCheckpoints(document),
    log: null,
    mode: text(document.get("mode")),
    assurance: text(document.get("assurance")),
    facts: toDisplay(without(document, "signature", "signer")),
  };
  const messages = options.messages?.trim();
  const media = options.media?.trim();
  if (messages && media) {
    throw new TraceError("give chat messages or a media input to check the prompt, not both");
  }
  if (messages || media) {
    let payload: Uint8Array;
    try {
      payload = messages ? normalizedMessages(messages) : mediaPayload(media as string);
    } catch (error) {
      if (error instanceof TraceError) throw error;
      throw new TraceError("the prompt given is not valid JSON");
    }
    if (!opens(member(document.get("request"), "prompt"), payload)) {
      throw new TraceError("trace does not refer to this prompt");
    }
    report.prompt = true;
  }
  if (options.output !== undefined && options.output !== "") {
    if (!opens(member(document.get("result"), "output"), utf8(options.output))) {
      throw new TraceError("trace does not refer to this output");
    }
    report.output = true;
  }
  if (options.proof?.trim()) report.log = verifyProof(document, options);
  else if (options.witnessKeys?.length) {
    throw new TraceError("a witness key can only be checked together with a log proof");
  }
  return report;
}
