"use client";

import { useEffect, useState } from "react";
import { grouped, hexText } from "@/lib/show/process";
import {
  GUESSES_PER_SECOND, KEYS, POSSIBLE_PIECES, YEARS_FOR_HALF, everyQuestionFits, fingerprintFacts, scientific, sealFacts, tryWrongKeys,
  type FingerprintFacts, type SealFacts,
} from "@/lib/show/strength";

/** What a stage hides, and with what. */
export type StrengthSpec =
  | { kind: "pieces"; piece: bigint; position: number; word: string; holder: string }
  | { kind: "seal"; plain: Uint8Array; layers: number; what: string }
  | { kind: "report"; code: string }
  | { kind: "disguise"; what: string; ways?: string }
  | { kind: "wipe"; before: number[]; after: number[] };

const hex = (bytes: ArrayLike<number>) => Array.from(bytes, (value) => (value & 255).toString(16).padStart(2, "0")).join(" ");

function Pieces({ spec }: { spec: Extract<StrengthSpec, { kind: "pieces" }> }) {
  const fits = everyQuestionFits(spec.piece, spec.position);
  return (
    <>
      <p>{spec.holder} holds <code>{grouped(spec.piece)}</code> for the word “{spec.word}”. In the engine that number is drawn at random from {grouped(POSSIBLE_PIECES)} equally likely values, whatever your question is. (This lab uses a repeatable generator, so that it shows the same numbers every time.)</p>
      <p>Could the holder work out which question it came from? For every prepared question there is another piece that makes this same number fit:</p>
      <table className="strength-table">
        <thead><tr><th>If the question were</th><th>the other piece would be</th><th>and the two add up to</th></tr></thead>
        <tbody>
          {fits.map((fit) => <tr key={fit.question}><td>{fit.question}</td><td><code>{grouped(fit.other)}</code></td><td>{fit.value} (“{spec.word}” {fit.value ? "is in it" : "is not in it"})</td></tr>)}
        </tbody>
      </table>
      <p>All {fits.length} fit, so the one number says nothing about which is true. This is arithmetic, not an estimate: it holds for anyone who has only this piece.</p>
      <p className="strength-limit"><strong>Where it stops.</strong> Whoever holds both matching pieces adds them and reads the value. The design relies on such holders not pooling what they have; claims are under evaluation.</p>
    </>
  );
}

function Seal({ spec }: { spec: Extract<StrengthSpec, { kind: "seal" }> }) {
  const [facts, setFacts] = useState<SealFacts | null>(null);
  const [tried, setTried] = useState({ keys: 0, opened: 0 });
  const [failure, setFailure] = useState<string | null>(null);
  useEffect(() => {
    let current = true;
    setFacts(null);
    setTried({ keys: 0, opened: 0 });
    void sealFacts(spec.plain).then((result) => current && setFacts(result)).catch((error: unknown) => current && setFailure(String(error)));
    return () => {
      current = false;
    };
  }, [spec.plain]);
  // The page really tries wrong keys against the packet shown, a few at a time, for as long as it is open.
  useEffect(() => {
    if (!facts) return;
    let current = true;
    const still = typeof window.matchMedia !== "function" || window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const round = (count: number) => tryWrongKeys(facts, count).then((opened) => current && setTried((before) => ({ keys: before.keys + count, opened: before.opened + opened })));
    void round(still ? 64 : 8);
    if (still) return () => { current = false; };
    const timer = window.setInterval(() => { if (!document.hidden) void round(8); }, 500);
    return () => {
      current = false;
      window.clearInterval(timer);
    };
  }, [facts]);
  if (failure) return <p role="alert">This could not be computed in this browser: {failure}</p>;
  if (!facts) return <p>Sealing…</p>;
  return (
    <>
      <dl className="strength-pair">
        <div><dt>Readable</dt><dd><code>{hexText(facts.plain, 12)}</code></dd></div>
        <div><dt>Sealed (AES-256-GCM)</dt><dd><code className="stage-cipher">{hexText(facts.sealed, 12)}</code></dd></div>
      </dl>
      <ul className="strength-facts">
        <li><strong>Guessing the key.</strong> The key is 256 bits, so there are 2²⁵⁶ ≈ {scientific(KEYS)} possible keys. Since you opened this page it has tried <strong data-testid="keys-tried">{grouped(tried.keys)}</strong> wrong keys against the sealed bytes above; <strong data-testid="keys-opened">{tried.opened}</strong> opened them. At {scientific(GUESSES_PER_SECOND)} guesses a second, trying half the keys would take about {scientific(YEARS_FOR_HALF)} years.</li>
        <li><strong>A nearly right key is no help.</strong> Sealed again with one bit of the key changed, {facts.keyBitChanged} of the {facts.sealedBits} sealed bits came out different (about half, as for an unrelated key).</li>
        <li><strong>Changing the bytes is noticed.</strong> With one bit of the sealed bytes flipped, the right key {facts.alteredOpens ? "still opened them" : "refused to open them"}: the 16-byte authenticity tag no longer matched.</li>
        {spec.layers > 1 ? <li><strong>{spec.layers} layers.</strong> {spec.what} is sealed {spec.layers} times with {spec.layers} different keys; each relay holds one of them.</li> : null}
      </ul>
      <p className="strength-limit"><strong>Where it stops.</strong> Guessing the key is not how such protection fails. Keys can be stolen, software can be faulty, and a device can be taken over; sealing hides what is sent, not how much or when. Here the relays stand in for a Tor circuit inside the page. Designed for defence in depth; claims under evaluation.</p>
    </>
  );
}

function Report({ spec }: { spec: Extract<StrengthSpec, { kind: "report" }> }) {
  const [facts, setFacts] = useState<FingerprintFacts | null>(null);
  useEffect(() => {
    let current = true;
    void fingerprintFacts(spec.code).then((result) => current && setFacts(result)).catch(() => undefined);
    return () => {
      current = false;
    };
  }, [spec.code]);
  if (!facts) return <p>Computing the fingerprint…</p>;
  return (
    <>
      <dl className="strength-pair">
        <div><dt>Fingerprint of the expected software (SHA-256)</dt><dd><code>{facts.before.slice(0, 32)}…</code></dd></div>
        <div><dt>With one character changed</dt><dd><code className="stage-cipher">{facts.after.slice(0, 32)}…</code></dd></div>
      </dl>
      <ul className="strength-facts">
        <li><strong>Any change shows.</strong> One character of the worker&apos;s software changed {facts.bitsChanged} of the 256 bits of its fingerprint, so the report no longer matches what your device expects and the worker is refused.</li>
        <li><strong>The report is signed.</strong> Each worker signs its fingerprint together with a fresh number from your device (ECDSA, P-256), so an old report cannot be replayed.</li>
      </ul>
      <p className="strength-limit"><strong>Where it stops.</strong> In this page the signing key lives in the page. In a real enclave it is held by the processor and vouched for by its maker, so the check rests on that hardware and that maker: an enclave is one more layer, not proof. Hardware attestation of workers has not yet been validated on real hardware.</p>
    </>
  );
}

/** The panel under a stage: what it would take to undo what the stage hides, in facts the page computes. */
export function StrengthPanel({ spec }: { spec: StrengthSpec }) {
  return (
    <div className={`strength strength-${spec.kind}`} role="group" aria-label="How hard is this to undo?">
      <p className="strength-title">How hard is this to undo?</p>
      {spec.kind === "pieces" ? <Pieces spec={spec} /> : null}
      {spec.kind === "seal" ? <Seal spec={spec} /> : null}
      {spec.kind === "report" ? <Report spec={spec} /> : null}
      {spec.kind === "disguise" ? (
        <>
          <p>{spec.what}{spec.ways ? ` There are ${spec.ways} ways to choose it.` : ""}</p>
          <p className="strength-limit"><strong>Where it stops.</strong> This is a disguise, not encryption. The numbers keep their overall size and their relations to one another, and enough observed examples can give the disguise away. Nothing in the design relies on it: it sits underneath the random pieces as an inner layer.</p>
        </>
      ) : null}
      {spec.kind === "wipe" ? (
        <>
          <dl className="strength-pair">
            <div><dt>In memory while your request runs</dt><dd><code>{hex(spec.before)}</code></dd></div>
            <div><dt>Read back by the next user</dt><dd><code className={spec.after.some((value) => value !== 0) ? "strength-leak" : "stage-cipher"}>{hex(spec.after)}</code></dd></div>
          </dl>
          <p>{spec.after.some((value) => value !== 0) ? "The wipe was skipped, so the next user reads your bytes as they were. Nothing has to be broken." : "After a completed wipe there is nothing left to recover: the bytes are zeros, and zeros are the same whatever was there before."}</p>
          <p className="strength-limit"><strong>Where it stops.</strong> This shows the memory a program can read back. A software wipe does not prove that every register and cache inside the chip was cleared.</p>
        </>
      ) : null}
    </div>
  );
}
