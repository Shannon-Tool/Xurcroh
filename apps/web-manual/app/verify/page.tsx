import type { Metadata } from "next";
import { TraceVerifier } from "@/components/TraceVerifier";

export const metadata: Metadata = {
  title: "Verify a task trace",
  description: "Check a Xurcroh task trace in your browser, without installing or sending anything.",
};

export default function VerifyPage() {
  return (
    <article className="article">
      <header className="hero">
        <span className="evidence">Runs on your device · independent of the Python client</span>
        <h1>Verify a task trace</h1>
        <p>
          Every completed request produces a signed trace: what was asked for, which model build ran, how
          the links were protected and what each worker signed for. This page checks one without any
          software from us running on your machine beyond the page itself.
        </p>
      </header>
      <TraceVerifier />
      <h2>What is checked</h2>
      <ul>
        <li>Both halves of the signature (Ed25519 and ML-DSA-65) over the whole trace.</li>
        <li>That the model facts hash to the manifest digest in the trace.</li>
        <li>Every worker checkpoint signature, and that each worker signed for the artifacts the manifest names.</li>
        <li>If you give them: that the trace is about your prompt and your output.</li>
        <li>If you give a proof: that the trace hash is in the log, under a log head signed by the log key and, if you name one, a witness.</li>
      </ul>
      <h2>What is not</h2>
      <ul>
        <li>Whether the log still holds its earlier history. That needs a live query: <code>xurcroh verify --log-url</code>.</li>
        <li>Whether a signer key or worker key belongs to who you think. Obtain keys from a source you trust.</li>
        <li>For a hosted trace, whether the gateway told the truth about the model and settings. Only running the client yourself gives that.</li>
      </ul>
      <p className="small">
        This verifier is a separate implementation from the Python one and is tested against traces the
        Python client signs. Neither has been externally audited.
      </p>
    </article>
  );
}
