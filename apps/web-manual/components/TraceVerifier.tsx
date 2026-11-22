"use client";

import { useState, type ChangeEvent } from "react";
import { TraceError, verifyTrace, type TraceReport } from "@/lib/verify/trace";

type Outcome = { report: TraceReport } | { error: string } | null;

const FIELDS = [
  ["trace", "Trace (JSON)", "The file from client.trace(task_id) or GET /v1/tasks/{id}/trace."],
  ["publicKey", "Expected signer key (hex, optional)", "Without it the trace only vouches for itself: damage is caught, a re-signed forgery is not."],
  ["messages", "Chat messages (JSON, optional)", "The exact messages list you sent. Checks the trace is about this prompt."],
  ["media", "Image or video request (JSON, optional)", "An object with prompt, image and mask, instead of chat messages."],
  ["output", "Output text (optional)", "The exact text returned. Checks the trace is about this output."],
  ["proof", "Log proof (JSON, optional)", "From GET /v1/tasks/{id}/proof. Shows the trace hash is in the public log."],
  ["logKey", "Log public key (hex, needed with a proof)", ""],
  ["witnessKey", "Witness public key (hex, optional)", "Requires this independent witness to have countersigned the log head."],
] as const;

type FieldName = (typeof FIELDS)[number][0];
const SINGLE_LINE: FieldName[] = ["publicKey", "logKey", "witnessKey"];
const EMPTY = Object.fromEntries(FIELDS.map(([name]) => [name, ""])) as Record<FieldName, string>;

function meaning(report: TraceReport): string[] {
  const lines = [
    report.signerPinned
      ? "The trace was signed by the key you expected and has not been changed since."
      : "The trace matches the key stored inside it. Give the expected signer key to rule out a trace re-signed by someone else.",
  ];
  if (report.prompt) lines.push("It refers to the prompt you gave.");
  if (report.output) lines.push("It refers to the output you gave.");
  lines.push(
    report.workerCheckpoints === null
      ? "No worker signed a checkpoint in this trace."
      : `${report.workerCheckpoints} worker checkpoint signatures verify, each against the artifacts named in the model manifest.`,
  );
  if (report.log) {
    lines.push(
      `Its hash is in the log at tree size ${report.log.treeSize}` +
        (report.log.witnesses.length
          ? `, and that log head is countersigned by the witness you named.`
          : `. No witness was checked, so this is only as trustworthy as the log operator.`),
    );
  }
  if (report.mode === "hosted") {
    lines.push(
      `This is a hosted trace (${report.assurance ?? "no assurance stated"}): the model, settings and routing are the gateway operator's signed statement, not something this check can prove.`,
    );
  }
  return lines;
}

export function TraceVerifier() {
  const [values, setValues] = useState(EMPTY);
  const [outcome, setOutcome] = useState<Outcome>(null);

  const update = (name: FieldName) => (event: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setValues((current) => ({ ...current, [name]: event.target.value }));
    setOutcome(null);
  };

  const check = () => {
    try {
      const report = verifyTrace(values.trace, {
        publicKey: values.publicKey,
        messages: values.messages,
        media: values.media,
        output: values.output,
        proof: values.proof,
        logKey: values.logKey,
        witnessKeys: values.witnessKey.trim() ? [values.witnessKey] : [],
      });
      setOutcome({ report });
    } catch (error) {
      setOutcome({
        error: error instanceof TraceError ? error.message : "The trace could not be checked.",
      });
    }
  };

  return (
    <section className="lab" aria-labelledby="trace-verifier-title">
      <h2 id="trace-verifier-title">Check a trace</h2>
      <p className="lab-caption">
        Everything below runs in this page on your device. Nothing you paste is sent anywhere; you
        can disconnect from the network before pasting.
      </p>
      <div className="verify-form">
        {FIELDS.map(([name, label, hint]) => (
          <label key={name}>
            {label}
            {SINGLE_LINE.includes(name) ? (
              <input value={values[name]} onChange={update(name)} spellCheck={false} autoComplete="off" />
            ) : (
              <textarea value={values[name]} onChange={update(name)} rows={name === "trace" ? 8 : 3} spellCheck={false} />
            )}
            {hint ? <span className="small">{hint}</span> : null}
          </label>
        ))}
      </div>
      <button type="button" onClick={check} disabled={!values.trace.trim()}>Verify</button>
      {outcome && "error" in outcome ? (
        <p role="alert" className="error"><strong>Not verified.</strong> {outcome.error}</p>
      ) : null}
      {outcome && "report" in outcome ? (
        <div role="status">
          <p className="success"><strong>Verified.</strong></p>
          <ul>
            {meaning(outcome.report).map((line) => <li key={line}>{line}</li>)}
          </ul>
          <p className="small">Signer key fingerprint: <code>{outcome.report.signer}</code></p>
          <details>
            <summary>Facts recorded in the trace</summary>
            <pre className="state-inspector">{JSON.stringify(outcome.report.facts, null, 2)}</pre>
          </details>
        </div>
      ) : null}
    </section>
  );
}
