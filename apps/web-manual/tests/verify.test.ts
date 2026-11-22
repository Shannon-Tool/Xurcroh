// @vitest-environment node
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { sha256Hex, utf8 } from "@/lib/verify/crypto";
import { canonicalText, parseJson, without, type JsonObject } from "@/lib/verify/json";
import { TraceError, verifyTrace } from "@/lib/verify/trace";

// Produced by scripts/generate_verify_fixtures.py with the production Python code.
const fixtureText = readFileSync(resolve("tests/fixtures/verify.json"), "utf8");
const fixture = parseJson(fixtureText) as JsonObject;
const text = (name: string) => fixture.get(name) as string;
const json = (name: string) => canonicalText(fixture.get(name)!, false);
const trace = text("trace_text");

function edited(source: string, change: (document: JsonObject) => void): string {
  const document = parseJson(source) as JsonObject;
  change(document);
  return canonicalText(document, false);
}

describe("Python-compatible canonical JSON", () => {
  it("reproduces the bytes Python signed, including float spelling and Unicode", () => {
    const body = without(parseJson(trace) as JsonObject, "signature");
    expect(canonicalText(body, false)).toBe(text("body_canonical"));
    expect(sha256Hex(utf8(canonicalText(parseJson(trace), false)))).toBe(
      text("trace_canonical_sha256"),
    );
  });

  it("spells numbers as Python does whatever spelling the file used", () => {
    const cases: [string, string][] = [
      ["1E22", "1e+22"],
      ["1e-7", "1e-07"],
      ["0.0001", "0.0001"],
      ["0.00001", "1e-05"],
      ["1.0", "1.0"],
      ["100.0", "100.0"],
      ["1e15", "1000000000000000.0"],
      ["1e16", "1e+16"],
      ["123456789.125", "123456789.125"],
      ["-0.0", "-0.0"],
      ["5e-324", "5e-324"],
      ["1.7976931348623157e308", "1.7976931348623157e+308"],
      ["12345678901234567890123", "12345678901234567890123"],
      ["-0", "0"],
    ];
    for (const [written, expected] of cases) {
      expect(canonicalText(parseJson(written), true)).toBe(expected);
    }
  });

  it("escapes and orders strings as Python does", () => {
    const value = parseJson('{"\\ud83d\\ude00":1,"\\uffff":2,"a\\u007f":"é\\u2028\\u0001"}');
    expect(canonicalText(value, true)).toBe(
      '{"a\\u007f":"\\u00e9\\u2028\\u0001","\\uffff":2,"\\ud83d\\ude00":1}',
    );
    expect(canonicalText(value, false)).toBe('{"a\u007f":"é \\u0001","￿":2,"😀":1}');
  });

  it("refuses duplicate property names and trailing text", () => {
    expect(() => parseJson('{"a":1,"a":2}')).toThrow(/duplicate/);
    expect(() => parseJson("{} x")).toThrow();
  });
});

describe("verifyTrace", () => {
  it("accepts a trace signed by the Python client and opens its commitments", () => {
    const report = verifyTrace(trace, {
      publicKey: text("signer_public"),
      messages: json("messages"),
      output: text("output"),
    });
    expect(report).toMatchObject({
      signature: true,
      signerPinned: true,
      prompt: true,
      output: true,
      workerCheckpoints: 2,
      mode: "local",
      log: null,
    });
  });

  it("accepts a media trace and reports that workers signed nothing", () => {
    const report = verifyTrace(json("media_trace"), { media: json("media") });
    expect(report).toMatchObject({
      prompt: true,
      workerCheckpoints: null,
      signerPinned: false,
      assurance: "operator-asserted",
    });
  });

  it("checks the log proof and the witness countersignature", () => {
    const report = verifyTrace(trace, {
      proof: json("proof"),
      logKey: text("log_key"),
      witnessKeys: [text("witness_key")],
    });
    expect(report.log).toEqual({
      treeSize: "7",
      witnesses: [expect.stringMatching(/^[0-9a-f]{64}$/)],
    });
  });

  const failures: [string, () => unknown, RegExp][] = [
    [
      "a changed fact",
      () => verifyTrace(edited(trace, (d) => d.set("mode", "hosted"))),
      /signature/,
    ],
    [
      "a changed number",
      () => verifyTrace(trace.replace("1795305601.5", "1795305601.6")),
      /signature/,
    ],
    [
      "another signer than the expected one",
      () => verifyTrace(trace, { publicKey: text("other_public") }),
      /signature/,
    ],
    [
      "a corrupted post-quantum half of the signature",
      () =>
        verifyTrace(
          edited(trace, (d) => {
            const signature = d.get("signature") as string;
            const flipped = signature.at(-1) === "0" ? "1" : "0";
            d.set("signature", signature.slice(0, -1) + flipped);
          }),
        ),
      /signature/,
    ],
    [
      "a corrupted classical half of the signature",
      () =>
        verifyTrace(
          edited(trace, (d) => {
            const signature = d.get("signature") as string;
            d.set("signature", (signature[0] === "0" ? "1" : "0") + signature.slice(1));
          }),
        ),
      /signature/,
    ],
    ["a different prompt", () => verifyTrace(trace, { messages: '[{"role":"user","content":"x"}]' }), /prompt/],
    ["a different output", () => verifyTrace(trace, { output: "Five" }), /output/],
    [
      "a proof for another log key",
      () => verifyTrace(trace, { proof: json("proof"), logKey: text("signer_public") }),
      /not in the log/,
    ],
    [
      "a proof for another trace",
      () => verifyTrace(json("media_trace"), { proof: json("proof"), logKey: text("log_key") }),
      /not in the log/,
    ],
    [
      "a proof with a changed path",
      () => {
        const proof = edited(json("proof"), (d) => {
          const path = d.get("path") as string[];
          path[0] = "00".repeat(32);
        });
        return verifyTrace(trace, { proof, logKey: text("log_key") });
      },
      /not in the log/,
    ],
    [
      "a witness that did not sign",
      () =>
        verifyTrace(trace, {
          proof: json("proof"),
          logKey: text("log_key"),
          witnessKeys: [text("other_public")],
        }),
      /witness/,
    ],
    [
      "a witness key without a proof",
      () => verifyTrace(trace, { witnessKeys: [text("witness_key")] }),
      /witness/,
    ],
  ];
  for (const [name, attempt, reason] of failures) {
    it(`rejects ${name}`, () => {
      expect(attempt).toThrow(TraceError);
      expect(attempt).toThrow(reason);
    });
  }

  // In these the outer signature is valid (Python re-signed after the change); only
  // re-checking the inner part catches them.
  const inner: [string, string, RegExp][] = [
    ["a checkpoint body changed after the worker signed it", "forged_checkpoint_trace", /checkpoint signature does not verify \(role-0\)/],
    ["a worker that signed for other artifacts", "wrong_artifacts_trace", /role-1 signed for artifacts other/],
    ["model facts that no longer match the manifest digest", "wrong_manifest_trace", /manifest digest/],
  ];
  for (const [name, key, reason] of inner) {
    it(`rejects ${name}`, () => {
      expect(() => verifyTrace(json(key))).toThrow(reason);
    });
  }
});
