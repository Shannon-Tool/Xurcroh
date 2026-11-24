"use client";

import { useMemo, useState } from "react";
import { LabShell } from "@/components/LabShell";
import { freivaldsCheck, mod } from "@/lib/simulation/fortress";
import type { LiveStory } from "@/components/LiveIO";
import { freivaldsScene } from "@/lib/scene/builders";

export function FreivaldsLab() {
  const [tamper, setTamper] = useState(false);
  const [challenge, setChallenge] = useState("11,13");
  const state = useMemo(() => {
    const values = challenge.split(",").map((value) => Number(value.trim()));
    if (values.length !== 2 || values.some((value) => !Number.isSafeInteger(value) || value < 0)) {
      return { error: "Enter two nonnegative safe integers", valid: false, product: [19, 43], challenge: [0, 0] };
    }
    const product = tamper ? [20, 43] : [19, 43];
    return { error: null, valid: freivaldsCheck([[1, 2], [3, 4]], [5, 7], product, values), product, challenge: values };
  }, [challenge, tamper]);
  const scene = useMemo(() => freivaldsScene({ x: [5, 7], product: state.product, challenge: state.challenge, expected: [19, 43], valid: state.valid }), [state]);
  // Both sides of the test, as the device computes them: x·(Wᵀr) and (product)·r.
  const [r0, r1] = state.challenge.map(BigInt) as [bigint, bigint];
  const left = mod(5n * (r0 + 3n * r1) + 7n * (2n * r0 + 4n * r1)).toString();
  const right = mod(BigInt(state.product[0]!) * r0 + BigInt(state.product[1]!) * r1).toString();
  const story: LiveStory = {
    plain: "A worker sends back an answer. Instead of redoing the whole multiplication, your device tests the answer against a challenge, which is far cheaper.",
    input: <p>The answer a worker returned: <code>{state.product.join(", ")}</code>{tamper ? " (one number was changed by one)" : ""}</p>,
    seenTitle: "The check your device runs",
    seen: (
      <>
        <p>Challenge: <code>{state.challenge.join(", ")}</code></p>
        <p>Side computed from your own data: <code>{left}</code></p>
        <p>Side computed from the answer: <code>{right}</code></p>
      </>
    ),
    output: state.valid ? <p>Accepted: the two sides agree.</p> : <p>Rejected: the two sides differ, so the answer is thrown away.</p>,
    outputTone: state.valid ? "good" : "warn",
    why: "A wrong answer makes the two sides of the test differ, so it is rejected before anything is built on it.",
    relies: "The worker cannot predict the challenge, and the arithmetic is exact. In this lab you type the challenge yourself.",
    limit: "It shows the answer is correct, not that the worker kept anything private. A wrong answer can pass only with a tiny probability.",
    attack: {
      label: "change one number by one",
      on: tamper,
      onToggle: () => setTamper(!tamper),
      outcome: tamper ? (state.valid ? "Not caught with this challenge. Try another challenge." : "Caught: the sides no longer agree.") : "The answer is untouched and passes.",
    },
    screens: { input: state.product.join(", "), seen: `${left}\nvs ${right}`, output: state.valid ? "accepted" : "rejected" },
  };
  return (
    <LabShell id="freivalds" scene={scene} story={story} title="Exact Freivalds verification" caption="The client checks x(Wᵀr) = (xWᵀ)r over the prime field. Exact products avoid the ring-to-prime reduction error demonstrated by this lab.">
      <div className="lab-controls">
        <label>Challenge vector r
          <input value={challenge} onChange={(event) => setChallenge(event.target.value)} />
        </label>
        <label><input type="checkbox" checked={tamper} onChange={(event) => setTamper(event.target.checked)} /> Add one to the first product</label>
      </div>
      {state.error ? <p role="alert">{state.error}</p> : null}
      <p>Product: <strong>{state.product.join(", ")}</strong></p>
      <p>Verification: <strong>{state.valid ? "accepted" : "rejected"}</strong></p>
      <p className="small">This is a deterministic teaching check, not a probability bound for BF16 arithmetic or a challenge hidden from a malicious worker.</p>
    </LabShell>
  );
}
