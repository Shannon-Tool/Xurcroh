"use client";

import { useMemo, useState } from "react";
import { LabShell } from "@/components/LabShell";
import { freivaldsCheck } from "@/lib/simulation/fortress";
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
  return (
    <LabShell id="freivalds" scene={scene} title="Exact Freivalds verification" caption="The client checks x(Wᵀr) = (xWᵀ)r over the prime field. Exact products avoid the ring-to-prime reduction error demonstrated by this lab.">
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
