"use client";

import { useMemo, useState } from "react";
import { LabShell } from "@/components/LabShell";
import type { LiveStory } from "@/components/LiveIO";
import { PlaybackControls } from "@/components/PlaybackControls";
import { usePlayback } from "@/components/usePlayback";
import { TOKEN_MS } from "@/lib/show/generation";
import { grouped, type ProcessStep } from "@/lib/show/process";
import { freivaldsCheck, mod } from "@/lib/simulation/fortress";
import { freivaldsScene } from "@/lib/scene/builders";
import { useJourney } from "@/lib/show/question";

export function FreivaldsLab() {
  // The workers' result for the chosen question: one score per prepared answer.
  const journey = useJourney();
  const [tamper, setTamper] = useState(false);
  const [challenge, setChallenge] = useState("11,13,17,19,23,29");
  // The result, the challenge, one side, the other side, the comparison.
  const playback = usePlayback(4, journey.tokens.length * TOKEN_MS + 2600);
  const index = playback.index;
  const state = useMemo(() => {
    const values = challenge.split(",").map((value) => Number(value.trim()));
    const product = journey.scores.map((score, index) => (tamper && index === 0 ? score + 1 : score));
    if (values.length !== journey.scores.length || values.some((value) => !Number.isSafeInteger(value) || value < 0)) {
      return { error: `Enter ${journey.scores.length} nonnegative safe integers`, valid: false, product, challenge: journey.scores.map(() => 0) };
    }
    return { error: null, valid: freivaldsCheck(journey.weightRows, journey.counts, product, values), product, challenge: values };
  }, [journey, challenge, tamper]);
  // Both sides of the test, as the device computes them: x·(Wᵀr) and (product)·r.
  const r = state.challenge.map(BigInt);
  const left = mod(journey.counts.reduce((sum, count, column) => sum + BigInt(count) * journey.weightRows.reduce((inner, row, index) => inner + BigInt(row[column]!) * r[index]!, 0n), 0n)).toString();
  const right = mod(state.product.reduce((sum, value, index) => sum + BigInt(value) * r[index]!, 0n)).toString();
  const scene = useMemo(
    () => freivaldsScene({ x: journey.counts.filter((count) => count > 0).slice(0, 6), product: state.product, challenge: state.challenge, expected: journey.scores, valid: state.valid, step: index, sides: { left: grouped(left), right: grouped(right) } }),
    [journey, state, index, left, right],
  );
  const worked = state.product.map((value, at) => `${value}×${state.challenge[at]}`).join(" + ");
  const steps: ProcessStep[] = [
    {
      id: "result", title: "The workers' result", tone: tamper ? "warn" : "accent",
      does: `The workers send back one score for each prepared answer${tamper ? "; on the way the first was changed by one" : ""}. Your device has not checked them yet.`,
      data: [{ kind: "numbers", label: "scores", values: state.product, show: 6 }],
    },
    {
      id: "challenge", title: "A challenge", tone: "accent",
      does: "Your device picks one number for every score. The workers did not know these numbers when they answered. (Fixed in this lab; you can change them below.)",
      data: [{ kind: "numbers", label: "challenge", values: state.challenge, show: 6 }],
    },
    {
      id: "right", title: "Side from the result", tone: "accent",
      does: `Each score is multiplied by its challenge number and all are added: ${worked} = ${grouped(right)}.`,
      data: [{ kind: "sum", text: `${worked.length > 34 ? `${worked.slice(0, 33)}…` : worked}` }, { kind: "numbers", values: [right] }],
    },
    {
      id: "left", title: "Side from your question", tone: "good",
      does: `Your device computes the same quantity its own way, from your question's ${journey.counts.length} numbers and the model's table mixed with the challenge: ${grouped(left)}. This costs one pass over the question, not the workers' whole job.`,
      data: [{ kind: "numbers", values: [left] }],
    },
    {
      id: "compare", title: state.valid ? "They match" : "They differ", tone: state.valid ? "good" : "warn",
      does: state.valid ? `${grouped(left)} equals ${grouped(right)}: the result is accepted and may be used.` : `${grouped(left)} is not ${grouped(right)}: the result is thrown away and no answer is written.`,
      data: [{ kind: "rows", rows: [{ name: "yours", value: grouped(left), state: "done" }, { name: "theirs", value: grouped(right), state: state.valid ? "done" : "bad" }] }],
    },
  ];
  const story: LiveStory = {
    plain: "The workers send back their result for your question. Instead of redoing all their work, your device tests the result against a challenge, which is far cheaper.",
    stage: "check",
    steps,
    step: index,
    done: index >= 4,
    speed: playback.speed,
    onStep: playback.goTo,
    result: [{ kind: "bars", label: "Checked score of each prepared answer", values: state.product, names: state.product.map((_, at) => `${at + 1}`) }],
    input: <p>The result the workers returned for it, one score per prepared answer: <code>{state.product.join(", ")}</code>{tamper ? " (one number was changed by one)" : ""}</p>,
    seenTitle: "The check your device runs",
    seen: (
      <>
        <p>Challenge: <code>{index >= 1 ? state.challenge.join(", ") : "not chosen yet"}</code></p>
        <p>Side computed from your own question: <code>{index >= 3 ? left : "…"}</code></p>
        <p>Side computed from the workers&apos; result: <code>{index >= 2 ? right : "…"}</code></p>
      </>
    ),
    output: state.valid ? <p>Accepted: the two sides agree, so the result may be used.</p> : <p>Rejected: the two sides differ, so the result is thrown away.</p>,
    released: state.valid,
    outputTone: state.valid ? "good" : "warn",
    why: "A wrong result makes the two sides of the test differ, so it is rejected before an answer is built on it.",
    relies: "The workers cannot predict the challenge, and the arithmetic is exact. In this lab the challenge is fixed, and you can change it below.",
    limit: "It shows the result is correct, not that the workers kept anything private. A wrong result can pass only with a tiny probability.",
    attack: {
      label: "change one number by one",
      on: tamper,
      onToggle: () => setTamper(!tamper),
      outcome: tamper ? (state.valid ? "Not caught with this challenge. Try another challenge." : "Caught: the sides no longer agree.") : "The result is untouched and passes.",
    },
    screenSeen: `${left}\nvs ${right}`,
  };
  return (
    <LabShell id="freivalds" plainTitle="Catching a wrong answer" scene={scene} story={story} watch={playback.watch} title="Exact Freivalds verification" caption="The client checks x(Wᵀr) = (xWᵀ)r over the prime field. Exact products avoid the ring-to-prime reduction error demonstrated by this lab.">
      <div className="lab-controls">
        <label>Challenge vector r
          <input value={challenge} onChange={(event) => setChallenge(event.target.value)} />
        </label>
        <label><input type="checkbox" checked={tamper} onChange={(event) => setTamper(event.target.checked)} /> Add one to the first product</label>
        <PlaybackControls {...playback} canStepBackward={index > 0} onToggle={() => playback.setPlaying(!playback.playing)} />
      </div>
      {state.error ? <p role="alert">{state.error}</p> : null}
      <p>Product: <strong>{state.product.join(", ")}</strong></p>
      <p>Verification: <strong>{state.valid ? "accepted" : "rejected"}</strong></p>
      <p className="small">This is a deterministic teaching check, not a probability bound for BF16 arithmetic or a challenge hidden from a malicious worker.</p>
    </LabShell>
  );
}
