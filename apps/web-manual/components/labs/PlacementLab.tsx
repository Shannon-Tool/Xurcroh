"use client";

import { useEffect, useMemo, useState } from "react";
import { LabShell } from "@/components/LabShell";
import { PlaybackControls } from "@/components/PlaybackControls";
import { usePlayback } from "@/components/usePlayback";
import { writingTime } from "@/lib/show/generation";
import { stepped, type ProcessStep } from "@/lib/show/process";
import { placeOperations } from "@/lib/simulation/fortress";
import { placementScene } from "@/lib/scene/builders";
import type { LiveStory } from "@/components/LiveIO";
import { useJourney } from "@/lib/show/question";

export function PlacementLab() {
  // Each question is dealt out to the providers differently; the shuffle number comes from the question.
  const journey = useJourney();
  const [epoch, setEpoch] = useState(journey.hash % 21);
  useEffect(() => setEpoch(journey.hash % 21), [journey]);
  const roles = useMemo(() => placeOperations(epoch), [epoch]);
  // The work listed, then one operation handed out at a time, then the check.
  const operations = useMemo(() => [...new Set(roles.map((role) => role.operation))], [roles]);
  const LAST = operations.length + 1;
  const playback = usePlayback(LAST, writingTime(journey.tokens.length));
  const index = playback.index;
  const placedCount = Math.max(0, Math.min(operations.length, index));
  const scene = useMemo(() => placementScene({ roles, placed: index > operations.length ? undefined : placedCount }), [roles, index, placedCount, operations]);
  // What each provider ends up holding, and whether any provider holds two pieces of one operation.
  const holdings = new Map<string, string[]>();
  let pairs = 0;
  for (const role of roles) {
    const held = holdings.get(role.provider) ?? [];
    if (held.some((entry) => entry.startsWith(`${role.operation} `))) pairs += 1;
    held.push(`${role.operation} (${role.activationShare},${role.weightShare})`);
    holdings.set(role.provider, held);
  }
  const [firstProvider, firstHeld] = [...holdings.entries()][0]!;
  const shownHeld = firstHeld.filter((entry) => operations.indexOf(entry.split(" ")[0]!) < placedCount);
  const steps: ProcessStep[] = [
    {
      id: "work", title: "The work", tone: "plain",
      does: `Answering your question takes ${operations.length} operations. Each is done on two pieces of your data and two pieces of the model, so by 4 workers: ${roles.length} pieces of work to hand out.`,
      data: [{ kind: "rows", rows: operations.map((operation) => ({ name: operation, value: "4 workers", state: "todo" as const })) }],
    },
    ...operations.map((operation, at): ProcessStep => {
      const own = roles.filter((role) => role.operation === operation);
      return {
        id: `op-${at}`, title: operation, tone: "accent",
        does: `The four pieces of “${operation}” go to four different providers: ${own.map((role) => `(${role.activationShare},${role.weightShare}) to ${role.provider} in ${role.country.toUpperCase()}`).join(", ")}.`,
        data: [{ kind: "rows", rows: own.map((role) => ({ name: `(${role.activationShare},${role.weightShare})`, value: `${role.provider} ${role.country.toUpperCase()}`, state: "done" as const })) }],
      };
    }),
    {
      id: "check", title: pairs === 0 ? "Nobody holds a pair" : "A pair is held", tone: pairs === 0 ? "good" : "warn",
      does: pairs === 0 ? `Checked: ${pairs} of ${holdings.size} providers hold two pieces of the same operation, so no single provider can put your data back together.` : `${pairs} of ${holdings.size} providers hold two pieces of the same operation: this placement must not be used.`,
      data: [{ kind: "rows", rows: [{ name: "providers", value: String(holdings.size), state: "done" }, { name: "holding a pair", value: String(pairs), state: pairs === 0 ? "done" : "bad" }] }],
    },
  ];
  const story: LiveStory = {
    plain: "The work on your question is cut into pieces, and every piece goes to a different provider. The planner makes sure no provider ends up with two pieces that fit together.",
    stage: "place",
    steps,
    ...stepped(playback, LAST),
    released: pairs === 0,
    input: <p>Answering it takes 4 operations, each done by 4 workers: {roles.length} pieces of work. For this question they were dealt out with shuffle number {epoch}.</p>,
    seenTitle: `What ${firstProvider} holds`,
    seen: shownHeld.length === 0 ? <p>Nothing yet. It is one of {holdings.size} providers.</p> : <p><code>{shownHeld.join("; ")}</code> and nothing else. It is one of {holdings.size} providers.</p>,
    output: <p>{pairs} of {holdings.size} providers hold two pieces of the same operation.</p>,
    outputTone: pairs === 0 ? "good" : "warn",
    why: "Rebuilding your question takes two matching pieces. A provider that only ever holds one piece of an operation cannot do it alone.",
    relies: "Providers are run by independent operators who do not cooperate. The provider names here are stand-ins.",
    limit: "Providers that work together defeat this. Placement and regular reshuffling make that harder; they cannot rule it out.",
    screenSeen: `${firstProvider}\n${firstHeld.join(", ")}`,
  };
  return (
    <LabShell id="placement" plainTitle="Deciding who gets which piece" scene={scene} story={story} watch={playback.watch} title="Operation placement and epoch reshuffle" caption="Adjacent and paired operations occupy disjoint providers. Country and CPU-vendor labels make diversity inspectable; production additionally enforces secret placement policy.">
      <div className="lab-controls">
        <label>Placement epoch
          <input type="number" min="0" max="20" value={epoch} onChange={(event) => setEpoch(Number(event.target.value))} />
        </label>
        <button onClick={() => setEpoch((value) => value + 1)}>Reshuffle</button>
        <PlaybackControls {...playback} canStepBackward={index > 0} onToggle={() => playback.setPlaying(!playback.playing)} />
      </div>
      <p>Current epoch: <strong>{epoch}</strong>; assignments: <strong>{roles.length}</strong></p>
      <div style={{ overflowX: "auto" }}>
        <table>
          <thead><tr><th>Operation</th><th>Role</th><th>Provider</th><th>Country</th><th>CPU</th></tr></thead>
          <tbody>
            {roles.map((role) => (
              <tr key={`${role.operation}-${role.activationShare}-${role.weightShare}`}>
                <td>{role.operation}</td><td>({role.activationShare},{role.weightShare})</td><td>{role.provider}</td><td>{role.country}</td><td>{role.cpuVendor}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </LabShell>
  );
}
