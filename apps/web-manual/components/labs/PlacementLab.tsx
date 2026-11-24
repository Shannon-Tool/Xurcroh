"use client";

import { useMemo, useState } from "react";
import { LabShell } from "@/components/LabShell";
import { placeOperations } from "@/lib/simulation/fortress";
import { placementScene } from "@/lib/scene/builders";
import type { LiveStory } from "@/components/LiveIO";

export function PlacementLab() {
  const [epoch, setEpoch] = useState(0);
  const roles = useMemo(() => placeOperations(epoch), [epoch]);
  const scene = useMemo(() => placementScene({ roles }), [roles]);
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
  const story: LiveStory = {
    plain: "Every operation is done by four workers, each holding one piece. The planner spreads the pieces over different providers so that no provider holds two pieces that fit together.",
    input: (
      <>
        <p>4 operations, 4 pieces each.</p>
        <label>Shuffle number <input type="number" min="0" max="20" value={epoch} onChange={(event) => setEpoch(Number(event.target.value))} /></label>
      </>
    ),
    seenTitle: `What ${firstProvider} holds`,
    seen: <p><code>{firstHeld.join("; ")}</code> and nothing else. It is one of {holdings.size} providers.</p>,
    output: <p>{pairs} of {holdings.size} providers hold two pieces of the same operation.</p>,
    outputTone: pairs === 0 ? "good" : "warn",
    why: "Rebuilding your data takes two matching pieces. A provider that only ever holds one piece of an operation cannot do it alone.",
    relies: "Providers are run by independent operators who do not cooperate. The provider names here are stand-ins.",
    limit: "Providers that work together defeat this. Placement and regular reshuffling make that harder; they cannot rule it out.",
    screens: { input: `4 operations\n${roles.length} pieces`, seen: `${firstProvider}\n${firstHeld.join(", ")}`, output: `${pairs} providers\nhold a pair` },
  };
  return (
    <LabShell id="placement" scene={scene} story={story} title="Operation placement and epoch reshuffle" caption="Adjacent and paired operations occupy disjoint providers. Country and CPU-vendor labels make diversity inspectable; production additionally enforces secret placement policy.">
      <div className="lab-controls">
        <label>Placement epoch
          <input type="number" min="0" max="20" value={epoch} onChange={(event) => setEpoch(Number(event.target.value))} />
        </label>
        <button onClick={() => setEpoch((value) => value + 1)}>Reshuffle</button>
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
