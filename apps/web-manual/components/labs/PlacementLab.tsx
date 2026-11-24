"use client";

import { useMemo, useState } from "react";
import { LabShell } from "@/components/LabShell";
import { placeOperations } from "@/lib/simulation/fortress";
import { placementScene } from "@/lib/scene/builders";

export function PlacementLab() {
  const [epoch, setEpoch] = useState(0);
  const roles = useMemo(() => placeOperations(epoch), [epoch]);
  const scene = useMemo(() => placementScene({ roles }), [roles]);
  return (
    <LabShell id="placement" scene={scene} title="Operation placement and epoch reshuffle" caption="Adjacent and paired operations occupy disjoint providers. Country and CPU-vendor labels make diversity inspectable; production additionally enforces secret placement policy.">
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
