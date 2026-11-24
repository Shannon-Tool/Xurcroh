"use client";

import { useState } from "react";
import { Scene3DView } from "@/components/Scene3DView";
import type { Scene3D } from "@/lib/scene/types";

export function LabShell({ id, title, caption, onExport, scene, children }: {
  id: string;
  title: string;
  caption: string;
  onExport?: () => void;
  /** 3D drawing of the current state; the children stay as the exact, readable view. */
  scene?: Scene3D;
  children: React.ReactNode;
}) {
  const [fullscreen, setFullscreen] = useState(false);
  const [show3d, setShow3d] = useState(true);
  return (
    <section id={id} className={`lab ${fullscreen ? "lab-fullscreen" : ""}`} aria-labelledby={`${id}-title`}>
      <div className="lab-header">
        <div>
          <h2 id={`${id}-title`}>{title}</h2>
          <p className="lab-caption">{caption}</p>
        </div>
        <div className="playback">
          {onExport ? <button onClick={onExport}>Export state</button> : null}
          {scene ? <button onClick={() => setShow3d(!show3d)}>{show3d ? "Hide 3D view" : "Show 3D view"}</button> : null}
          <button onClick={() => setFullscreen(!fullscreen)}>{fullscreen ? "Exit fullscreen" : "Fullscreen"}</button>
        </div>
      </div>
      {scene && show3d ? <Scene3DView scene={scene} /> : null}
      {children}
    </section>
  );
}
