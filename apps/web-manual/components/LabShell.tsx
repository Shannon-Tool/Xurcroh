"use client";

import { useState } from "react";

export function LabShell({ id, title, caption, onExport, children }: {
  id: string;
  title: string;
  caption: string;
  onExport?: () => void;
  children: React.ReactNode;
}) {
  const [fullscreen, setFullscreen] = useState(false);
  return (
    <section id={id} className={`lab ${fullscreen ? "lab-fullscreen" : ""}`} aria-labelledby={`${id}-title`}>
      <div className="lab-header">
        <div>
          <h2 id={`${id}-title`}>{title}</h2>
          <p className="lab-caption">{caption}</p>
        </div>
        <div className="playback">
          {onExport ? <button onClick={onExport}>Export state</button> : null}
          <button onClick={() => setFullscreen(!fullscreen)}>{fullscreen ? "Exit fullscreen" : "Fullscreen"}</button>
        </div>
      </div>
      {children}
    </section>
  );
}
