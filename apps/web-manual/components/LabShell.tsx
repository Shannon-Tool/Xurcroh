"use client";

import { useEffect, useMemo, useState } from "react";
import { LiveIO, type LiveStory } from "@/components/LiveIO";
import { Scene3DView } from "@/components/Scene3DView";
import { withIOScreens } from "@/lib/scene/io-screens";
import type { Scene3D } from "@/lib/scene/types";
import { paintCard } from "@/lib/show/paint";

export function LabShell({ id, title, caption, onExport, scene, story, watch, children }: {
  id: string;
  title: string;
  caption: string;
  onExport?: () => void;
  /** 3D drawing of the current state; the children stay as the exact, readable view. */
  scene?: Scene3D;
  /** The lab as a live story: in, what others see, out. Shown above the exact numbers. */
  story?: LiveStory;
  /** Lets the header's Watch button play the lab from the start. */
  watch?: { playing: boolean; onToggle: () => void };
  children: React.ReactNode;
}) {
  const [fullscreen, setFullscreen] = useState(false);
  const [show3d, setShow3d] = useState(true);
  const [screens, setScreens] = useState<Record<string, HTMLCanvasElement> | null>(null);
  const [painted, setPainted] = useState(0);

  // The same three values the strip shows are drawn on screens inside the 3D view.
  const hasScreens = story?.screens !== undefined;
  const inText = story?.screens?.input ?? "";
  const seenText = story?.screens?.seen ?? "";
  const outText = story?.screens?.output ?? "";
  const warn = story?.outputTone === "warn";
  useEffect(() => {
    if (!hasScreens) return;
    const canvases = screens ?? { "io-in": document.createElement("canvas"), "io-seen": document.createElement("canvas"), "io-out": document.createElement("canvas") };
    paintCard(canvases["io-in"]!, "Goes in", inText, "good");
    paintCard(canvases["io-seen"]!, "Others see", seenText, "accent");
    paintCard(canvases["io-out"]!, "Comes out", outText, warn ? "warn" : "good");
    if (!screens) setScreens(canvases);
    setPainted((current) => current + 1);
  }, [hasScreens, inText, seenText, outText, warn, screens]);

  const staged = useMemo(() => (scene && hasScreens ? withIOScreens(scene, warn ? "warn" : "good") : scene), [scene, hasScreens, warn]);

  return (
    <section id={id} className={`lab ${fullscreen ? "lab-fullscreen" : ""}`} aria-labelledby={`${id}-title`}>
      <div className="lab-header">
        <div>
          <h2 id={`${id}-title`}>{title}</h2>
          {story ? <p className="lab-plain">{story.plain}</p> : null}
          <p className="lab-caption">{caption}</p>
        </div>
        <div className="playback">
          {watch ? <button className="lab-watch" onClick={watch.onToggle}>{watch.playing ? "Stop" : "Watch"}</button> : null}
          {onExport ? <button onClick={onExport}>Export state</button> : null}
          {scene ? <button onClick={() => setShow3d(!show3d)}>{show3d ? "Hide 3D view" : "Show 3D view"}</button> : null}
          <button onClick={() => setFullscreen(!fullscreen)}>{fullscreen ? "Exit fullscreen" : "Fullscreen"}</button>
        </div>
      </div>
      {story ? <LiveIO story={story} /> : null}
      {staged && show3d ? <Scene3DView scene={staged} screens={screens ?? undefined} screensVersion={painted} /> : null}
      {story ? (
        <details className="lab-exact" open>
          <summary>Exact numbers and controls</summary>
          {children}
        </details>
      ) : children}
    </section>
  );
}
