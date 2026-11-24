"use client";

import { createContext, useContext, useState } from "react";
import { LiveIO, LiveNotes, useLabPanels, type LiveStory } from "@/components/LiveIO";
import { Scene3DView } from "@/components/Scene3DView";
import { StageBox } from "@/components/StageBox";
import type { Scene3D } from "@/lib/scene/types";

/** Inside this context a lab shows only its three panels: used where a lab is quoted on another page. */
export const LabEmbedContext = createContext(false);

const EMPTY_SCENE: Scene3D = { caption: "", nodes: [], links: [] };

/** The lab's panels laid over its own 3D scene on the shared 16:9 stage. */
function LabStage({ title, scene, story }: { title: string; scene: Scene3D; story: LiveStory }) {
  const panels = useLabPanels(story);
  return (
    <StageBox
      label={`${title}: live stage`}
      scene={scene}
      left={<>{panels.input}{panels.seen}</>}
      right={panels.output}
      caption={scene.caption || story.plain}
      why={story.why}
      forms={panels.forms}
      current={panels.current}
      ribbonLabel="The trip from your question to the answer"
    />
  );
}

export function LabShell({ id, title, plainTitle, caption, onExport, scene, story, watch, children }: {
  id: string;
  /** The technical name of the lab. */
  title: string;
  /** The everyday heading; when given, the technical name moves to the caption line. */
  plainTitle?: string;
  caption: string;
  onExport?: () => void;
  /** 3D drawing of the current state; the children stay as the exact, readable view. */
  scene?: Scene3D;
  /** The lab as a live story: in, what others see, out. Shown on the stage above the exact numbers. */
  story?: LiveStory;
  /** Lets the header's Watch button play the lab from the start. */
  watch?: { playing: boolean; onToggle: () => void };
  children: React.ReactNode;
}) {
  const [fullscreen, setFullscreen] = useState(false);
  const [show3d, setShow3d] = useState(true);
  const embedded = useContext(LabEmbedContext);

  if (embedded) return story ? <div className="lab-embed" role="group" aria-label={plainTitle ?? title}><LiveIO story={story} /></div> : null;

  return (
    <section id={id} className={`lab ${fullscreen ? "lab-fullscreen" : ""}`} aria-labelledby={`${id}-title`}>
      <div className="lab-header">
        <div>
          <h2 id={`${id}-title`}>{plainTitle ?? title}</h2>
          {story ? <p className="lab-plain">{story.plain}</p> : null}
          <p className="lab-caption">{plainTitle ? <><strong>{title}.</strong> </> : null}{caption}</p>
        </div>
        <div className="playback">
          {watch ? <button className="lab-watch" onClick={watch.onToggle}>{watch.playing ? "Stop" : "Watch"}</button> : null}
          {onExport ? <button onClick={onExport}>Export state</button> : null}
          {scene || story ? <button onClick={() => setShow3d(!show3d)}>{show3d ? "Hide 3D view" : "Show 3D view"}</button> : null}
          <button onClick={() => setFullscreen(!fullscreen)}>{fullscreen ? "Exit fullscreen" : "Fullscreen"}</button>
        </div>
      </div>
      {story && show3d ? <LabStage title={plainTitle ?? title} scene={scene ?? EMPTY_SCENE} story={story} /> : null}
      {story && !show3d ? <LiveIO story={story} /> : null}
      {story && show3d ? <LiveNotes story={story} /> : null}
      {!story && scene && show3d ? <Scene3DView scene={scene} /> : null}
      {story ? (
        <details className="lab-exact" open>
          <summary>Exact numbers and controls</summary>
          {children}
        </details>
      ) : children}
    </section>
  );
}
