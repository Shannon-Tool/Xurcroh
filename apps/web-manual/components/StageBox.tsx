"use client";

import type { ReactNode } from "react";
import { FlowRibbon, type FlowForm } from "@/components/FlowRibbon";
import { Scene3DView } from "@/components/Scene3DView";
import type { Direction } from "@/lib/scene/orbit";
import type { Scene3D } from "@/lib/scene/types";

/**
 * The stage every animation in the manual is shown on: a dark 16:9 box with the 3D scene in the middle,
 * a column of panels on the left (what goes in), another on the right (what comes out or what others
 * see), and along the bottom the caption and the data in each form it takes.
 */
export function StageBox({ scene, direct, autoRotate, screens, screensVersion, left, right, caption, why, forms, current, ribbonLabel, label }: {
  scene: Scene3D;
  direct?: Direction;
  autoRotate?: boolean;
  screens?: Record<string, HTMLCanvasElement | null>;
  screensVersion?: number;
  left: ReactNode;
  right?: ReactNode;
  caption: string;
  why?: string;
  forms?: FlowForm[];
  current?: string[];
  ribbonLabel?: string;
  /** Names the stage for assistive technology. */
  label: string;
}) {
  // The panels sit on top of the 3D view; a press on them must not start a drag of the view underneath.
  const keep = (event: React.PointerEvent) => event.stopPropagation();
  const overlay = (
    <>
      <div className="stage-side" data-inset="left" onPointerDown={keep}>{left}</div>
      {right ? <div className="stage-side stage-right" data-inset="right" onPointerDown={keep}>{right}</div> : null}
      <div className="stage-foot" data-inset-bottom="">
        <div className="stage-caption">
          <p>{caption}</p>
          {why ? <p className="stage-why"><span>Why it matters</span> {why}</p> : null}
        </div>
        {forms && forms.length > 0 ? <FlowRibbon forms={forms} current={current ?? []} label={ribbonLabel} /> : null}
      </div>
    </>
  );
  return (
    <div className="stage" role="group" aria-label={label}>
      <Scene3DView scene={scene} variant="show" direct={direct} autoRotate={autoRotate} screens={screens} screensVersion={screensVersion} overlay={overlay} />
    </div>
  );
}
