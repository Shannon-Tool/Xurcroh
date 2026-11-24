"use client";

import type { ReactNode } from "react";
import { FlowRibbon, type FlowForm } from "@/components/FlowRibbon";
import { Scene3DView } from "@/components/Scene3DView";
import type { Direction } from "@/lib/scene/orbit";
import type { Scene3D } from "@/lib/scene/types";

/**
 * The stage every animation in the manual is shown on: a dark 16:9 box with the 3D scene, what goes in
 * and what comes out on the left, and along the bottom the caption and the data in each form it takes.
 */
export function StageBox({ scene, direct, autoRotate, screens, screensVersion, input, output, corner, caption, why, forms, current, label }: {
  scene: Scene3D;
  direct?: Direction;
  autoRotate?: boolean;
  screens?: Record<string, HTMLCanvasElement | null>;
  screensVersion?: number;
  /** "In": what the reader puts in. */
  input: ReactNode;
  /** "Out": what comes back. */
  output: ReactNode;
  /** Small read-outs for the top right corner. */
  corner?: ReactNode;
  caption: string;
  why?: string;
  forms?: FlowForm[];
  current?: string[];
  /** Names the stage for assistive technology. */
  label: string;
}) {
  const overlay = (
    <>
      <div className="stage-side" data-inset="left">
        {input}
        {output}
      </div>
      {corner ? <div className="stage-corner">{corner}</div> : null}
      <div className="stage-foot" data-inset-bottom="">
        <div className="stage-caption">
          <p>{caption}</p>
          {why ? <p className="stage-why"><span>Why it matters</span> {why}</p> : null}
        </div>
        {forms && forms.length > 0 ? <FlowRibbon forms={forms} current={current ?? []} /> : null}
      </div>
    </>
  );
  return (
    <div className="stage" role="group" aria-label={label}>
      <Scene3DView scene={scene} variant="show" direct={direct} autoRotate={autoRotate} screens={screens} screensVersion={screensVersion} overlay={overlay} />
    </div>
  );
}
