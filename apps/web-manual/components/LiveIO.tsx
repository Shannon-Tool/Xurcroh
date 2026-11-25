"use client";

import { useEffect, useRef, type ReactNode } from "react";
import type { FlowForm } from "@/components/FlowRibbon";
import { DataViews, ProcessTrack } from "@/components/ProcessTrack";
import { QuestionPicker } from "@/components/QuestionPicker";
import { TypedText } from "@/components/TypedText";
import { useGeneration } from "@/lib/show/generation";
import { paintPixels } from "@/lib/show/paint";
import type { DataView, ProcessStep } from "@/lib/show/process";
import { useJourney, usePictureJourney } from "@/lib/show/question";

/** The stops on the way from a question to its answer. Each lab shows one of them. */
export const TRIP = [
  { id: "prepare", name: "Prepare" },
  { id: "split", name: "Split" },
  { id: "place", name: "Place" },
  { id: "send", name: "Send" },
  { id: "compute", name: "Compute" },
  { id: "check", name: "Check" },
  { id: "wipe", name: "Wipe" },
] as const;
export type TripStage = (typeof TRIP)[number]["id"];

/** A lab told as one trip: the reader's question goes in, the other party sees something, the model's answer comes out. */
export interface LiveStory {
  /** The whole lab in one or two plain sentences. */
  plain: string;
  /** Which stop on the trip this lab shows. */
  stage: TripStage;
  /** A written question, or a request for a picture. */
  kind?: "text" | "picture";
  /** What this lab takes from the question; shown under the question picker. */
  input: ReactNode;
  /** Heading of the middle card, for example "What one worker sees". */
  seenTitle: string;
  seen: ReactNode;
  /** This lab's own result, shown under the model's answer. */
  output: ReactNode;
  /** False when the step failed and no answer may be released. */
  released?: boolean;
  /** "warn" when what came out is a failure the lab is showing on purpose. */
  outputTone?: "good" | "warn";
  why: string;
  relies: string;
  limit: string;
  /** An attack the reader can switch on, and what happened to it. */
  attack?: { label: string; on: boolean; onToggle: () => void; outcome: string };
  /** One short line of what the other party sees, for the ribbon along the bottom of the stage. */
  screenSeen?: string;
  /**
   * The lab's computation as steps, each with its real data. With these, the stage shows the steps
   * as a row of cards and the answer is written only once the last step is done.
   */
  steps?: ProcessStep[];
  /** Index of the step being done now. */
  step?: number;
  /** True once the last step has been done: from here the output may appear. */
  done?: boolean;
  /** What the steps produced, shown above the answer (the scores that chose it, for example). */
  result?: DataView[];
  /** Playback speed, so that the answer is written at the pace the steps were played. */
  speed?: number;
  /** Goes to a step when the reader presses its card. */
  onStep?: (step: number) => void;
}

function AnswerPicture() {
  const picture = usePictureJourney();
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (canvas.current) paintPixels(canvas.current, picture.picture, picture.width, picture.height);
  }, [picture]);
  return <canvas ref={canvas} className="liveio-picture" width={picture.width} height={picture.height} role="img" aria-label={`The finished picture: ${picture.request}`} />;
}

/** The answer as it is written: one token for every protected product that has been computed and checked. */
function GeneratedAnswer({ story }: { story: LiveStory }) {
  const journey = useJourney();
  const generation = useGeneration(journey.id, story.done === true, story.speed ?? 1);
  return (
    <>
      {story.result ? <div className="liveio-produced"><DataViews views={story.result} /></div> : null}
      <div className="liveio-answer">
        <span className="liveio-tag">The model&apos;s answer</span>
        {/* The newest words stay in view: the box shows the end of the text while it is being written. */}
        <div className={generation.complete ? "liveio-writing complete" : "liveio-writing"}><p className="liveio-written">{generation.tokens.join("").trimStart()}{generation.complete ? null : <span className="show-caret" aria-hidden="true">▌</span>}</p></div>
      </div>
      <p className="liveio-count" data-testid="generation">
        Token {generation.tokens.length} of {generation.total}. Each one is its own product on two pieces; {generation.passed} of {generation.checks} results passed their check.
      </p>
    </>
  );
}

/** The output panel of a stage that shows its steps: nothing until the last step is done. */
function SteppedOutput({ story }: { story: LiveStory }) {
  const steps = story.steps ?? [];
  const at = Math.max(0, Math.min(steps.length - 1, story.step ?? 0));
  const released = story.released !== false;
  if (!story.done) {
    return (
      <>
        <p className="liveio-answer waiting"><span className="liveio-tag">The model&apos;s answer</span> Nothing yet.</p>
        <p className="liveio-count">It can only be written when all {steps.length} steps are done. Now at step {at + 1}: {steps[at]?.title}.</p>
      </>
    );
  }
  if (!released) {
    return (
      <>
        <p className="liveio-answer withheld"><span className="liveio-tag">The model&apos;s answer</span> No answer is released.</p>
        <div className="liveio-result">{story.output}</div>
      </>
    );
  }
  return (
    <>
      {story.kind === "picture" ? <div className="liveio-answer"><span className="liveio-tag">The model&apos;s picture</span><AnswerPicture /></div> : <GeneratedAnswer story={story} />}
      <div className="liveio-result">{story.output}</div>
    </>
  );
}

/** The three panels of a lab: what goes in, what the other party sees, what comes out. */
export function useLabPanels(story: LiveStory): { input: ReactNode; seen: ReactNode; output: ReactNode; forms: FlowForm[]; current: string[] } {
  const journey = useJourney();
  const picture = usePictureJourney();
  const kind = story.kind ?? "text";
  const released = story.released !== false;
  const question = kind === "picture" ? picture.request : journey.question;
  const seenLine = (story.screenSeen ?? "").split("\n")[0] ?? "";
  const forms: FlowForm[] = [
    { id: "question", label: "Your question", value: question.length > 22 ? `${question.slice(0, 21)}…` : question, tone: "good" },
    ...TRIP.map((stop): FlowForm => ({ id: stop.id, label: stop.name, value: stop.id === story.stage ? seenLine || "…" : "·", tone: stop.id === story.stage ? "accent" : "plain" })),
    { id: "answer", label: "Answer", value: !released ? "none released" : kind === "picture" ? "the picture" : `${journey.tokens.slice(0, 3).join("").trim()}…`, tone: released ? "good" : "warn" },
  ];
  return {
    forms,
    current: ["question", story.stage, "answer"],
    input: (
      <section className="stage-panel stage-in liveio-card liveio-in">
        <h3 className="stage-label">What goes in</h3>
        <div className="liveio-body">
          <QuestionPicker kind={kind} />
          {story.input}
        </div>
      </section>
    ),
    seen: (
      <section className="stage-panel stage-seen liveio-card liveio-seen">
        <h3 className="stage-label">{story.seenTitle}</h3>
        <div className="liveio-body">{story.seen}</div>
      </section>
    ),
    output: (
      <section className={story.outputTone === "warn" && (!story.steps || story.done) ? "stage-panel stage-out liveio-card liveio-out warn" : "stage-panel stage-out liveio-card liveio-out"}>
        <h3 className="stage-label">What comes out</h3>
        {story.steps ? <div className="liveio-body"><SteppedOutput story={story} /></div> : null}
        <div className="liveio-body" hidden={Boolean(story.steps)}>
          {story.steps ? null : !released ? <p className="liveio-answer withheld"><span className="liveio-tag">The model&apos;s answer</span> No answer is released.</p>
            : kind === "picture" ? <div className="liveio-answer"><span className="liveio-tag">The model&apos;s picture</span><AnswerPicture /></div>
              : <p className="liveio-answer"><span className="liveio-tag">The model&apos;s answer</span> <TypedText text={journey.answer} /></p>}
          {story.steps ? null : <div className="liveio-result">{story.output}</div>}
        </div>
      </section>
    ),
  };
}

/** What sits under a lab's stage: the attack to try, the three reasons, and the note on what is prepared. */
export function LiveNotes({ story }: { story: LiveStory }) {
  return (
    <>
      {story.attack ? (
        <div className={story.attack.on ? "liveio-attack on" : "liveio-attack"}>
          <button aria-pressed={story.attack.on} onClick={story.attack.onToggle}>Try to break it: {story.attack.label}</button>
          <p role="status">{story.attack.outcome}</p>
        </div>
      ) : null}
      <dl className="liveio-facts">
        <div><dt>Why this protects you</dt><dd>{story.why}</dd></div>
        <div><dt>It relies on</dt><dd>{story.relies}</dd></div>
        <div><dt>Where it stops</dt><dd>{story.limit}</dd></div>
      </dl>
      <p className="liveio-honest">The questions and answers are prepared: no language or image model runs in this manual. The numbers on this page are computed in your browser from the question you chose.</p>
    </>
  );
}

/** A lab without its 3D stage: the three panels side by side. Used where a lab is quoted on another page. */
export function LiveIO({ story }: { story: LiveStory }) {
  const panels = useLabPanels(story);
  return (
    <div className="liveio">
      <div className="liveio-row">
        {panels.input}
        <span className="liveio-arrow" aria-hidden="true" />
        {panels.seen}
        <span className="liveio-arrow" aria-hidden="true" />
        {panels.output}
      </div>
      {story.steps ? <ProcessTrack steps={story.steps} step={story.step ?? 0} onStep={story.onStep} /> : null}
      <LiveNotes story={story} />
    </div>
  );
}
