"use client";

import { useEffect, useRef, type ReactNode } from "react";
import { QuestionPicker } from "@/components/QuestionPicker";
import { paintPixels } from "@/lib/show/paint";
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
  /** Short text for the middle screen in the 3D view. */
  screenSeen?: string;
}

function AnswerPicture() {
  const picture = usePictureJourney();
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (canvas.current) paintPixels(canvas.current, picture.picture, picture.width, picture.height);
  }, [picture]);
  return <canvas ref={canvas} className="liveio-picture" width={picture.width} height={picture.height} role="img" aria-label={`The finished picture: ${picture.request}`} />;
}

export function LiveIO({ story }: { story: LiveStory }) {
  const journey = useJourney();
  const kind = story.kind ?? "text";
  const released = story.released !== false;
  return (
    <div className="liveio">
      <ol className="trip" aria-label="The trip from your question to the answer">
        <li className="trip-end">Your question</li>
        {TRIP.map((stop) => (
          <li key={stop.id} className={stop.id === story.stage ? "trip-here" : undefined} aria-current={stop.id === story.stage ? "step" : undefined}>
            {stop.name}{stop.id === story.stage ? <span> · this lab</span> : null}
          </li>
        ))}
        <li className="trip-end">Answer</li>
      </ol>
      <div className="liveio-row">
        <section className="liveio-card liveio-in">
          <h3>What goes in</h3>
          <div className="liveio-body">
            <QuestionPicker kind={kind} />
            {story.input}
          </div>
        </section>
        <span className="liveio-arrow" aria-hidden="true" />
        <section className="liveio-card liveio-seen">
          <h3>{story.seenTitle}</h3>
          <div className="liveio-body">{story.seen}</div>
        </section>
        <span className="liveio-arrow" aria-hidden="true" />
        <section className={story.outputTone === "warn" ? "liveio-card liveio-out warn" : "liveio-card liveio-out"}>
          <h3>What comes out</h3>
          <div className="liveio-body">
            {!released ? <p className="liveio-answer withheld"><span>The model&apos;s answer</span> No answer is released.</p>
              : kind === "picture" ? <div className="liveio-answer"><span>The model&apos;s picture</span><AnswerPicture /></div>
                : <p className="liveio-answer"><span>The model&apos;s answer</span> {journey.answer}</p>}
            <div className="liveio-result">{story.output}</div>
          </div>
        </section>
      </div>
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
    </div>
  );
}
