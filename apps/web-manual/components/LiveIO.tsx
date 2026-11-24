import type { ReactNode } from "react";

/** A lab told as one story: what goes in, what the other party sees, what comes out, and why that matters. */
export interface LiveStory {
  /** The whole lab in one or two plain sentences. */
  plain: string;
  /** The reader's input; normally an editable control. */
  input: ReactNode;
  /** Heading of the middle card, for example "What one worker sees". */
  seenTitle: string;
  seen: ReactNode;
  output: ReactNode;
  /** "warn" when what came out is a failure the lab is showing on purpose. */
  outputTone?: "good" | "warn";
  why: string;
  relies: string;
  limit: string;
  /** An attack the reader can switch on, and what happened to it. */
  attack?: { label: string; on: boolean; onToggle: () => void; outcome: string };
  /** Short texts for the three screens in the 3D view. */
  screens?: { input: string; seen: string; output: string };
}

export function LiveIO({ story }: { story: LiveStory }) {
  return (
    <div className="liveio">
      <div className="liveio-row">
        <section className="liveio-card liveio-in">
          <h3>What goes in</h3>
          <div className="liveio-body">{story.input}</div>
        </section>
        <span className="liveio-arrow" aria-hidden="true" />
        <section className="liveio-card liveio-seen">
          <h3>{story.seenTitle}</h3>
          <div className="liveio-body">{story.seen}</div>
        </section>
        <span className="liveio-arrow" aria-hidden="true" />
        <section className={story.outputTone === "warn" ? "liveio-card liveio-out warn" : "liveio-card liveio-out"}>
          <h3>What comes out</h3>
          <div className="liveio-body">{story.output}</div>
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
    </div>
  );
}
