"use client";

import { ScrambleText } from "@/components/ScrambleText";
import { grouped, hexText, numbersText, type DataView, type ProcessStep } from "@/lib/show/process";

const MARK = { done: "✓", now: "●", todo: "·", bad: "✗" } as const;

/** A step's data, drawn: numbers, bytes, bars, lines or a worked sum. Values morph when they change. */
export function DataViews({ views }: { views: DataView[] }) {
  return (
    <>
      {views.map((view, index) => {
        const label = "label" in view && view.label ? <span className="data-label">{view.label}</span> : null;
        if (view.kind === "numbers" && view.mark) {
          const show = view.show ?? 12;
          return (
            <div key={index} className="data data-cells">
              {label}
              <span className="data-row">
                {view.values.slice(0, show).map((value, at) => <span key={at} className={view.mark!.includes(at) ? "data-cell on" : "data-cell"}>{String(value)}</span>)}
                {view.values.length > show ? <span className="data-more">… {view.values.length} in all</span> : null}
              </span>
            </div>
          );
        }
        if (view.kind === "numbers") return <div key={index} className="data">{label}<ScrambleText className="data-value" text={numbersText(view.values, view.show)} /></div>;
        if (view.kind === "bytes") {
          return (
            <div key={index} className={view.readable ? "data data-bytes readable" : "data data-bytes"}>
              {label}<ScrambleText className="data-value" text={hexText(view.bytes, view.show)} />
              {view.readable ? <span className="data-readable">“{view.readable}”</span> : null}
            </div>
          );
        }
        if (view.kind === "bars") {
          const most = Math.max(1, ...view.values.map((value) => Math.abs(value)));
          // `top: -1` lights no bar: a count of work has no winner.
          const top = view.top ?? view.values.indexOf(Math.max(...view.values));
          return (
            <div key={index} className="data data-bars">
              {label}
              {view.values.map((value, at) => (
                <span key={at} className={at === top ? "data-bar top" : "data-bar"}>
                  <span className="data-bar-name">{view.names?.[at] ?? at + 1}</span>
                  <span className="data-bar-fill" style={{ width: `${Math.max(2, (Math.abs(value) / most) * 100)}%` }} />
                  <span className="data-bar-value">{grouped(value)}</span>
                </span>
              ))}
            </div>
          );
        }
        if (view.kind === "rows") {
          return (
            <div key={index} className={view.rows.length >= 3 ? "data data-rows two" : "data data-rows"}>
              {view.rows.map((row) => (
                <span key={row.name} className={`data-line ${row.state ?? "done"}`}>
                  <span className="data-mark" aria-hidden="true">{MARK[row.state ?? "done"]}</span>
                  <span className="data-name">{row.name}</span>
                  <span className="data-value">{row.value}</span>
                </span>
              ))}
            </div>
          );
        }
        return <div key={index} className={view.kind === "sum" ? "data data-sum" : "data data-text"}>{view.text}</div>;
      })}
    </>
  );
}

/**
 * The computation as a row of cards, one per step, each holding the data as it is after that step.
 * Steps already done keep their values, the current one is lit, and the ones to come are empty: the
 * reader watches the input change form, step by step, until it is the output.
 */
export function ProcessTrack({ steps, step, why, onStep, label = "The steps from what goes in to what comes out" }: {
  steps: ProcessStep[];
  step: number;
  why?: string;
  /** Lets the reader go to a step by pressing its card. */
  onStep?: (step: number) => void;
  label?: string;
}) {
  const at = Math.max(0, Math.min(steps.length - 1, step));
  const current = steps[at];
  if (!current) return null;
  return (
    <div className="track" role="group" aria-label={label}>
      <ol className="track-steps">
        {steps.map((entry, index) => {
          const state = index < at ? "done" : index === at ? "now" : "todo";
          return (
            <li key={entry.id} className={`track-step ${state} track-${entry.tone ?? "plain"}`} aria-current={state === "now" ? "step" : undefined}>
              {onStep
                ? <button type="button" className="track-head track-go" onClick={() => onStep(index)} aria-label={`Step ${index + 1}: ${entry.title}`}><span className="track-n">{index + 1}</span> <span className="track-title">{entry.title}</span></button>
                : <span className="track-head"><span className="track-n">{index + 1}</span> <span className="track-title">{entry.title}</span></span>}
              <div className="track-data">{state === "todo" ? <span className="track-empty" aria-label="not yet">—</span> : <DataViews views={entry.data} />}</div>
            </li>
          );
        })}
      </ol>
      <div className="track-now">
        <p role="status"><span className="track-count">Step {at + 1} of {steps.length}</span> {current.does}</p>
        {why ? <p className="stage-why"><span>Why it matters</span> {why}</p> : null}
      </div>
    </div>
  );
}
